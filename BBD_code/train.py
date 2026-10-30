import models
import os
import csv
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import dataset
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import losses
import numpy as np
import semi

from collections import defaultdict


class AverageLossArtifacts:
    def __init__(self, tracker, base_dir, predict_result):
        self._tracker = tracker
        self.base_dir = base_dir
        self.predict_result = predict_result
        self.split_time = tracker.split_time

    def write_split_results(self, clean_pool, poison_pool, threshold):
        self._tracker._write_split_results(
            self.base_dir,
            clean_pool,
            poison_pool,
            threshold,
        )

def train(device, dataset_name, train_dataset,num_workers, model_config,split_time,
                                            save_model=True,save_folder = None):
    model_name = model_config.model_name
    batch_size = model_config.batch_size
    epoch_num = model_config.epoch_num
    lr = model_config.lr
    weight_decay = model_config.weight_decay
    os.makedirs(save_folder, exist_ok=True)
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True
    )
    model = models.get_model(model_name=model_name,num_classes=dataset.get_num_classes(dataset_name, split_time))
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    criterion = losses.SCELoss( beta=0.5,num_classes=dataset.get_num_classes(dataset_name, split_time))
    scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num)

    progress_bar = tqdm(range(epoch_num))
    for epoch in progress_bar:
        model.train()
        total_loss = 0
        step = 0
        for image_name, inputs, labels in train_data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            step += 1
            loss = criterion(outputs, labels)
            loss = loss.mean()
            total_loss += loss*inputs.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    progress_bar.close()
    if (save_model):
        torch.save(model.state_dict(),os.path.join(save_folder, f'model_ID_{split_time}.pth.tar'))
    return model

def LGA_train(
    device,
    dataset_name,
    train_dataset,
    num_workers,
    model_config,
    split_time,
    save_model=True,
    save_folder=None,
    clean_data_names=None,
    poison_data_names=None,
    attack_type=None,
    logger=None,
    dataset_root=None,
    poison_target=0,
    train_data = None
):

    model_name = model_config.model_name
    batch_size = model_config.batch_size
    epoch_num = model_config.epoch_num
    lr = model_config.lr
    weight_decay = model_config.weight_decay
    os.makedirs(save_folder, exist_ok=True)
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True
    )

    model = models.get_model(model_name=model_name,num_classes=dataset.get_num_classes(dataset_name,split_time))
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    criterion = losses.LGALoss(0.5,criterion).to(device)

    tracker = None

    if (
        'no_defense' in split_time or 'PTM' in split_time
        and clean_data_names is not None
        and poison_data_names is not None
        and attack_type is not None
    ):
        tracker = LabelChangeTracker(
            train_dataset=dataset.get_train_dataset(
                clean_data_names,
                poison_data_names,
                train_data,
                'no_defense',
                dataset_name,
            ),
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            dataset_name=dataset_name,
            attack_type=attack_type,
            split_time=split_time,
            clean_names=clean_data_names,
            poison_names=poison_data_names,
            logger=logger,
            total_epochs=epoch_num,
            dataset_root=dataset_root,
            poison_target=poison_target,
        )

    progress_bar = tqdm(range(epoch_num))
    for epoch in progress_bar:
        model.train()
        total_loss = 0
        step = 0
        for image_name, inputs, labels in train_data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            step += 1
            loss = criterion(outputs, labels)
            total_loss += loss*inputs.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if tracker is not None and epoch >= 19:
            tracker.update(model, epoch)
    progress_bar.close()
    artifacts = None
    if tracker is not None:
        artifacts = tracker.finalize()
    if (save_model):
        torch.save(model.state_dict(),os.path.join(save_folder, f'model_ID_{split_time}.pth.tar'))
    return model, artifacts



class MemoryDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, image, label = self.samples[idx]
        if self.transform is not None:
            image = self.transform(image)
        return name, image, label


class LabelChangeTracker:

    def __init__(
        self,
        train_dataset,
        device,
        batch_size,
        num_workers,
        dataset_name,
        attack_type,
        split_time,
        clean_names,
        poison_names,
        logger=None,
        output_root="avg_losses",
        total_epochs=None,
        dataset_root=None,
        poison_target=0,
    ):
        self.device = device
        self.logger = logger
        self.dataset_name = dataset_name
        self.attack_type = attack_type
        self.split_time = split_time
        self.clean_names = set(clean_names or [])
        self.poison_names = set(poison_names or [])
        self.output_root = output_root
        self.total_epochs = total_epochs
        self.dataset_root = dataset_root
        self.poison_target = poison_target
        self.clean_history = []
        self.poison_history = []
        self.epoch_summaries = []
        self.sample_loss_sums = defaultdict(float)
        self.sample_norm_sums = defaultdict(float)
        self.sample_eval_counts = defaultdict(int)
        self.sample_last_pred = {}
        self.sample_label_map = {}
        self.sample_order = []

        eval_samples = self._extract_samples(train_dataset)
        eval_transform = dataset.get_train_transforms(dataset_name, is_enhance=False)
        self.eval_dataset = MemoryDataset(eval_samples, transform=eval_transform)
        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
        )

        for name, _, label in eval_samples:
            if name not in self.sample_label_map:
                self.sample_label_map[name] = int(label)
                self.sample_order.append(name)

    def _extract_samples(self, ds):
        if isinstance(ds, ConcatDataset):
            samples = []
            for sub in ds.datasets:
                samples.extend(self._extract_samples(sub))
            return samples
        if isinstance(ds, MemoryDataset):
            return list(ds.samples)
        if hasattr(ds, "image_names") and hasattr(ds, "images") and hasattr(ds, "labels"):
            return list(zip(ds.image_names, ds.images, ds.labels))
        samples = []
        for idx in range(len(ds)):
            name, image, label = ds[idx]
            samples.append((name, image, label))
        return samples

    def update(self, model, epoch_idx):
        was_training = model.training
        model.eval()
        epoch = epoch_idx + 1
        epoch_records = []
        criterion = losses.SCELoss(num_classes=10,
                               reduction='none')
        with torch.no_grad():
            for names, inputs, labels in self.eval_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                logits = model(inputs)
                loss = criterion(logits, labels)
                preds = logits.argmax(dim=1)
                for name, loss_val, pred_val in zip(
                    names,
                    loss.tolist(),
                    preds.tolist(),
                ):
                    epoch_records.append(
                        (name, float(loss_val), int(pred_val))
                    )
        if was_training:
            model.train()
        max_loss = max((loss for _, loss, _ in epoch_records), default=0.0)
        clean_losses = []
        clean_norm_losses = []
        poison_losses = []
        poison_norm_losses = []
        is_last_epoch = (
            self.total_epochs is not None and epoch == self.total_epochs
        )
        for name, loss_val, pred_val in epoch_records:
            norm_loss = loss_val / max_loss if max_loss > 0 else 0.0
            # norm_loss = loss_val
            self.sample_loss_sums[name] += loss_val
            self.sample_norm_sums[name] += norm_loss
            self.sample_eval_counts[name] += 1
            self.sample_last_pred[name] = pred_val
            avg_loss = (
                self.sample_loss_sums[name] / self.sample_eval_counts[name]
                if is_last_epoch and self.sample_eval_counts[name] > 0
                else None
            )
            avg_norm = (
                self.sample_norm_sums[name] / self.sample_eval_counts[name]
                if is_last_epoch and self.sample_eval_counts[name] > 0
                else None
            )
            record = (epoch, name, loss_val, norm_loss, avg_loss, avg_norm)
            if name in self.clean_names:
                self.clean_history.append(record)
                clean_losses.append(loss_val)
                clean_norm_losses.append(norm_loss)
            elif name in self.poison_names:
                self.poison_history.append(record)
                poison_losses.append(loss_val)
                poison_norm_losses.append(norm_loss)
        clean_avg = float(np.mean(clean_losses)) if clean_losses else 0.0
        poison_avg = float(np.mean(poison_losses)) if poison_losses else 0.0
        clean_norm_avg = float(np.mean(clean_norm_losses)) if clean_norm_losses else 0.0
        poison_norm_avg = float(np.mean(poison_norm_losses)) if poison_norm_losses else 0.0
        self.epoch_summaries.append(
            (
                epoch,
                max_loss,
                clean_avg,
                poison_avg,
                clean_norm_avg,
                poison_norm_avg,
            )
        )
        message = (
            f"[LabelChangeTracker] Epoch {epoch}: max_loss={max_loss:.6f}, "
            f"avg_clean_loss={clean_avg:.6f}, avg_poison_loss={poison_avg:.6f}"
        )
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(message)
        return max_loss, clean_avg, poison_avg

    def finalize(self):
        base_dir = os.path.join(
            self.output_root,
            self.dataset_name,
            self.attack_type,
        )
        clean_dir = os.path.join(base_dir, "trainSet_clean_list")
        poison_dir = os.path.join(base_dir, "trainSet_poisoned_list")
        os.makedirs(clean_dir, exist_ok=True)
        os.makedirs(poison_dir, exist_ok=True)

        self._write_records(
            clean_dir,
            self.clean_history,
            f"sample_losses_{self.split_time}.csv",
        )
        self._write_records(
            poison_dir,
            self.poison_history,
            f"sample_losses_{self.split_time}.csv",
        )
        self._write_average_records(
            clean_dir,
            self.clean_names,
            f"average_sample_losses_{self.split_time}.csv",
        )
        self._write_average_records(
            poison_dir,
            self.poison_names,
            f"average_sample_losses_{self.split_time}.csv",
        )
        self._write_epoch_summary(
            base_dir,
            f"epoch_loss_summary_{self.split_time}.csv",
        )

        avg_predict_result = self._build_average_predict_result()
        if avg_predict_result:
            return AverageLossArtifacts(self, base_dir, avg_predict_result)
        return None

    def _write_records(self, folder, records, filename):
        path = os.path.join(folder, filename)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "image_name",
                    "loss",
                    "normalized_loss",
                    "average_loss",
                    "average_normalized_loss",
                ]
            )
            for row in records:
                epoch, name, loss, norm_loss, avg_loss, avg_norm = row
                writer.writerow(
                    [
                        epoch,
                        name,
                        f"{loss:.8f}",
                        f"{norm_loss:.8f}",
                        "" if avg_loss is None else f"{avg_loss:.8f}",
                        "" if avg_norm is None else f"{avg_norm:.8f}",
                    ]
                )

    def _write_average_records(self, folder, names, filename):
        path = os.path.join(folder, filename)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "image_name",
                    "average_loss",
                    "average_normalized_loss",
                    "label",
                    "last_prediction",
                ]
            )
            for name in sorted(names):
                count = self.sample_eval_counts.get(name, 0)
                if count == 0:
                    continue
                avg_loss = self.sample_loss_sums[name] / count
                avg_norm = self.sample_norm_sums[name] / count
                label = self.sample_label_map.get(name, -1)
                pred = self.sample_last_pred.get(name, -1)
                writer.writerow(
                    [
                        name,
                        f"{avg_loss:.8f}",
                        f"{avg_norm:.8f}",
                        label,
                        pred,
                    ]
                )

    def _write_epoch_summary(self, folder, filename):
        path = os.path.join(folder, filename)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "max_loss",
                    "avg_clean_loss",
                    "avg_poison_loss",
                    "avg_clean_normalized_loss",
                    "avg_poison_normalized_loss",
                ]
            )
            writer.writerows(self.epoch_summaries)

    def _build_average_predict_result(self):
        if not self.attack_type:
            attack = ""
        else:
            attack = self.attack_type
        clean_entries = []
        poison_entries = []
        for name in self.sample_order:
            count = self.sample_eval_counts.get(name, 0)
            if count == 0:
                continue
            avg_norm = (
                self.sample_norm_sums[name] / count
                if count > 0
                else 0.0
            )
            label = int(self.sample_label_map.get(name, -1))
            pred = int(self.sample_last_pred.get(name, -1))
            record = [attack, label, float(avg_norm), name, pred]
            if name in self.clean_names:
                clean_entries.append(record)
            elif name in self.poison_names:
                poison_entries.append(record)
        return clean_entries + poison_entries

    def _write_split_results(self, base_dir, clean_pool, poison_pool, threshold):
        split_dir = os.path.join(base_dir, "average_loss_split")
        os.makedirs(split_dir, exist_ok=True)
        clean_path = os.path.join(split_dir, f"clean_{self.split_time}.txt")
        poison_path = os.path.join(split_dir, f"poison_{self.split_time}.txt")
        meta_path = os.path.join(split_dir, f"summary_{self.split_time}.txt")
        with open(clean_path, "w") as f:
            for name in clean_pool:
                f.write(f"{name}\n")
        with open(poison_path, "w") as f:
            for name in poison_pool:
                f.write(f"{name}\n")
        with open(meta_path, "w") as f:
            f.write(f"threshold,{threshold}\n")
            f.write(f"clean_count,{len(clean_pool)}\n")
            f.write(f"poison_count,{len(poison_pool)}\n")

def semi_train(
    device,
    dataset_name,
    x_dataset,
    u_dataset,
    num_workers,
    model_config,
    split_time,
    save_model,
    save_folder,
    logger,
    **kwargs,
):
    model_name = model_config.model_name
    batch_size = model_config.batch_size
    epoch_num = model_config.epoch_num
    lr = model_config.lr
    weight_decay = model_config.weight_decay
    os.makedirs(save_folder, exist_ok=True)
    xloader = DataLoader(
        x_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True
    )
    uloader = DataLoader(
        u_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        persistent_workers=True,
    )
    num_classes = dataset.get_num_classes(dataset_name, split_time)
    model = models.get_model(model_name=model_name,
                             num_classes=num_classes)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    criterion = losses.MixMatchLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num)
    progress_bar = tqdm(range(epoch_num))

    for epoch in progress_bar:
        semi.mixmatch_train(
            model,
            xloader,
            uloader,
            criterion,
            optimizer,
            epoch,
            num_classes,
            **kwargs,
        )
        scheduler.step()
    progress_bar.close()
    del uloader
    torch.cuda.empty_cache()
    if (save_model):
        torch.save(model.state_dict(),
                   os.path.join(save_folder, f'model_ID_{split_time}.pth.tar'))
        return model