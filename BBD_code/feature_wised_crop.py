import torch
import numpy as np
import cv2
from PIL import Image

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
def integrated_gradients_batch(model, inputs, target_classes, baseline=None, steps=100):
    if baseline is None:
        baseline = torch.zeros_like(inputs)
    else:
        baseline = torch.mean(inputs, dim=0, keepdim=True)

    grads = torch.zeros_like(inputs)
    device = inputs.device
    alphas = torch.linspace(0, 1, steps, device=device)

    for alpha in alphas:
        interpolated = baseline + alpha * (inputs - baseline)
        interpolated.requires_grad_(True)
        model.zero_grad()
        output = model(interpolated)  # logits
        selected = output[torch.arange(inputs.size(0), device=device), target_classes]
        selected.sum().backward()
        grads += interpolated.grad.detach()

    integrated_grads = (inputs - baseline) * grads / steps
    return integrated_grads


def _collate(batch):
    names, images, labels = [], [], []
    for n, img, label in batch:
        names.append(n)
        images.append(img)
        labels.append(label)
    return names, images, torch.tensor(labels)


def crop_dataset(
    ds,
    model,
    preprocess,
    device,
    transform,
    threshold = 0.95,
    batch_size=32,
    num_workers=4,
    use_pred_label=False,
    crop_high_activation=True,
    logger=None,
):
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
    )

    samples = []
    if threshold is None or threshold > 1:
        if logger:
            logger.info(f"threshold {threshold} > 1, skip cropping")
        for names, images, labels in loader:
            for name, image, label in zip(names, images, labels.tolist()):
                samples.append((name, image, label))
        return MemoryDataset(samples, transform=transform)

    if logger:
        if crop_high_activation:
            logger.info(f"cropping pixels with activation above {threshold}")
        else:
            logger.info(f"cropping pixels with activation below {threshold}")
    for names, images, labels in loader:
        inputs = torch.stack([preprocess(img) for img in images]).to(device)
        if use_pred_label:
            with torch.no_grad():
                preds = model(inputs).argmax(dim=1)
            target = preds
        else:
            target = labels.to(device)

        grads_batch = integrated_gradients_batch(model, inputs, target)
        grads_batch = grads_batch.cpu().numpy()

        for name, image, label, grads in zip(names, images, labels.tolist(), grads_batch):
            grads = np.abs(grads)
            if grads.ndim == 3:
                grads = grads.sum(axis=0)
            if grads.shape != (image.size[1], image.size[0]):
                grads = cv2.resize(grads, (image.size[0], image.size[1]))
            percentile = np.percentile(grads, 100 * threshold)
            if crop_high_activation:
                mask = grads > percentile
            else:
                mask = grads < percentile

            image_np = np.array(image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            cropped_bgr = image_bgr.copy()
            cropped_bgr[mask] = 0
            inpaint_mask = mask.astype(np.uint8)
            inpainted_bgr = cv2.inpaint(cropped_bgr, inpaint_mask, 3, flags=cv2.INPAINT_NS)
            inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
            samples.append((name, Image.fromarray(inpainted_rgb), label))
    return MemoryDataset(samples, transform=transform)

