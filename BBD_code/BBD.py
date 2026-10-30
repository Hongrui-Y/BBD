import torch
import csv


import dataset
import predict
import models
import split
import train
from model_config import model_config
from torch.utils.data import ConcatDataset
import argparse
import utils
import os
import shutil
from feature_wised_crop import crop_dataset
from torchvision import transforms
from train import MemoryDataset


def get_preprocess(dataset_name):
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        dataset.getNormal(dataset_name)
    ])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='./config/baseline_bbd.yaml', help='Path to the config file.')
parser.add_argument('-r','--random_seed', type=int, default=0, help='random seeds')

opts = parser.parse_args()

config = utils.get_config(opts.config)
model_name = config['model_name']
num_workers = config['num_workers']
dataset_root = config['dataset_root']
dataset_name = config['dataset_name']
if opts.random_seed :
    random_seed = opts.random_seed
else:
    random_seed = config['random_seed']
save_root = f"{config['save_root']}"

attack_type_list = config['attack_type_list']
log_dir = config['log_dir']

if not os.path.exists(save_root):
    os.makedirs(save_root)
config_filename = os.path.basename(opts.config)
shutil.copy(opts.config, os.path.join(save_root, config_filename))
logger = utils.setup_logger(log_dir)

for key, value in config.items():
    logger.info(f"{key}: {value}")

csv_file_name = os.path.join(save_root, 'results.csv')
logger.info(f"result_csv_name:{csv_file_name}")

with open(csv_file_name, 'a', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(
        ['dataset_name', 'attack_type', 'ACC', 'ASR', 'split_time', 'correct_clean',
         'wrong_poisoned', 'wrong_clean', 'correct_poisoned','random_seed'])


split_times = ["no_defense","CTMv1","PTMv1","CTMv2","PTMv2","CTMv3","PTMv3","semi"]

split_result_dict = {time: {"clean_pool_image_names": [], "poison_pool_image_names": []} for time in split_times}
for attack_type in attack_type_list:
    if random_seed is not None:
        utils.fix_random(random_seed)
    poison_target = 0
    logger.info(f"-------------------- Start defending {attack_type} --------------------")
    logger.info(f"attack_type:{attack_type}")
    logger.info(f"dataset_name:{dataset_name}")
    train_data_path = os.path.join(dataset_root,f'{attack_type}/{dataset_name}')

    clean_data = dataset.CustomDataset(
        data_folder=train_data_path,
        file_name='trainSet_clean_list',
        dataset_name=dataset_name,
    )
    clean_data_names = clean_data.get_names()
    poison_data = dataset.CustomDataset(
        data_folder=train_data_path,
        file_name='trainSet_poisoned_list',
        dataset_name=dataset_name,
        label_flipping='poison',
        target=poison_target
    )
    poison_data_names = poison_data.get_names()
    poison_real_dataset = dataset.CustomDataset(
        data_folder=train_data_path,
        file_name='trainSet_poisoned_list',
        dataset_name=dataset_name,
    )
    train_data = ConcatDataset([clean_data, poison_data])

    clean_test_data = dataset.CustomDataset(
        data_folder=train_data_path,
        file_name='testSetClear_labels',
        dataset_name=dataset_name,
        split_time='test'
    )
    poison_test_data = dataset.CustomDataset(
        data_folder=train_data_path,
        file_name='testSetPoisoned_poisoned_list',
        dataset_name=dataset_name,
        split_time='test',
        label_flipping='poison',
        target=poison_target
    )
    clean_pool_image_names = None
    poison_pool_image_names = None
    for split_time in split_times:
        avg_loss_artifacts = None
        if random_seed is not None:
            utils.fix_random(random_seed)

        save_folder = os.path.join(save_root, f'backdoor-defense-model/{dataset_name}/{attack_type}/{model_name}')
        if 'semi' in split_time:
            print(f'len{len(clean_test_data)},{len(poison_test_data)}')
            clean_test_dataset, poison_test_dataset = dataset.get_test_dataset(
                clean_test_data,
                poison_test_data,
                dataset_name
            )

        else:
            clean_test_dataset, poison_test_dataset = dataset.get_test_dataset(
                clean_data,
                poison_data,
                dataset_name
            )
        logger.info(f"clean test data length{len(clean_test_dataset)}")
        logger.info(f"poison test data length{len(poison_test_dataset)}")
        logger.info("---------------------------------------------------")
        logger.info(split_time)


        if ('PTM' in split_time):
            clean_pool_image_names = split_result_dict[split_times[split_times.index(split_time) - 2]]['clean_pool_image_names']
            poison_pool_image_names = split_result_dict[split_times[split_times.index(split_time) - 1]]['poison_pool_image_names']
        elif ('CTM' in split_time or 'semi' in split_time):
            clean_pool_image_names = split_result_dict[split_times[split_times.index(split_time) - 1]][
                'clean_pool_image_names']
            poison_pool_image_names = split_result_dict[split_times[split_times.index(split_time) - 1]][
                'poison_pool_image_names']
        else:
            clean_pool_image_names = clean_data_names
            poison_pool_image_names = poison_data_names


        logger.info(f"clean_pool_image_names len:{len(clean_pool_image_names)}")
        if poison_pool_image_names is not None:
            logger.info(f"poison_pool_image_names len:{len(poison_pool_image_names)}")
        if 'semi' not in split_time:
            train_dataset = dataset.get_train_dataset(
                clean_pool_image_names,
                poison_pool_image_names,
                train_data,
                split_time,
                dataset_name,
            )
            logger.info(f"train_dataset_length:{len(train_dataset)}")


        if ('PTM' in split_time or 'no_defense' in split_time):
            m_config = model_config(
                model_name=model_name,
                batch_size=256,
                epoch_num=30,
                lr=0.01,
                weight_decay=0.0001,
                description=f'{dataset_name}_{attack_type}_{split_time}',
            )
        elif (f'semi' in split_time):
            m_config = model_config(
                model_name=model_name,
                batch_size=128,
                epoch_num=200,
                lr=0.01,
                weight_decay=5e-4,
                description=f'{dataset_name}_{attack_type}_{split_time}',
            )
        else:
            m_config = model_config(
                model_name=model_name,
                batch_size=128,
                epoch_num=100,
                lr=0.01,
                weight_decay=5e-4,
                description=f'{dataset_name}_{attack_type}_{split_time}',
            )
        logger.info(m_config.__str__())
        if ('semi' in split_time):
            preprocess = get_preprocess(dataset_name)
            base_tf = dataset.get_train_transforms(dataset_name, is_enhance=True)
            plain_tf = dataset.get_train_transforms(dataset_name, is_enhance=False)
            num_classes = dataset.get_num_classes(dataset_name, split_time)
            ptm_model = models.get_model(model_name=model_name,
                                         num_classes=dataset.get_num_classes(dataset_name, 'PTMv3'))
            ptm_model.to(device)
            ptm_path = os.path.join(
                save_root,
                f'backdoor-defense-model/{dataset_name}/{attack_type}/{model_name}/',
                f'model_ID_PTMv3.pth.tar',
            )
            logger.info(f"loading PTMv3 model from {ptm_path}")
            ptm_model.load_state_dict(torch.load(ptm_path, map_location=device))
            ptm_model.eval()

            dc = dataset.SubsetFromIdx(
                clean_pool_image_names,
                train_data,
                base_tf,
                num_classes=num_classes,
            )
            dp_trap = dataset.SubsetFromIdx(
                poison_pool_image_names,
                train_data,
                plain_tf,
                num_classes=num_classes,
                label_flipping='trap',
            )
            dp_poison_raw = dataset.SubsetFromIdx(
                poison_pool_image_names,
                train_data,
                None,
                num_classes=num_classes,
                label_flipping='poison',
                target=poison_target
            )

            logger.info(f"cropping Dp' length{len(dp_poison_raw)}")
            dp_prime = crop_dataset(
                dp_poison_raw,
                ptm_model,
                preprocess,
                device,
                base_tf,
                batch_size=128,
                num_workers=num_workers,
                crop_high_activation=True,
                logger=logger,
            )

            x_dataset = dataset.MixMatchLabeledDataset(
                ConcatDataset([
                    dc,
                    dp_trap,
                ])
            )
            u_dataset = dataset.MixMatchUnLabeledDataset(
                MemoryDataset(dp_prime.samples, base_tf)
            )

            logger.info(f"x_dataset length:{len(x_dataset)}")
            logger.info(f"u_dataset length:{len(u_dataset)}")

            model = train.semi_train(
                device,
                dataset_name,
                x_dataset,
                u_dataset,
                num_workers,
                m_config,
                split_time,
                True,
                save_folder,
                logger,
                **config["semi"]["mixmatch"]
            )
        elif('no_defense' in split_time or 'PTM' in split_time):
            model, avg_loss_artifacts = train.LGA_train(
                device=device,
                dataset_name=dataset_name,
                train_dataset=train_dataset,
                num_workers=num_workers,
                model_config=m_config,
                split_time=split_time,
                save_model=True,
                save_folder=save_folder,
                clean_data_names=clean_data_names,
                poison_data_names=poison_data_names,
                attack_type=attack_type,
                logger=logger,
                dataset_root=dataset_root,
                poison_target=poison_target,
                train_data = train_data,
            )
        else:
            model = train.train(
                device=device,
                dataset_name=dataset_name,
                train_dataset=train_dataset,
                num_workers=num_workers,
                model_config=m_config,
                split_time=split_time,
                save_model=True,
                save_folder=save_folder,
            )
        predict_result, acc, asr = predict.predict(
            model=model,
            clean_test_dataset=clean_test_dataset,
            poison_test_dataset=poison_test_dataset,
            num_workers=num_workers,
            device=device,
            dataset_name=dataset_name,
            attack_type=attack_type,
            split_time=split_time,
            logger=logger
        )
        threshold = 0
        clean_in_clean_count = 0
        clean_in_poison_count = 0
        poison_in_clean_count = 0
        poison_in_poison_count = 0
        if('semi' not in split_time):
            split_input = predict_result
            if ('no_defense' in split_time or 'PTM' in split_time) and avg_loss_artifacts is not None and avg_loss_artifacts.predict_result:
                split_input = avg_loss_artifacts.predict_result
            (split_result_dict[split_time]["clean_pool_image_names"],
             split_result_dict[split_time]["poison_pool_image_names"]) = split.split(split_time, split_input, dataset_name)
            if ('no_defense' in split_time or 'PTM' in split_time) and avg_loss_artifacts is not None:
                avg_loss_artifacts.write_split_results(
                    split_result_dict[split_time]["clean_pool_image_names"],
                    split_result_dict[split_time]["poison_pool_image_names"],
                    threshold,
                )
            clean_in_clean_count = sum(1 for path in split_result_dict[split_time]["clean_pool_image_names"] if path in clean_data_names)
            clean_in_poison_count = sum(1 for path in split_result_dict[split_time]["clean_pool_image_names"] if path in poison_data_names)
            poison_in_clean_count = sum(1 for path in split_result_dict[split_time]["poison_pool_image_names"] if path in clean_data_names)
            poison_in_poison_count = sum(1 for path in split_result_dict[split_time]["poison_pool_image_names"] if path in poison_data_names)
            logger.info(f"correct_clean: {clean_in_clean_count}")
            logger.info(f"wrong_poisoned: {clean_in_poison_count}")
            logger.info(f"wrong_clean: {poison_in_clean_count}")
            logger.info(f"correct_poisoned: {poison_in_poison_count}")
        with open(csv_file_name, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([dataset_name, attack_type, acc, asr, split_time, clean_in_clean_count,
                                 clean_in_poison_count, poison_in_clean_count, poison_in_poison_count,
                                 f'{random_seed}'])