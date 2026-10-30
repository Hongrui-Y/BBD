import copy
import os
import torch
from PIL import Image
from torch.utils.data import ConcatDataset
from torchvision import transforms
from io import BytesIO
import utils
from torchvision.transforms import InterpolationMode



class MixMatchLabeledDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super(MixMatchLabeledDataset, self).__init__()
        self.dataset = copy.deepcopy(dataset)
        self.labeled = True

    def __getitem__(self, index):
        sample = self.dataset[index]
        item =  {"img": sample[1], "target": sample[2]}
        item["labeled"] = True
        return item

    def __len__(self):
        return len(self.dataset)

class MixMatchUnLabeledDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        super(MixMatchUnLabeledDataset, self).__init__()
        self.dataset = copy.deepcopy(dataset)
        self.labeled = False

    def __getitem__(self, index):
        name1, img1, label = self.dataset[index]
        name2, img2, label2 = self.dataset[index]
        assert name1 == name2 and label == label2

        item = {
            "img1": img1,
            "img2": img2,
            "target": label,
            "name": name1,
            "labeled": False,
        }
        return item

    def __len__(self):
        return len(self.dataset)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, file_name, dataset_name, split_time=None, label_flipping=None, target=None):
        bin_data_path = data_folder
        index_file = file_name
        self.bin_data_path = bin_data_path
        self.images = []
        self.labels = []
        self.num_classes = get_num_classes(dataset_name, split_time)
        self.image_names = []
        index_file_path = os.path.join(bin_data_path, f'{index_file}_index.txt')
        with open(index_file_path, 'r') as f:
            lines = f.readlines()
        bin_file_path = os.path.join(bin_data_path, f'{index_file}.bin')
        with open(bin_file_path, 'rb') as bin_file:
            bin_data = bin_file.read()
        for line in lines:
            image_name, label, offset, size = line.strip().split(',')
            offset, size = int(offset), int(size)
            label = int(label)
            if label_flipping == 'poison':
                label = int(target)
            else:
                label = label
            image_data = bin_data[offset:offset + size]
            image = Image.open(BytesIO(image_data)).convert('RGB')
            self.image_names.append(image_name)
            self.images.append(image)
            self.labels.append(label)

    def __getitem__(self, index):
        return self.image_names[index], self.labels[index], self.images[index],

    def __len__(self):
        return len(self.images)

    def get_names(self):
        data_names = set()
        for idx in range(self.__len__()):
            image_name, _, _ = self.__getitem__(idx)
            data_names.add(image_name)
        return data_names


class SubsetFromIdx(torch.utils.data.Dataset):
    def __init__(self, image_names, dataset, transform, num_classes, label_flipping=None, target=None):
        if (label_flipping == None):
            label_flipping == 'normal'
        if (image_names == None):
            self.data = copy.deepcopy(dataset)
        else:
            self.data = [copy.deepcopy(dataset[i]) for i in range(len(dataset)) if dataset[i][0] in image_names]
        self.transform = transform
        self.num_classes = num_classes
        self.image_names = []
        self.labels = []
        self.images = []
        self.label_flipping = label_flipping

        for image_name, label, image in self.data:
            if (label_flipping == 'poison'):
                flipped_label = int(target)
            elif (label_flipping == 'random'):
                flipped_label = utils.generate_random_except(0, self.num_classes - 1, int(label))
            elif (label_flipping == 'trap'):
                flipped_label = self.num_classes - 1
            else:
                flipped_label = label
            self.image_names.append(image_name)
            self.labels.append(int(flipped_label))
            self.images.append(image)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        label = self.labels[index]
        image = self.images[index]
        if self.transform:
            image = self.transform(image)

        return image_name, image, label

    def __len__(self):
        return len(self.image_names)


def get_train_dataset(clean_pool_image_paths, poison_pool_image_paths, train_data, split_time, dataset_name, ):
    if 'CTM' in split_time or 'all_clean' in split_time:
        train_dataset = SubsetFromIdx(
            clean_pool_image_paths,
            train_data,
            get_train_transforms(
                dataset_name=dataset_name,
                is_enhance=True
            ),
            num_classes=get_num_classes(dataset_name, split_time),
        )
    elif ('PTM' in split_time):
        adding_dataset = SubsetFromIdx(
            clean_pool_image_paths,
            train_data,
            get_random_crop_transform(
                dataset_name=dataset_name,
            ),
            num_classes=get_num_classes(dataset_name, split_time),
            label_flipping='random'
        )
        poison_train_dataset = adding_dataset
        train_dataset = ConcatDataset(
            [
                SubsetFromIdx(
                    poison_pool_image_paths,
                    train_data,
                    get_train_transforms(
                        dataset_name=dataset_name,
                        is_enhance=False
                    ),
                    num_classes=get_num_classes(dataset_name, split_time),
                ),
                poison_train_dataset
            ]
        )
    elif ('no_defense' == split_time):
        clean_train_data = SubsetFromIdx(
            clean_pool_image_paths,
            train_data,
            get_train_transforms(
                dataset_name=dataset_name,
                is_enhance=False
            ),
            num_classes=get_num_classes(dataset_name, split_time),
        )
        if poison_pool_image_paths is not None:
            poison_train_data = SubsetFromIdx(
                poison_pool_image_paths,
                train_data,
                get_train_transforms(
                    dataset_name=dataset_name,
                    is_enhance=False
                ),
                num_classes=get_num_classes(dataset_name, split_time),
                label_flipping='poison',
                target=0
            )
            train_dataset = ConcatDataset([clean_train_data, poison_train_data])
        else:
            train_dataset = clean_train_data
    return train_dataset


def get_test_dataset(clean_test_data, poison_test_data, dataset_name):
    size = 32
    clean_test_dataset = SubsetFromIdx(
        None,
        clean_test_data,
        transform=transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            getNormal(dataset_name=dataset_name)
        ]),
        num_classes=get_num_classes(dataset_name),
    )
    poison_test_dataset = SubsetFromIdx(
        None,
        poison_test_data,
        transform=transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            getNormal(dataset_name=dataset_name)
        ]),
        num_classes=get_num_classes(dataset_name),
    )
    return clean_test_dataset, poison_test_dataset



def get_num_classes(dataset_name, split_time=None):
    if (dataset_name == 'cifar10'):
        num_classes = 10
    else:
        num_classes = -1

    if split_time != None and  'semi' in split_time:
        num_classes = num_classes + 1
    return num_classes


def get_train_transforms(dataset_name, is_enhance=False):
    size = 32
    if (not is_enhance):
        transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            getNormal(dataset_name=dataset_name)
        ])
    elif (is_enhance):
        if (dataset_name == 'cifar10'):
            transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.RandomCrop(size, padding=int(size / 8), padding_mode='reflect'),
                # transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                getNormal(dataset_name=dataset_name)

            ])
    return transform


def get_random_crop_transform(dataset_name):
    if (dataset_name == 'cifar10'):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            getNormal(dataset_name=dataset_name)
        ])
    return transform



def getNormal(dataset_name):
    if (dataset_name == 'cifar10'):
        Normal = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    else:
        Normal = None
    return Normal
