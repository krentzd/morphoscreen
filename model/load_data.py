from torchvision import datasets, transforms
import torchvision
import torch
import os
import math
import numpy as np

from dataloaders import TripletDataset, ClassifierDataset
from utils import OverlappingCrop, OverlappingCropMultiChannel, intersect_dicts, convert_to_list

def dataset_loader(type='triplet'):
    if type == 'class':
        return ClassifierDataset
    elif type == 'triplet':
        return TripletDataset

def load_data(root_dir,
              train_val_test_dir=[],
              dropped_classes=[],
              batch_size=32,
              val_split=0.2,
              crop_size=512,
              **kwargs):
    """
    Return train, val and test dataloaders
    """

    class_merge_dict = get_class_merge_dict()

    if kwargs.get('use_cglu_moa', False) and kwargs.get('classify_by_moa', False):
        cglu_moa_dict = get_cglu_moa_dict()

        class_merge_dict = intersect_dicts(class_merge_dict, cglu_moa_dict)

    ch_list = kwargs.get('channels', [0,1,2]):
    data_mean = 0.5
    data_std = 0.5

    train_transforms_list = [transforms.RandomVerticalFlip(),
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomRotation(90),
                             transforms.Lambda(lambda image: [transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1)(image_slice) for image_slice in convert_to_list(image)]), #Apply color jitter to channels individually, since they're uncorrelated
                             transforms.Lambda(lambda image_list: torch.stack([image_slice for image_slice in image_list]).squeeze(1)),
                             transforms.CenterCrop(math.ceil(np.sqrt(0.5) * 2160)),
                             transforms.RandomCrop(1500),
                             OverlappingCropMultiChannel(crop_size, crop_size, pad=False),
                             transforms.Lambda(lambda crops: torch.stack([crop for crop in crops])),
                             transforms.Resize((256, 256)),
                             transforms.Normalize(mean=data_mean, std=data_std)]

    train_transforms_list_png = [transforms.RandomVerticalFlip(),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(90),
                                 transforms.CenterCrop(math.ceil(np.sqrt(0.5) * 2160)),
                                 transforms.RandomCrop(1500),
                                 OverlappingCrop(crop_size, crop_size, pad=False),
                                 transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                 transforms.Resize((256, 256)),
                                 transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
                                 transforms.Lambda(lambda x: x[:,ch_list]),
                                 transforms.Normalize(mean=data_mean, std=data_std)]

    test_transforms_list = [transforms.CenterCrop(1500),
                            OverlappingCropMultiChannel(crop_size, crop_size, pad=False),
                            transforms.Lambda(lambda crops: torch.stack([crop for crop in crops])),
                            transforms.Resize((256, 256)),
                            transforms.Normalize(mean=data_mean, std=data_std)]

    test_transforms_list_png = [transforms.CenterCrop(1500),
                                OverlappingCrop(crop_size, crop_size, pad=False),
                                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                transforms.Resize((256, 256)),
                                transforms.Lambda(lambda x: x[:,ch_list]),
                                transforms.Normalize(mean=data_mean, std=data_std)]

    train_transforms =  transforms.Compose(train_transforms_list)
    test_transforms =  transforms.Compose(test_transforms_list)

    train_transforms_png =  transforms.Compose(train_transforms_list_png)
    test_transforms_png =  transforms.Compose(test_transforms_list_png)

    if len(train_val_test_dir) > 0:
        if isinstance(train_val_test_dir[0], str):
            root_train = [os.path.join(root_dir, train_val_test_dir[0])]
        else:
            print(train_val_test_dir[0])
            root_train = [os.path.join(root_dir, dir) for dir in train_val_test_dir[0]]

    train_dataset = dataset_loader(type=kwargs.get('loader_type', 'class'))(root=root_train,
                                                                        dropped_classes=dropped_classes,
                                                                        transform=train_transforms if kwargs.get('data_type', 'tiff') == 'tiff' else train_transforms_png,
                                                                        channels=kwargs.get('channels', None),
                                                                        class_merge_dict=class_merge_dict,
                                                                        subsampling_factor=kwargs.get('subsampling_factor', 1.),
                                                                        classify_by_moa=kwargs.get('classify_by_moa', False))
    if len(train_val_test_dir) > 0:
        if isinstance(train_val_test_dir[1], str):
            root_val = [os.path.join(root_dir, train_val_test_dir[1])]
        elif isinstance(train_val_test_dir[1], list):
            root_val = [os.path.join(root_dir, dir) for dir in train_val_test_dir[1]]
        elif train_val_test_dir[1] == None:
            train_size = int((1 - val_split) * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    if 'val_dataset' not in locals():
        val_dataset = dataset_loader(type=kwargs.get('loader_type', 'class'))(root=root_val,
                                                                            dropped_classes=dropped_classes,
                                                                            transform=train_transforms if kwargs.get('data_type', 'tiff') == 'tiff' else train_transforms_png,
                                                                            channels=kwargs.get('channels', None),
                                                                            class_merge_dict=class_merge_dict,
                                                                            classify_by_moa=kwargs.get('classify_by_moa', False))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=12)

    if len(train_val_test_dir) > 0:
        if isinstance(train_val_test_dir[2], str):
            root_test = [os.path.join(root_dir, train_val_test_dir[2])]
        else:
            root_test = [os.path.join(root_dir, dir) for dir in train_val_test_dir[2]]

    test_dataset = dataset_loader(type=kwargs.get('loader_type', 'class'))(root=root_test,
                                                                        is_test=True,
                                                                        dropped_classes=dropped_classes,
                                                                        transform=test_transforms if kwargs.get('data_type', 'tiff') == 'tiff' else test_transforms_png,
                                                                        channels=kwargs.get('channels', None),
                                                                        class_merge_dict=class_merge_dict,
                                                                        classify_by_moa=kwargs.get('classify_by_moa', False))

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=12)

    if kwargs.get('use_e_coli_moa', False) and kwargs.get('classify_by_moa', False):
        temp_dict = dict()
        for x in test_dataset.classes:
            temp_dict[x] = e_coli_moa_dict[x]
        class_weights = [1 - (list(temp_dict.values()).count(x) / len(temp_dict.values())) for x in test_dataset.classes_moa]
    elif kwargs.get('use_h_pylori_moa', False) and kwargs.get('classify_by_moa', False):
        temp_dict = dict()
        for x in test_dataset.classes:
            temp_dict[x] = h_pylori_moa_dict[x]
        class_weights = [1 - (list(temp_dict.values()).count(x) / len(temp_dict.values())) for x in test_dataset.classes_moa]
    elif kwargs.get('use_cglu_moa', False) and kwargs.get('classify_by_moa', False):
        temp_dict = dict()
        for x in test_dataset.classes:
            temp_dict[x] = cglu_moa_dict[x]
        class_weights = [1 - (list(temp_dict.values()).count(x) / len(temp_dict.values())) for x in test_dataset.classes_moa]
    else:
        class_weights = [1 - (6 / len(test_dataset.classes)) if c=='DMSO' or c=='Water' else 1 - (1 / len(test_dataset.classes)) for c in test_dataset.classes]

    if kwargs.get('loader_type', 'class') == 'class':
        return train_loader, val_loader, test_loader, test_dataset.classes, [], class_weights
    elif kwargs.get('loader_type', 'class') == 'triplet':
        return train_loader, val_loader, test_loader, test_dataset.classes, test_dataset.classes_moa, class_weights
