# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    
    # 检查数据路径的最后一级目录名称
    data_path_last_dir = os.path.basename(os.path.normpath(args.data_path))
    
    # 如果是cifar10数据集
    if 'cifar10' in data_path_last_dir.lower():
        dataset = datasets.CIFAR10(
            args.data_path, train=is_train, transform=transform, download=True)
        if is_train:
            print(f"Training dataset: CIFAR10, {len(dataset)} images")
        else:
            print(f"Validation dataset: CIFAR10, {len(dataset)} images")
        return dataset
    
    # 默认ImageNet格式的文件夹数据集
    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)
    print(dataset)

    return dataset


def build_transform(is_train, args):
    # 检查数据路径的最后一级目录名称
    data_path_last_dir = os.path.basename(os.path.normpath(args.data_path))
    
    # 如果是cifar10数据集，使用CIFAR10的均值和标准差
    if 'cifar10' in data_path_last_dir.lower():
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else:
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
        
    # train transform
    if is_train:
        # 检查是否是CIFAR10
        if 'cifar10' in data_path_last_dir.lower():
            transform = transforms.Compose([
                transforms.RandomResizedCrop(args.input_size, scale=(0.8, 1.0), interpolation=PIL.Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            return transform
        else:
            # 其他数据集使用原始transform
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation='bicubic',
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
                mean=mean,
                std=std,
            )
            return transform

    # eval transform
    t = []
    
    # 对于CIFAR10，使用简单的变换
    if 'cifar10' in data_path_last_dir.lower():
        t.append(transforms.Resize(args.input_size, interpolation=PIL.Image.BICUBIC))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)
        
    # 其他数据集使用原始的eval transform
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
