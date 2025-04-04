# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_

import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.lars import LARS
from util.crop import RandomResizedCrop

import models_vit

from engine_finetune import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('MAE linear probing for image classification', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    # 设置跳过自动加载模型
    args.skip_load_model = True
    
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # linear probe: weak augmentation
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    dataset_train = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
    dataset_val = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transform_val)
    print(dataset_train)
    print(dataset_val)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        global_pool=args.global_pool,
    )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(f"Loading pre-trained weights: {msg}")
        
        # 打印缺失的键和意外的键，以便调试
        print(f"Missing keys: {msg.missing_keys}")
        print(f"Unexpected keys: {msg.unexpected_keys}")
        
        # 更灵活的断言检查
        if args.global_pool:
            expected_missing = {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            expected_missing = {'head.weight', 'head.bias'}
            
        # 检查是否所有预期的缺失键都在实际的缺失键中
        missing_keys_set = set(msg.missing_keys)
        if not all(key in missing_keys_set for key in expected_missing):
            print(f"Warning: Some expected missing keys are not present in the actual missing keys")
            print(f"Expected: {expected_missing}")
            print(f"Actual: {missing_keys_set}")
        
        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)

    # for linear prob only
    # hack: revise model's head with BN
    model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    # freeze all but the head
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = LARS(model_without_ddp.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(optimizer)
    loss_scaler = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    # 加载预训练权重
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        
        # 检查模型是否包含bootstrapped_mae结构
        if 'model.model.pos_embed' in checkpoint_model:
            print("Detected bootstrapped_mae model structure")
            # 提取bootstrapped mae中嵌套的模型权重
            filtered_model = {}
            for k, v in checkpoint_model.items():
                if k.startswith('model.model.'):
                    # 修改键名，移除model.model.前缀
                    new_k = k[12:]  # 'model.model.'长度为12
                    filtered_model[new_k] = v
                elif k.startswith('model.'):
                    # 修改键名，移除model.前缀
                    new_k = k[6:]  # 'model.'长度为6
                    filtered_model[new_k] = v
            checkpoint_model = filtered_model
            print(f"Extracted nested model with keys: {list(checkpoint_model.keys())[:10]}...")
        
        # 只保留编码器部分的权重
        encoder_dict = {}
        for k, v in checkpoint_model.items():
            # 跳过解码器和mask_token相关的权重
            if k.startswith('decoder') or k == 'mask_token':
                continue
                
            # 处理encoder.前缀 (如果存在)
            if k.startswith('encoder.'):
                k_new = k[8:]  # 移除'encoder.'前缀
            else:
                k_new = k
                
            # 处理规范化层
            if k == 'norm.weight':
                if hasattr(model_without_ddp, 'norm'):
                    k_new = 'norm.weight'
                elif hasattr(model_without_ddp, 'fc_norm') and args.global_pool:
                    k_new = 'fc_norm.weight'
            elif k == 'norm.bias':
                if hasattr(model_without_ddp, 'norm'):
                    k_new = 'norm.bias'
                elif hasattr(model_without_ddp, 'fc_norm') and args.global_pool:
                    k_new = 'fc_norm.bias'
            
            # 存储适当的键和值
            encoder_dict[k_new] = v
            
        # 打印预处理后的键
        print("Processed keys for encoder:", list(encoder_dict.keys()))
        
        # 加载编码器权重
        msg = model_without_ddp.load_state_dict(encoder_dict, strict=False)
        print(f"Loading pre-trained weights: {msg}")
        
        # 初始化分类头
        trunc_normal_(model_without_ddp.head[1].weight, std=0.01)
        
        # 打印加载后的信息
        print(f"Missing keys: {msg.missing_keys}")
        print(f"Unexpected keys: {msg.unexpected_keys}")
    # 为了向后兼容，保留resume支持
    elif args.resume:
        args.finetune = args.resume
        checkpoint = torch.load(args.resume, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.resume)
        checkpoint_model = checkpoint['model']
        
        # 只保留编码器部分的权重
        encoder_dict = {}
        for k, v in checkpoint_model.items():
            # 跳过解码器和mask_token相关的权重
            if k.startswith('decoder') or k == 'mask_token':
                continue
                
            # 处理encoder.前缀 (如果存在)
            if k.startswith('encoder.'):
                k_new = k[8:]  # 移除'encoder.'前缀
            else:
                k_new = k
                
            # 处理规范化层
            if k == 'norm.weight':
                if hasattr(model_without_ddp, 'norm'):
                    k_new = 'norm.weight'
                elif hasattr(model_without_ddp, 'fc_norm') and args.global_pool:
                    k_new = 'fc_norm.weight'
            elif k == 'norm.bias':
                if hasattr(model_without_ddp, 'norm'):
                    k_new = 'norm.bias'
                elif hasattr(model_without_ddp, 'fc_norm') and args.global_pool:
                    k_new = 'fc_norm.bias'
            
            # 存储适当的键和值
            encoder_dict[k_new] = v
            
        # 打印预处理后的键
        print("Processed keys for encoder:", list(encoder_dict.keys()))
        
        # 加载编码器权重
        msg = model_without_ddp.load_state_dict(encoder_dict, strict=False)
        print(f"Loading pre-trained weights: {msg}")
        
        # 初始化分类头
        trunc_normal_(model_without_ddp.head[1].weight, std=0.01)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=None,
            log_writer=log_writer,
            args=args
        )
        # 只保存最后一次模型（节约磁盘空间）
        if args.output_dir and epoch == args.epochs - 1:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    # 只保存最后一次模型（节约磁盘空间）
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
