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

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae
import models_bootstrapped_mae

from engine_bootstrapped_pretrain import train_one_epoch

def get_args_parser():
    parser = argparse.ArgumentParser('Bootstrapped MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='bootstrapped_mae_tiny_patch4_dec96d4b', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--num_mae', default=2, type=int,
                        help='Number of MAE models in bootstrapping')
    parser.add_argument('--current_mae_idx', default=0, type=int,
                        help='Current MAE model index to train')
    parser.add_argument('--epochs_per_mae', default=100, type=int,
                        help='Number of epochs to train each MAE model')
    parser.add_argument('--prev_mae_path', default='', type=str,
                        help='Path to previous MAE checkpoint')

    parser.add_argument('--input_size', default=32, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/cifar10', type=str,
                        help='dataset path')

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
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    # 使用CIFAR10数据集
    dataset_train = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
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
    
    # 加载前一个MAE模型（如果不是第一个MAE）
    prev_model = None
    # 在加载前一个MAE模型的部分（大约在第150行左右）
    if args.current_mae_idx > 0 and args.prev_mae_path:
        if args.current_mae_idx == 1:
            # 如果是第二个MAE（idx=1），加载原始MAE模型
            prev_model = models_mae.__dict__['mae_vit_tiny_patch4']()
            checkpoint = torch.load(args.prev_mae_path, map_location='cpu')
            # 处理状态字典键名
            state_dict = checkpoint['model']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v  # 移除'model.'前缀
                else:
                    new_state_dict[k] = v
            prev_model.load_state_dict(new_state_dict)
        else:
            # 如果是第三个或更高级的MAE（idx>1），加载bootstrapped MAE模型
            prev_model = models_bootstrapped_mae.__dict__['bootstrapped_mae_tiny_patch4_dec96d4b'](
                num_mae=args.num_mae,
                current_mae_idx=args.current_mae_idx - 1
            )
            checkpoint = torch.load(args.prev_mae_path, map_location='cpu')
            # 加载状态字典
            state_dict = checkpoint['model']
            prev_model.load_state_dict(state_dict)
        
        prev_model.to(device)
        prev_model.eval()

        # 如果是第三个或更高级的MAE，我们需要获取内部的FeatureMAE模型
        if args.current_mae_idx > 1:
            prev_model = prev_model.model  # 获取内部的FeatureMAE模型
    
    # 定义当前模型
    model = models_bootstrapped_mae.__dict__[args.model](
    num_mae=args.num_mae,
    current_mae_idx=args.current_mae_idx
    )

    # 添加：如果不是第一个模型，加载前一个模型的权重来初始化
    if args.current_mae_idx > 0 and args.prev_mae_path:
        print(f"Initializing current model (idx={args.current_mae_idx}) with weights from previous model...")
        checkpoint = torch.load(args.prev_mae_path, map_location='cpu')
        state_dict = checkpoint['model']
        
        # 获取当前模型的 state_dict
        current_state_dict = model.state_dict()
        
        # 创建新的 state_dict，只复制匹配的层
        new_state_dict = {}
        for k, v in state_dict.items():
            # 对于第一个模型到第二个模型的转换
            if args.current_mae_idx == 1:
                # 需要处理键名，因为结构不同
                if k.startswith('model.'):
                    k = k[6:]  # 移除 'model.' 前缀
                # 只复制编码器部分的权重
                if 'encoder' in k or 'patch_embed' in k or 'blocks' in k or 'norm' in k:
                    if k in current_state_dict and v.shape == current_state_dict[k].shape:
                        new_state_dict[k] = v
            # 对于第二个及以后的模型
            else:
                # 直接复制匹配的层
                if k in current_state_dict and v.shape == current_state_dict[k].shape:
                    new_state_dict[k] = v
        
        # 加载权重
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"Initialized from previous model with message: {msg}")

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training MAE {args.current_mae_idx + 1}/{args.num_mae} for {args.epochs_per_mae} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs_per_mae):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        # 如果不是第一个MAE，获取前一个模型的特征
        target_features = None
        if args.current_mae_idx > 0 and prev_model is not None:
            print(f"Extracting features from previous MAE model...")
            all_features = []
            with torch.no_grad():
                for batch in data_loader_train:
                    batch = batch[0].to(device)
                    features, _, _ = prev_model.forward_encoder(batch, mask_ratio=0)
                    features = features[:, 1:, :]  # 移除CLS token
                    all_features.append(features)
            target_features = torch.cat(all_features, dim=0)
            print(f"Extracted features shape: {target_features.shape}")
        
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args,
            target_features=target_features
        )
        
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs_per_mae):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                        'mae_idx': args.current_mae_idx}

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
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args) 