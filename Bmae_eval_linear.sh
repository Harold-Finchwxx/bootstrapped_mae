#!/bin/bash

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 创建输出目录
mkdir -p output_dir/bootstrapped_mae_linear

# 线性评估Bootstrapped MAE模型
python main_linprobe.py \
    --model bootstrapped_mae_tiny_patch4_dec96d4b \
    --batch_size 256 \
    --epochs 100 \
    --accum_iter 1 \
    --input_size 32 \
    --data_path ./data/cifar10 \
    --output_dir ./output_dir/bootstrapped_mae_linear \
    --log_dir ./output_dir/bootstrapped_mae_linear \
    --resume ./output_dir/bootstrapped_mae/checkpoint-199.pth \
    --blr 1e-3 \
    --weight_decay 0.05 \
    --warmup_epochs 40 \
    --num_workers 4 