#!/bin/bash

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=2

# 创建输出目录
mkdir -p output_dir/bootstrapped_mae_linear/4_mae

# 线性评估Bootstrapped MAE模型
python main_linprobe.py \
    --model vit_tiny_patch4 \
    --batch_size 256 \
    --epochs 100 \
    --accum_iter 1 \
    --data_path ./data/cifar10 \
    --nb_classes 10 \
    --global_pool \
    --output_dir ./output_dir/bootstrapped_mae_linear/4_mae \
    --log_dir ./output_dir/bootstrapped_mae_linear/4_mae \
    --finetune ./output_dir/bootstrapped_mae/4_mae/mae_3/checkpoint-49.pth \
    --blr  0.01\
    --weight_decay 0.05 \
    --warmup_epochs 10 \
    --num_workers 4 \
    --dist_eval 