#!/bin/bash

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 创建输出目录
mkdir -p output_dir/bootstrapped_mae

# 训练Bootstrapped MAE模型
python main_bootstrapped_pretrain.py \
    --model bootstrapped_mae_tiny_patch4_dec96d4b \
    --batch_size 256 \
    --epochs 200 \
    --accum_iter 1 \
    --input_size 32 \
    --mask_ratio 0.75 \
    --data_path ./data/cifar10 \
    --output_dir ./output_dir/bootstrapped_mae \
    --log_dir ./output_dir/bootstrapped_mae \
    --blr 1e-3 \
    --weight_decay 0.05 \
    --warmup_epochs 40 \
    --num_workers 4 \
    --target_model ./output_dir/mae/checkpoint-199.pth \
    --use_ema \
    --ema_decay 0.999 