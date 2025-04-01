#!/bin/bash --login 

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 创建输出目录
mkdir -p output_dir/ema_mae

python main_ema_pretrain.py \
    --batch_size 256 \
    --model ema_mae_vit_tiny_patch4 \
    --mask_ratio 0.75 \
    --ema_decay 0.999 \
    --epochs 200 \
    --accum_iter 1 \
    --warmup_epochs 40 \
    --blr 1e-3 \
    --weight_decay 0.05 \
    --output_dir ./output_dir/ema_mae \
    --log_dir ./output_dir/ema_mae \
    --data_path ./data/cifar10