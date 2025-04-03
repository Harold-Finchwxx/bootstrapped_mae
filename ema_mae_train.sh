#!/bin/bash --login

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0


#TODO:每次训练，调整参数，记得修改输出目录ema_mae/后的部分

# 创建输出目录
mkdir -p output_dir/ema_mae/adamw/ed0.999_lr1e-3_wd0.05_mask0.75

python main_ema_pretrain.py \
    --batch_size 256 \
    --model ema_mae_vit_tiny_patch4 \
    --epochs 100 \
    --accum_iter 1 \
    --blr 1e-3 \
    --weight_decay 0.05 \
    --output_dir ./output_dir/ema_mae/adamw/ed0.999_lr1e-3_wd0.05_mask0.75 \
    --log_dir ./output_dir/ema_mae/adamw/ed0.999_lr1e-3_wd0.05_mask0.75 \
    --data_path ./data/cifar10 \
    --input_size 32 \
    --mask_ratio 0.75 \
    --ema_decay 0.999 \
    --norm_pix_loss \
    --num_workers 4 