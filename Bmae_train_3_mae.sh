#!/bin/bash --login 

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=1

# 创建输出目录
mkdir -p output_dir/bootstrapped_mae/3_mae

# 训练第一个MAE
python main_bootstrapped_pretrain.py \
    --batch_size 256 \
    --model bootstrapped_mae_tiny_patch4_dec96d4b \
    --num_mae 2 \
    --current_mae_idx 0 \
    --epochs_per_mae 68 \
    --mask_ratio 0.75 \
    --accum_iter 1 \
    --warmup_epochs 20 \
    --blr 1e-3 \
    --weight_decay 0.05 \
    --output_dir ./output_dir/bootstrapped_mae/3_mae/mae_0 \
    --log_dir ./output_dir/bootstrapped_mae/3_mae/mae_0 \
    --data_path ./data/cifar10

# 训练第二个MAE
python main_bootstrapped_pretrain.py \
    --batch_size 256 \
    --model bootstrapped_mae_tiny_patch4_dec96d4b \
    --num_mae 2 \
    --current_mae_idx 1 \
    --epochs_per_mae 66 \
    --mask_ratio 0.75 \
    --accum_iter 1 \
    --warmup_epochs 20 \
    --blr 5e-4 \
    --weight_decay 0.05 \
    --output_dir ./output_dir/bootstrapped_mae/3_mae/mae_1 \
    --log_dir ./output_dir/bootstrapped_mae/3_mae/mae_1 \
    --data_path ./data/cifar10 \
    --prev_mae_path ./output_dir/bootstrapped_mae/3_mae/mae_0/checkpoint-67.pth 

# 训练第三个MAE
python main_bootstrapped_pretrain.py \
    --batch_size 256 \
    --model bootstrapped_mae_tiny_patch4_dec96d4b \
    --num_mae 2 \
    --current_mae_idx 2 \
    --epochs_per_mae 66 \
    --mask_ratio 0.75 \
    --accum_iter 1 \
    --warmup_epochs 20 \
    --blr 5e-4 \
    --weight_decay 0.05 \
    --output_dir ./output_dir/bootstrapped_mae/3_mae/mae_2 \
    --log_dir ./output_dir/bootstrapped_mae/3_mae/mae_2 \
    --data_path ./data/cifar10 \
    --prev_mae_path ./output_dir/bootstrapped_mae/3_mae/mae_1/checkpoint-65.pth 