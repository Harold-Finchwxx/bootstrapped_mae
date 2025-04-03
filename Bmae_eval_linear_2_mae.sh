#!/bin/bash

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0   

#TODO:每次训练，调整参数，记得修改输出目录k_mae/后的部分(根据train的输出目录修改)

# 创建输出目录
mkdir -p output_dir/bootstrapped_mae_linear/2_mae/adamw/lr1e-3_wd0.05_mask0.75

# 线性评估Bootstrapped MAE模型
python main_linprobe.py \
    --model vit_tiny_patch4 \
    --batch_size 256 \
    --epochs 100 \
    --accum_iter 1 \
    --data_path ./data/cifar10 \
    --nb_classes 10 \
    --global_pool \
    --output_dir ./output_dir/bootstrapped_mae_linear/2_mae/adamw/lr1e-3_wd0.05_mask0.75 \
    --log_dir ./output_dir/bootstrapped_mae_linear/2_mae/adamw/lr1e-3_wd0.05_mask0.75 \
    --finetune ./output_dir/bootstrapped_mae/2_mae/adamw/lr1e-3_wd0.05_mask0.75/mae_1/checkpoint-99.pth \
    --blr  0.01\
    --weight_decay 0.05 \
    --warmup_epochs 10 \
    --num_workers 4 