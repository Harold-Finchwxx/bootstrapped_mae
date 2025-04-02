#!/bin/bash

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 创建输出目录
mkdir -p output_dir/mae_finetune

# 微调评估MAE模型
python main_finetune.py \
    --model vit_tiny_patch4 \
    --batch_size 256 \
    --epochs 100 \
    --accum_iter 1 \
    --input_size 32 \
    --data_path ./data/cifar10 \
    --nb_classes 10 \
    --output_dir ./output_dir/mae_finetune \
    --log_dir ./output_dir/mae_finetune \
    --finetune ./output_dir/mae/checkpoint-199.pth \
    --blr 5e-4 \
    --weight_decay 0.05 \
    --drop_path 0.1 \
    --warmup_epochs 10 \
    --num_workers 4 \
    --dist_eval 