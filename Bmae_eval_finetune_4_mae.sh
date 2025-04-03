#!/bin/bash

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=2

#TODO:每次训练，调整参数，记得修改输出目录k_mae/后的部分(根据train的输出目录修改)

# 创建输出目录
mkdir -p output_dir/bootstrapped_mae_finetune/4_mae/adamw/lr1e-3_wd0.05_mask0.75

#TODO: 注意此处finetune的模型是mae_1, 也就是2次级联的模型，更多级联的模型需要相应修改

# 微调评估Bootstrapped MAE模型
python main_finetune.py \
    --model vit_tiny_patch4 \
    --batch_size 256 \
    --epochs 100 \
    --accum_iter 1 \
    --input_size 32 \
    --data_path ./data/cifar10 \
    --nb_classes 10 \
    --output_dir ./output_dir/bootstrapped_mae_finetune/4_mae/adamw/lr1e-3_wd0.05_mask0.75 \
    --log_dir ./output_dir/bootstrapped_mae_finetune/4_mae/adamw/lr1e-3_wd0.05_mask0.75 \
    --finetune ./output_dir/bootstrapped_mae/4_mae/adamw/lr1e-3_wd0.05_mask0.75/mae_3/checkpoint-49.pth \
    --blr 5e-4 \
    --weight_decay 0.05 \
    --drop_path 0.1 \
    --warmup_epochs 10 \
    --num_workers 4 