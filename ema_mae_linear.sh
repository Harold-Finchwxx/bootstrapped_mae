#!/bin/bash --login

export CUDA_VISIBLE_DEVICES=0

#TODO:每次训练，调整参数，记得修改输出目录ema_mae_linear/后的部分(根据train的输出目录修改)

mkdir -p output_dir/ema_mae_linear/adamw/ed0.999_lr1e-3_wd0.05_mask0.75

python main_linprobe.py \
    --batch_size 256 \
    --model ema_mae_vit_tiny_patch4 \
    --epochs 100 \
    --accum_iter 1 \
    --blr 0.01 \
    --weight_decay 0.05 \
    --output_dir ./output_dir/ema_mae_linear/adamw/ed0.999_lr1e-3_wd0.05_mask0.75 \
    --log_dir ./output_dir/ema_mae_linear/adamw/ed0.999_lr1e-3_wd0.05_mask0.75 \
    --resume ./output_dir/ema_mae/adamw/ed0.999_lr1e-3_wd0.05_mask0.75/checkpoint-199.pth \
    --data_path ./data/cifar10 \
    --nb_classes 10 \
    --global_pool \
    --num_workers 4 