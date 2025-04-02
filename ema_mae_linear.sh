#!/bin/bash --login

export CUDA_VISIBLE_DEVICES=0

mkdir -p output_dir/ema_mae_linear

python main_linprobe.py \
    --batch_size 256 \
    --model ema_mae_vit_tiny_patch4 \
    --epochs 100 \
    --accum_iter 1 \
    --blr 0.01 \
    --weight_decay 0.0 \
    --output_dir ./output_dir/ema_mae_linear \
    --log_dir ./output_dir/ema_mae_linear \
    --resume ./output_dir/ema_mae/checkpoint-199.pth \
    --data_path ./data/cifar10 \
    --nb_classes 10 \
    --global_pool \
    --num_workers 4 \
    --dist_eval 