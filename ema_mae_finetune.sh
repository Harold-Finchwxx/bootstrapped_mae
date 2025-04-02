#!/bin/bash --login

export CUDA_VISIBLE_DEVICES=0

mkdir -p output_dir/ema_mae_finetune

python main_finetune.py \
    --batch_size 256 \
    --model ema_mae_vit_tiny_patch4 \
    --epochs 100 \
    --accum_iter 1 \
    --blr 5e-4 \
    --weight_decay 0.05 \
    --warmup_epochs 10 \
    --output_dir ./output_dir/ema_mae_finetune \
    --log_dir ./output_dir/ema_mae_finetune \
    --resume ./output_dir/ema_mae/checkpoint-199.pth \
    --data_path ./data/cifar10 \
    --nb_classes 10 \
    --global_pool \
    --input_size 32 \
    --num_workers 4 \
    --dist_eval 