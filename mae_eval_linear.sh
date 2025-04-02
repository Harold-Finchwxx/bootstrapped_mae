#!/bin/bash --login
#SBATCH --job-name=mae_linear
#SBATCH --output=./output_dir/mae_linear/%x-%j.out
#SBATCH --error=./output_dir/mae_linear/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --mail-user=your.email@example.com
#SBATCH --mail-type=ALL
#SBATCH --partition=learnfair
#SBATCH --account=your_account

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 创建输出目录
mkdir -p output_dir/mae_linear

# 线性评估MAE模型
python main_linprobe.py \
    --batch_size 256 \
    --model vit_tiny_patch4 \
    --epochs 100 \
    --accum_iter 1 \
    --blr 0.01 \
    --weight_decay 0.05 \
    --warmup_epochs 20 \
    --data_path ./data/cifar10 \
    --output_dir ./output_dir/mae_linear \
    --log_dir ./output_dir/mae_linear \
    --resume ./output_dir/mae/checkpoint-199.pth \
    --global_pool \
    --nb_classes 10 