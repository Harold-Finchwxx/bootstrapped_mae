#!/bin/bash --login

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 定义参数范围
ema_decays=(0.99)
learning_rates=(5e-3 1e-3 5e-4 1e-4)
weight_decays=(0.05 0.1)
mask_ratios=(0.75)

# 遍历所有参数组合
for ema_decay in "${ema_decays[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for wd in "${weight_decays[@]}"; do
            for mask_ratio in "${mask_ratios[@]}"; do
                # 创建参数目录名
                param_dir="ed${ema_decay}_lr${lr}_wd${wd}_mask${mask_ratio}"
                
                echo "============================================="
                echo "开始训练参数组合: ${param_dir}"
                echo "============================================="
                
                # 1. 训练阶段
                echo "开始训练..."
                mkdir -p output_dir/ema_mae/adamw/${param_dir}
                python main_ema_pretrain.py \
                    --batch_size 256 \
                    --model ema_mae_vit_tiny_patch4 \
                    --epochs 200 \
                    --accum_iter 1 \
                    --blr ${lr} \
                    --weight_decay ${wd} \
                    --output_dir ./output_dir/ema_mae/adamw/${param_dir} \
                    --log_dir ./output_dir/ema_mae/adamw/${param_dir} \
                    --data_path ./data/cifar10 \
                    --input_size 32 \
                    --mask_ratio ${mask_ratio} \
                    --ema_decay ${ema_decay} \
                    --norm_pix_loss \
                    --num_workers 4
                
                # 检查训练是否成功
                if [ $? -ne 0 ]; then
                    echo "训练失败，跳过当前参数组合"
                    continue
                fi
                
                # 2. 线性评估阶段
                echo "开始线性评估..."
                mkdir -p output_dir/ema_mae_linear/adamw/${param_dir}
                python main_linprobe.py \
                    --batch_size 256 \
                    --model ema_mae_vit_tiny_patch4 \
                    --epochs 100 \
                    --accum_iter 1 \
                    --blr 0.01 \
                    --weight_decay 0.05 \
                    --output_dir ./output_dir/ema_mae_linear/adamw/${param_dir} \
                    --log_dir ./output_dir/ema_mae_linear/adamw/${param_dir} \
                    --resume ./output_dir/ema_mae/adamw/${param_dir}/checkpoint-199.pth \
                    --data_path ./data/cifar10 \
                    --nb_classes 10 \
                    --global_pool \
                    --num_workers 4
                
                # 3. 微调评估阶段
                echo "开始微调评估..."
                mkdir -p output_dir/ema_mae_finetune/adamw/${param_dir}
                python main_finetune.py \
                    --batch_size 256 \
                    --model ema_mae_vit_tiny_patch4 \
                    --epochs 100 \
                    --accum_iter 1 \
                    --blr 1e-3 \
                    --weight_decay 0.05 \
                    --output_dir ./output_dir/ema_mae_finetune/adamw/${param_dir} \
                    --log_dir ./output_dir/ema_mae_finetune/adamw/${param_dir} \
                    --resume ./output_dir/ema_mae/adamw/${param_dir}/checkpoint-199.pth \
                    --data_path ./data/cifar10 \
                    --nb_classes 10 \
                    --global_pool \
                    --input_size 32 \
                    --num_workers 4 \
                    --dist_eval
                
                echo "完成参数组合: ${param_dir}"
                echo "============================================="
            done
        done
    done
done

echo "所有参数组合训练和评估完成！" 