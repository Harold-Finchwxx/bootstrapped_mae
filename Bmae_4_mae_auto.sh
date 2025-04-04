#!/bin/bash --login

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=2

# 定义参数范围
learning_rates=(1e-2 5e-3 1e-3 5e-4 1e-4)
weight_decays=(0.05 0.1)
mask_ratios=(0.75)

# 遍历所有参数组合
for lr in "${learning_rates[@]}"; do
    for wd in "${weight_decays[@]}"; do
        for mask_ratio in "${mask_ratios[@]}"; do
            # 创建参数目录名
            param_dir="lr${lr}_wd${wd}_mask${mask_ratio}"
        
            echo "============================================="
            echo "开始训练参数组合: ${param_dir}"
            echo "============================================="
            
            # 创建输出目录
            train_dir="output_dir/bootstrapped_mae/4_mae/adamw/weighted_epoch/${param_dir}"
            linear_dir="output_dir/bootstrapped_mae_linear/4_mae/adamw/weighted_epoch/${param_dir}"
            finetune_dir="output_dir/bootstrapped_mae_finetune/4_mae/adamw/weighted_epoch/${param_dir}"
            
            # 1. 训练第一个MAE
            echo "开始训练第一个MAE..."
            mkdir -p ${train_dir}/mae_0
            python main_bootstrapped_pretrain.py \
                --batch_size 256 \
                --model bootstrapped_mae_tiny_patch4_dec96d4b \
                --num_mae 4 \
                --current_mae_idx 0 \
                --epochs_per_mae 80 \
                --mask_ratio ${mask_ratio} \
                --accum_iter 1 \
                --warmup_epochs 8 \
                --blr ${lr} \
                --weight_decay ${wd} \
                --output_dir ${train_dir}/mae_0 \
                --log_dir ${train_dir}/mae_0 \
                --data_path ./data/cifar10
            
            if [ $? -ne 0 ]; then
                echo "第一个MAE训练失败,跳过当前参数组合"
                continue
            fi
            
            # 2. 训练第二个MAE
            echo "开始训练第二个MAE..."
            mkdir -p ${train_dir}/mae_1
            python main_bootstrapped_pretrain.py \
                --batch_size 256 \
                --model bootstrapped_mae_tiny_patch4_dec96d4b \
                --num_mae 4 \
                --current_mae_idx 1 \
                --epochs_per_mae 50 \
                --mask_ratio ${mask_ratio} \
                --accum_iter 1 \
                --warmup_epochs 5 \
                --blr ${lr} \
                --weight_decay ${wd} \
                --output_dir ${train_dir}/mae_1 \
                --log_dir ${train_dir}/mae_1 \
                --data_path ./data/cifar10 \
                --prev_mae_path ${train_dir}/mae_0/checkpoint-79.pth
            
            if [ $? -ne 0 ]; then
                echo "第二个MAE训练失败,跳过当前参数组合"
                continue
            fi
            
            # 3. 训练第三个MAE
            echo "开始训练第三个MAE..."
            mkdir -p ${train_dir}/mae_2
            python main_bootstrapped_pretrain.py \
                --batch_size 256 \
                --model bootstrapped_mae_tiny_patch4_dec96d4b \
                --num_mae 4 \
                --current_mae_idx 2 \
                --epochs_per_mae 35 \
                --mask_ratio ${mask_ratio} \
                --accum_iter 1 \
                --warmup_epochs 4 \
                --blr ${lr} \
                --weight_decay ${wd} \
                --output_dir ${train_dir}/mae_2 \
                --log_dir ${train_dir}/mae_2 \
                --data_path ./data/cifar10 \
                --prev_mae_path ${train_dir}/mae_1/checkpoint-49.pth
            
            if [ $? -ne 0 ]; then
                echo "第三个MAE训练失败,跳过当前参数组合"
                continue
            fi
            
            # 4. 训练第四个MAE
            echo "开始训练第四个MAE..."
            mkdir -p ${train_dir}/mae_3
            python main_bootstrapped_pretrain.py \
                --batch_size 256 \
                --model bootstrapped_mae_tiny_patch4_dec96d4b \
                --num_mae 4 \
                --current_mae_idx 3 \
                --epochs_per_mae 35 \
                --mask_ratio ${mask_ratio} \
                --accum_iter 1 \
                --warmup_epochs 4 \
                --blr ${lr} \
                --weight_decay ${wd} \
                --output_dir ${train_dir}/mae_3 \
                --log_dir ${train_dir}/mae_3 \
                --data_path ./data/cifar10 \
                --prev_mae_path ${train_dir}/mae_2/checkpoint-34.pth
            
            if [ $? -ne 0 ]; then
                echo "第四个MAE训练失败,跳过当前参数组合"
                continue
            fi
            
            # 5. 线性评估
            echo "开始线性评估..."
            mkdir -p ${linear_dir}
            python main_linprobe.py \
                --model vit_tiny_patch4 \
                --batch_size 512 \
                --epochs 100 \
                --accum_iter 1 \
                --data_path ./data/cifar10 \
                --nb_classes 10 \
                --global_pool \
                --output_dir ${linear_dir} \
                --log_dir ${linear_dir} \
                --finetune ${train_dir}/mae_3/checkpoint-34.pth \
                --blr 0.01 \
                --weight_decay 0.05 \
                --warmup_epochs 10 \
                --num_workers 4
            
            # 6. 微调评估
            echo "开始微调评估..."
            mkdir -p ${finetune_dir}
            python main_finetune.py \
                --model vit_tiny_patch4 \
                --batch_size 512 \
                --epochs 100 \
                --accum_iter 1 \
                --input_size 32 \
                --data_path ./data/cifar10 \
                --nb_classes 10 \
                --output_dir ${finetune_dir} \
                --log_dir ${finetune_dir} \
                --finetune ${train_dir}/mae_3/checkpoint-34.pth \
                --blr 5e-4 \
                --weight_decay 0.05 \
                --drop_path 0.1 \
                --warmup_epochs 10 \
                --num_workers 4
            
            echo "完成参数组合: ${param_dir}"
            echo "============================================="
            
        done
    done
done

echo "所有参数组合训练和评估完成！"