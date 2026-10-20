#!/bin/bash
# 超参数搜索脚本
# 调整的参数：lr_quant_bit_fea, lr_quant_scale_fea, a_loss, num_prompts

# 定义各参数的候选取值
prob_mask_low_values=(0.1 0.2 0.3 0.4 0.5)
prob_mask_change_values=(0.1 0.2 0.3 0.4 0.5)


# 遍历所有组合
for low_value in "${prob_mask_low_values[@]}"; do
    for change_value in "${prob_mask_change_values[@]}"; do
                python lr_gin_mnist_bit_gpf_indegree.py \
                    --lr_quant_scale_fea 5e-3 \
                    --lr_quant_scale_xw 5e-3 \
                    --lr_quant_scale_weight 5e-4 \
                    --lr_quant_bit_fea 1e-4 \
                    --init uniform \
                    --a_loss 5e-5 \
                    --dataset_name MNIST \
                    --prob_mask_low ${low_value} \
                    --prob_mask_change ${change_value} \
      
    done
done
