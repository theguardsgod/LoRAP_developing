#!/bin/bash
# 超参数搜索脚本
# 调整的参数：lr_quant_bit_fea, lr_quant_scale_fea, a_loss, num_prompts

# 定义各参数的候选取值
lr_quant_bit_fea_values=(0.01 0.02 0.04 0.08 0.1)
lr_quant_scale_fea_values=(0.01 0.04 0.05 0.1)
a_loss_values=(1 1.5 2 2.5 3)
num_prompts_values=(1 5 10 15 20 25 30 35 40 45 50)

# 遍历所有组合
for bit in "${lr_quant_bit_fea_values[@]}"; do
    for scale in "${lr_quant_scale_fea_values[@]}"; do
        for loss in "${a_loss_values[@]}"; do
            for prompts in "${num_prompts_values[@]}"; do
                echo "运行参数: lr_quant_bit_fea=${bit}, lr_quant_scale_fea=${scale}, a_loss=${loss}, num_prompts=${prompts}"
                python node_level_1_gpf.py \
                    --lr_quant_bit_fea ${bit} \
                    --lr_quant_scale_fea ${scale} \
                    --a_loss ${loss} \
                    --lr_quant_scale_weight 0.005 \
                    --lr_quant_scale_xw 0.005 \
                    --dataset_name CiteSeer \
                    --model GIN \
                    --path2dataset /data/cyliu/dataset/CiteSeer \
                    --num_prompts ${prompts}
            done
        done
    done
done
