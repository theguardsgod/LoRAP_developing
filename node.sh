

rank_list=(1 2 4 8)
num_prompts_list=(1 5 10 20 40)
for num_prompts in "${num_prompts_list[@]}"; do
    for rank in "${rank_list[@]}"; do
    
               CUDA_VISIBLE_DEVICES=1  python runs/node/gat_nc_lsb_gpf_aggr.py --weight_decay 1e-3 --lr 0.01 --lr_quant_scale_fea 0.05 --lr_quant_scale_weight 0.005 --lr_quant_scale_gat_fea 0.05 --lr_quant_scale_gat 0.005 --lr_quant_bit_fea 0.1 --a_loss 0.3 --drop_out 0.6 --drop_attn 0.6 --dataset_name Cora \
                    --num_prompts ${num_prompts} \
                    --rank ${rank} \
      
    done
done
