
# Test the number of prompts 
num_prompts_list=(10 20 40 60)
for num_prompts in "${num_prompts_list[@]}"; do
    
              CUDA_VISIBLE_DEVICES=1  python runs/mnist/lr_gcn_mnist_bit_gpf_aggr_LoRA_node.py \
                    --lr_quant_scale_fea 1e-3 \
                    --lr_quant_scale_xw 1e-3 \
                    --lr_quant_scale_weight 1e-3 \
                    --lr_quant_bit_fea 1e-3 \
                    --init uniform \
                    --a_loss 1e-3 \
                    --dataset_name MNIST \
                    --num_prompts ${num_prompts} \
      
    
done
