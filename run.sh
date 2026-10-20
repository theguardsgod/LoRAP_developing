# Run gpf_plus on mnist gin a2q 
#python lr_gin_mnist_bit_gpf.py --lr_quant_scale_fea 5e-3 --lr_quant_scale_xw 5e-3 --lr_quant_scale_weight 5e-4 --lr_quant_bit_fea 1e-4 --init uniform --a_loss 5e-5 --dataset_name MNIST

# Test the number of prompts 
# num_prompts_list=(2 5 10 40 80 100)
# for num_prompts in "${num_prompts_list[@]}"; do
    
#                 python runs/mnist/lr_gin_mnist_bit_gpf_aggr.py \
#                     --lr_quant_scale_fea 5e-3 \
#                     --lr_quant_scale_xw 5e-3 \
#                     --lr_quant_scale_weight 5e-4 \
#                     --lr_quant_bit_fea 1e-4 \
#                     --init uniform \
#                     --a_loss 5e-5 \
#                     --dataset_name MNIST \
#                     --num_prompts ${num_prompts} \
      
    
# done

# rank_list=(1 2 4 8)
# num_prompts_list=(5 10 20 40 60)
# for num_prompts in "${num_prompts_list[@]}"; do
#     for rank in "${rank_list[@]}"; do
    
#                CUDA_VISIBLE_DEVICES=1  python runs/mnist/lr_gin_mnist_bit_gpf_aggr_LoRA_node.py \
#                     --lr_quant_scale_fea 5e-3 \
#                     --lr_quant_scale_xw 5e-3 \
#                     --lr_quant_scale_weight 5e-4 \
#                     --lr_quant_bit_fea 1e-4 \
#                     --init uniform \
#                     --a_loss 5e-5 \
#                     --dataset_name MNIST \
#                     --num_prompts ${num_prompts} \
#                     --rank ${rank} \
      
#     done
# done

# rank_list=(2 4 8)
# num_prompts_list=(5 10 20 40)
# for num_prompts in "${num_prompts_list[@]}"; do
#     for rank in "${rank_list[@]}"; do
    
#                CUDA_VISIBLE_DEVICES=1  python ./runs/zinc/lr_gin_zinc_bit_gpf_aggr_LoRA_node.py --a_loss 1e-2 --lr_quant_scale_fea 1e-2 --lr_quant_scale_xw 5e-2 --lr_quant_scale_weight 1e-3 --lr_quant_bit_fea 1e-3 \
#                     --num_prompts ${num_prompts} \
#                     --rank ${rank} \
      
#     done
# done

rank_list=(1 2 4 8)
num_prompts_list=(1 5 10 20)
for num_prompts in "${num_prompts_list[@]}"; do
    for rank in "${rank_list[@]}"; do
    
               CUDA_VISIBLE_DEVICES=3  python runs/arxiv/gcn_ogb_arxiv_LoRAP.py --lr_quant_scale_fea 1e-2 --lr_quant_scale_weight 1e-3 --lr_quant_bit_fea 1e-2 --a_loss 1e-4 --dataset_name ogbn-arxiv \
                    --num_prompts ${num_prompts} \
                    --rank ${rank} \
      
    done
done

# python runs/node/node_level_1_node_LoRA_aggr.py --lr_quant_bit_fea 0.1 --lr_quant_scale_fea 0.04 --a_loss 2.5 --lr_quant_scale_weight 0.02 --lr_quant_scale_xw 0.008 --drop_out 0.35 --weight_decay 0.02 --dataset_name Cora --model GIN
# 
# Test froze the pretrained model and only adjust prompts and alpha gamma
# python runs/mnist/lr_gin_mnist_bit_gpf_frozen.py --lr_quant_scale_fea 5e-3 --lr_quant_scale_xw 5e-3 --lr_quant_scale_weight 5e-4 --lr_quant_bit_fea 1e-4 --init uniform --a_loss 5e-5 --dataset_name MNIST

# Test soft mask using gpf plus
#python runs/mnist/lr_gin_mnist_bit_gpf_mask.py --lr_quant_scale_fea 5e-3 --lr_quant_scale_xw 5e-3 --lr_quant_scale_weight 5e-4 --lr_quant_bit_fea 1e-4 --init uniform --a_loss 5e-5 --dataset_name MNIST

# lr_prompt_list=(1e-5 5e-5 1e-4 5e-4)
# for lr_prompt in "${lr_prompt_list[@]}"; do
#     python runs/mnist/lr_gin_mnist_bit_gpf_frozen.py --lr_quant_scale_fea 5e-3 --lr_quant_scale_xw 5e-3 --lr_quant_scale_weight 5e-4 --lr_quant_bit_fea 1e-4 --init uniform --a_loss 5e-5 --dataset_name MNIST \
#         --lr_prompt ${lr_prompt} \

# done

# CUDA_VISIBLE_DEVICES=1 python runs/mnist/lr_gin_mnist_bit_gpf_laplacian.py --lr_quant_scale_fea 5e-3 --lr_quant_scale_xw 5e-3 --lr_quant_scale_weight 5e-4 --lr_quant_bit_fea 1e-4 --init uniform --a_loss 5e-5 --dataset_name MNIST
# python runs/mnist/lr_gin_mnist_bit_gpf_IDPG.py --lr_quant_scale_fea 5e-3 --lr_quant_scale_xw 5e-3 --lr_quant_scale_weight 5e-4 --lr_quant_bit_fea 1e-4 --init uniform --a_loss 5e-5 --dataset_name MNIST
# python runs/mnist/lr_gin_mnist_bit_gpf_quan_loss_emb.py --lr_quant_scale_fea 5e-3 --lr_quant_scale_xw 5e-3 --lr_quant_scale_weight 5e-4 --lr_quant_bit_fea 1e-4 --init uniform --a_loss 5e-5 --dataset_name MNIST
# prompt_hidden_dim_list=(1 8 16 32 64)
# for prompt_hidden_dim in "${prompt_hidden_dim_list[@]}"; do
#     python runs/mnist/lr_gin_mnist_bit_gpf_IDPG.py \
#         --lr_quant_scale_fea 5e-3 \
#         --lr_quant_scale_xw 5e-3 \
#         --lr_quant_scale_weight 5e-4 \
#         --lr_quant_bit_fea 1e-4 \
#         --init uniform \
#         --a_loss 5e-5 \
#         --dataset_name MNIST \
#         --prompt_hidden_dim ${prompt_hidden_dim}
        
# done

# python runs/cifar10/lr_gin_cifar10_bit_gpf_aggr.py --lr_quant_scale_fea 5e-3 --lr_quant_scale_xw 5e-3 --lr_quant_scale_weight 5e-4 --lr_quant_bit_fea 1e-4 --init uniform --a_loss 5e-5 --dataset_name CIFAR10 
# CUDA_VISIBLE_DEVICES=2 python runs/mnist/lr_gin_mnist_normal_gpf_aggr_LoRA_node.py --lr_quant_scale_fea 5e-3 --lr_quant_scale_xw 5e-3 --lr_quant_scale_weight 5e-4 --lr_quant_bit_fea 1e-4 --init uniform --a_loss 5e-5 --dataset_name MNIST --rank 4