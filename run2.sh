# Run gpf_plus on mnist gin a2q 
#python lr_gin_mnist_bit_gpf.py --lr_quant_scale_fea 5e-3 --lr_quant_scale_xw 5e-3 --lr_quant_scale_weight 5e-4 --lr_quant_bit_fea 1e-4 --init uniform --a_loss 5e-5 --dataset_name MNIST


# Test froze the pretrained model and only adjust prompts and alpha gamma
# python runs/mnist/lr_gin_mnist_bit_gpf_frozen.py --lr_quant_scale_fea 5e-3 --lr_quant_scale_xw 5e-3 --lr_quant_scale_weight 5e-4 --lr_quant_bit_fea 1e-4 --init uniform --a_loss 5e-5 --dataset_name MNIST

# Test soft mask using gpf plus
#python runs/mnist/lr_gin_mnist_bit_gpf_mask.py --lr_quant_scale_fea 5e-3 --lr_quant_scale_xw 5e-3 --lr_quant_scale_weight 5e-4 --lr_quant_bit_fea 1e-4 --init uniform --a_loss 5e-5 --dataset_name MNIST

lr_prompt_list=(1e-7 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4)
for lr_prompt in "${lr_prompt_list[@]}"; do
    python runs/mnist/lr_gin_mnist_bit_gpf_fewshot.py --lr_quant_scale_fea 5e-3 --lr_quant_scale_xw 5e-3 --lr_quant_scale_weight 5e-4 --lr_quant_bit_fea 1e-4 --init uniform --a_loss 5e-5 --dataset_name MNIST \
        --lr_prompt ${lr_prompt} \

done

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