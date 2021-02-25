#!/usr/bin/zsh

# save config
echo "CUDA_VISIBLE_DEVICES=0,1,2,3 python -u main.py \ 
--output_dir debug-exp/ \ 
--do_train \ 
--train_batch_size 24 \ 
--eval_batch_size 128 \ 
--bert_model albert-xxlarge-v2 \ 
--learning_rate $1 \ 
--num_train_epochs $2 \ 
--model_load_dir $3 \ 
--model_save_dir $3 \ 
--scheduler $4 " > $3/run_config.txt

# -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr 127.0.0.1 --master_port 29504 
# CUDA_VISIBLE_DEVICES=0,1,2,3 mpirun -H localhost:4 --mca btl_openib_allow_ib 1
CUDA_VISIBLE_DEVICES=0,1,2,3 python  -m torch.distributed.launch  \
--nproc_per_node=4 --nnodes=4 --node_rank=3 --master_addr 10.18.18.1 --master_port 11222 \
main.py \
--output_dir debug-exp/ \
--do_train \
--train_batch_size 4 \
--eval_batch_size 128 \
--bert_model albert-xxlarge-v2 \
--learning_rate $1 \
--num_train_epochs $2 \
--model_load_dir $3 \
--model_save_dir $3 \
--scheduler $4 \
--local_rank 3 \
--do_eval


#save model as pytorch_model_(epoch_num).bin