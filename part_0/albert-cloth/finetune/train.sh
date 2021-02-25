#!/usr/bin/zsh
CUDA_VISIBLE_DEVICES=0,1,2,3 python3.6 -u main.py \
--output_dir debug-exp/ \
--do_train \
--train_batch_size 24 \
--eval_batch_size 128 \
--bert_model albert-xxlarge-v2 \
--learning_rate 1e-5 \
--num_train_epochs 2 \
--model_load_dir /mnt/shared/engine210/LE/model/albert \
--model_save_dir /mnt/shared/engine210/LE/model/albert-cloth \
--scheduler "cosine"