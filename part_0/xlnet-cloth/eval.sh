CUDA_VISIBLE_DEVICES=0,1,2,3 python -u  \
main.py \
--output_dir debug-exp/ \
--do_eval \
--eval_batch_size 400 \
--bert_model albert-xxlarge-v2 \
--model_load_dir $1 \
--model_save_dir $1 \
--scheduler $2
