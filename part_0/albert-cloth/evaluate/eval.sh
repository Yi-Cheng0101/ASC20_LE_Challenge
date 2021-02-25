CUDA_VISIBLE_DEVICES=0,1,2,3 python3.6 -u main.py \
--output_dir debug-exp/ \
--do_eval \
--eval_batch_size 100 \
--bert_model albert-xxlarge-v2 \
--model_load_dir /mnt/shared/engine210/LE/model/albert-cloth
