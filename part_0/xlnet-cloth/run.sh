#!/usr/bin/zsh
#----------------------------------------------------------------------
#CONFIG
# model location
MODEL_LOCATE=/mnt/shared/engine210/LE/model/albert-cloth

# choose which train, eval data    
TRAIN_DATA=CLOTH # can choose "ELE", "CLOTH"
DEV_DATA=dev     # can choose "dev", "single", "multi"
DO_AUTO_EVAL=true

# config hyperparameter
EPOCH_NUM=1
LEARNING_RATE=1e-5
SCHEDULER="linear"  # can choose "cosine", "linear"

# DDP config
NODE_RANK=$1
echo "$1"
export OMP_NUM_THREADS=4

#----------------------------------------------------------------------
main(){
    if [[ $NODE_RANK == 0 ]]; then
        file_security_check
        data_enable
    fi
    do_train
    if [[ $NODE_RANK == 0 ]]; then
        data_disable
        do_eval
    fi
}
do_train(){
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3.6 -m torch.distributed.launch  \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=$NODE_RANK \
    --master_addr 10.18.18.1 \
    --master_port 11228 \
    main.py \
    --output_dir debug-exp/ \
    --do_train \
    --train_batch_size 4 \
    --eval_batch_size 128 \
    --bert_model albert-xxlarge-v2 \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $EPOCH_NUM \
    --model_load_dir $MODEL_LOCATE \
    --model_save_dir $MODEL_LOCATE \
    --scheduler $SCHEDULER \
    --local_rank=$NODE_RANK
}
do_eval(){
    if [[ $DO_AUTO_EVAL == true ]]; then
        ./auto_eval.sh $MODEL_LOCATE $EPOCH_NUM $SCHEDULER
    fi
}
file_security_check(){
    # ensure valid MODEL_LOCATE
    if [ ! -d $MODEL_LOCATE ]; then
        echo "ERROR: directory $MODEL_LOCATE is NOT exist"
        exit
    fi
    # not allow to have old checkpoint
    if [ -f $MODEL_LOCATE/pytorch_model0.bin ]; then
        echo "ERROR: pytorch_model0.bin is in your dir, please new a dir or remove old ckpt"
        exit
    fi
}

data_enable(){
    # move data.pt
    cp ./data/$TRAIN_DATA-train-albert-xxlarge-v2.pt ./data/train-albert-xxlarge-v2.pt
    cp ./data/ELE-$DEV_DATA-albert-xxlarge-v2.pt     ./data/dev-albert-xxlarge-v2.pt
    # move model.bin
    if [ -f $MODEL_LOCATE/model.bin ]; then
        mv $MODEL_LOCATE/model.bin $MODEL_LOCATE/pytorch_model.bin
    fi 
}
data_disable(){
    mv $MODEL_LOCATE/pytorch_model.bin $MODEL_LOCATE/model.bin
}
main