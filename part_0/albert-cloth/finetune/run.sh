#!/usr/bin/zsh
#----------------------------------------------------------------------
# CONFIG
MODEL_LOCATE=/mnt/shared/engine210/LE/model/albert-cloth
TRAIN_DATA=CLOTH # can choose "ELE", "CLOTH"
DEV_DATA=dev     # can choose "dev", "single", "multi"
DO_AUTO_EVAL=false
# hyperparameters for train
# recommand to read train.sh before run
EPOCH_NUM=1
LEARNING_RATE=1e-5
SCHEDULER="cosine"  # can choose "cosine", "linear"
#----------------------------------------------------------------------
main(){

    file_security_check
    data_enable

    # do train
    ./train.sh $LEARNING_RATE $EPOCH_NUM $MODEL_LOCATE $SCHEDULER

    mv $MODEL_LOCATE/pytorch_model.bin $MODEL_LOCATE/model.bin
    # do auto_eval
    if [ $DO_AUTO_EVAL ]; then
        ./auto_eval.sh $MODEL_LOCATE $EPOCH_NUM
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
    # pytorch_model.bin
    if [ -f $MODEL_LOCATE/pytorch_model.bin ]; then
        echo "WARRING: $MODEL_LOCATE/pytorch_model.bin already exit before run, rename to model.bin"
        mv $MODEL_LOCATE/pytorch_model.bin $MODEL_LOCATE/model.bin
    fi 
    # 
    
}

data_enable(){
    # move data.pt
    cp ./data/$TRAIN_DATA-train-albert-xxlarge-v2.pt ./data/train-albert-xxlarge-v2.pt
    cp ./data/ELE-$DEV_DATA-albert-xxlarge-v2.pt     ./data/dev-albert-xxlarge-v2.pt
    # move model.bin
    mv $MODEL_LOCATE/model.bin $MODEL_LOCATE/pytorch_model.bin
}

main