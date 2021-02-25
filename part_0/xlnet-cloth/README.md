# Test ALBERT on CLOTH Dataset

This repository test the ALBERT model on the CLOTH and ELE Dataset. The framework of implementation is provided by [Transformers](https://github.com/huggingface/transformers). The original code is forked from [bert-cloth](https://github.com/laiguokun/bert-cloth).

## Code Usage

### Data

Need to copy from '/mnt/shared/michael1017/CLOTH_ESSENTIAL' for first run
```
cp -r /mnt/shared/michael1017/CLOTH_ESSENTIAL/* /path/to/ALBERT-CLOTH
```

## Finetune Model with Multi-Node
<B style="color:yellow">
    WARRING: setup CONFIG and DO_TRAIN function in run.sh before run
</B> <br>
Since DistributedDataParallel (called DDP) have better performance than DataParallel, we only show how to run DDP and setup CONFIG and DO_TRAIN  

* CONFIG  
    1. config model location
    2. choose which train, eval data
    3. config hyperparameter
    4. DDP config
* DO_TRAIN
    1. nproc_per_node means how many GPUs available per_node
    2. nnodes means  how many nodes available
    3. master_addr can be host infiniband ip 
    4. master_port is an unused port number

for example, run DDP with 4 nodes and 4 gpu per node
* CONFIG  
    ```
    # model location
    MODEL_LOCATE=/path/to/model.bin

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
    export OMP_NUM_THREADS=4
    ```
* DO_TRAIN
    ```
    --nproc_per_node=4 \
    --nnodes=4 \
    --node_rank=$NODE_RANK \
    --master_addr 10.18.18.1 \
    --master_port 1122 \
    ```
in art1
```
./run.sh 0
```
in art2
```
./run.sh 1
```
in art3
```
./run.sh 2
```
in art4
```
./run.sh 3
```