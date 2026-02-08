#!/bin/bash

#changable
TRAINING_DATASETS="DTD MNIST EuroSAT Flowers"
FOLDER_PATH="ckpt/clean/5000_iter/lora_finetune_test_continue"
ITERATIONS=5000
JOB_NAME="lora_finetune"
#static
PREV_JOB_ID=""
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

TRAIN_CMD="sbatch --parsable --job-name=${JOB_NAME}_train_base $SCRIPT_DIR/train.sh DTD $FOLDER_PATH/base 0"    
PREV_JOB_ID=$($TRAIN_CMD)
echo "Initializing base model - Job ID: $PREV_JOB_ID"

PREV_MODEL_PATH="$FOLDER_PATH/base/DTD.pth"
FOLDER_PATH="$FOLDER_PATH/base"

for dataset in $TRAINING_DATASETS
do
    FOLDER_PATH=$FOLDER_PATH/$dataset
    
    TRAIN_CMD="sbatch --parsable --job-name=${JOB_NAME}_train_${dataset} --dependency=afterok:$PREV_JOB_ID $SCRIPT_DIR/train.sh $dataset $FOLDER_PATH $PREV_MODEL_PATH $ITERATIONS"
    PREV_JOB_ID=$($TRAIN_CMD)
    echo "Queued training - dataset: $dataset - Job ID: $PREV_JOB_ID"

    PREV_MODEL_PATH=$FOLDER_PATH/$dataset.pth
done