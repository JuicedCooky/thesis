#!/bin/bash

#changable
TRAINING_DATASETS=(DTD MNIST EuroSAT Flowers)
FOLDER_PATH="ckpt/clean/5000_iter/zscl_continue"
ITERATIONS=5000
JOB_NAME="zscl"

#static
PREV_JOB_ID=""
PREV_GRADIENT_PATH=""
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

TRAIN_CMD=(
    sbatch
    --parsable
    "--job-name=${JOB_NAME}_train_base"
    "--time=00:15:00"
    "${SCRIPT_DIR}/train.sh"
    "DTD"
    "${FOLDER_PATH}/base"
    ""
    "0"
    ""
)
PREV_JOB_ID=$( "${TRAIN_CMD[@]}" | tr -d ' \n' )
echo "Initializing base model - Job ID: $PREV_JOB_ID"

sbatch --job-name=${JOB_NAME}_base_eval --dependency=afterok:${PREV_JOB_ID} ${SCRIPT_DIR}/evaluate.sh ${FOLDER_PATH}/base/DTD.pth
echo "Queued eval - base model"


PREV_MODEL_PATH="$FOLDER_PATH/base/DTD.pth"
FOLDER_PATH="$FOLDER_PATH/base"

for i in "${!TRAINING_DATASETS[@]}"
do
    dataset=${TRAINING_DATASETS[$i]}
    FOLDER_PATH=$FOLDER_PATH/$dataset

    TRAIN_CMD=(
        sbatch
        --parsable
        --job-name=${JOB_NAME}_train_${dataset}
        --dependency=afterok:$PREV_JOB_ID
        $SCRIPT_DIR/train.sh
        $dataset
        $FOLDER_PATH
        $PREV_MODEL_PATH
        $ITERATIONS
        "$PREV_GRADIENT_PATH"
    )

    PREV_JOB_ID=$( "${TRAIN_CMD[@]}" | tr -d ' \n' )
    echo "Queued training - dataset: $dataset - Job ID: $PREV_JOB_ID"

    PREV_MODEL_PATH=$FOLDER_PATH/$dataset.pth
    PREV_GRADIENT_PATH=$FOLDER_PATH/grad_${dataset}.pth

    sbatch --job-name=${JOB_NAME}_${dataset}_eval --dependency=afterok:${PREV_JOB_ID} ${SCRIPT_DIR}/evaluate.sh ${PREV_MODEL_PATH}
    echo "Queued eval - model:${dataset}"
done
