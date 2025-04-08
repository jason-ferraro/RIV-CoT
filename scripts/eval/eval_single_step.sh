#!/bin/bash

# =====================================
# Single Step Evaluation (QP-A, QP-EA, QP-REA, QP-RBEA, QP-IEA, QP-IBEA)
# =====================================

# === CONFIGURATION ===
VARIANT_TYPE="qp-ea"
RUN_NUMBER=1
DATASET_PATH="/mnt/vita-vldriving/scratch/datasets/DrivingVQA"
IMAGE_FOLDER="original"
EXPLANATION_TYPE="original"
MODEL_PATH="/mnt/vita-vldriving/scratch/users/corbiere/models/riv-cot/drivingvqa/llava-onevision-siglip384-qwen2-7b-instruct-ft-${EXPLANATION_TYPE}-${VARIANT_TYPE}-${IMAGE_FOLDER}-10ep-run${RUN_NUMBER}"
echo "MODEL_PATH: ${MODEL_PATH}"

# === RUN EVALUATION ===
python3 llava/eval/eval_single_step.py \
    --model_path ${MODEL_PATH} \
    --image_folder ${DATASET_PATH}/images/${IMAGE_FOLDER} \
    --llava_format_path ${DATASET_PATH}/llava_format/llava-test-${EXPLANATION_TYPE}-${VARIANT_TYPE}-${IMAGE_FOLDER}.json \
    --conv_mode qwen_1_5 \
    --temperature 0. \
    --max_new_tokens 4096

