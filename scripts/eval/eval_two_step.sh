#!/bin/bash

# ================================
# Two-Step Evaluation (QP-RB-RV-EA)
# ================================

# === CONFIGURATION ===
RUN_NUMBER=1
DATASET_PATH="/mnt/vita-vldriving/scratch/datasets/DrivingVQA"
CROP_METHOD="normal_50"
EXPLANATION_TYPE="original"
MODEL_PATH="/mnt/vita-vldriving/scratch/users/corbiere/models/riv-cot/drivingvqa/llava-onevision-siglip384-qwen2-7b-instruct-ft-${EXPLANATION_TYPE}-qp-rb-rv-ea-crops_${CROP_METHOD}-10ep-run${RUN_NUMBER}"
echo "MODEL_PATH: ${MODEL_PATH}"

# === RUN EVALUATION ===
# First step: predict list of entities and bbox + extract visual crops
python3 llava/eval/predict_entities.py \
    --model_path ${MODEL_PATH} \
    --dataset drivingvqa \
    --image_folder ${DATASET_PATH}/images/original \
    --llava_format_path ${DATASET_PATH}/llava_format/llava-test-${EXPLANATION_TYPE}-qp-rb-rv-ea-crops_${CROP_METHOD}.json \
    --test_json_path ${DATASET_PATH}/splits/test.json \
    --crop_method ${CROP_METHOD} \
    --conv_mode qwen_1_5 \
    --temperature 0. \
    --max_new_tokens 4096

# Second step: predict answer based on predicted visual crops
python3 llava/eval/eval_single_step.py \
    --model_path ${MODEL_PATH} \
    --image_folder ${MODEL_PATH}/eval_${CROP_METHOD}/crops_${CROP_METHOD} \
    --llava_format_path ${MODEL_PATH}/eval_${CROP_METHOD}/llava-pred_bboxes-${EXPLANATION_TYPE}-qp-rb-rv-ea-crops_${CROP_METHOD}.json \
    --conv_mode qwen_1_5 \
    --temperature 0. \
    --max_new_tokens 4096

