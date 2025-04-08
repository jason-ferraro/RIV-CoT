#!/bin/bash

# ================================
# Multi-Step Evaluation (RIV-CoT)
# ================================

# === CONFIGURATION ===
RUN_NUMBER=1
DATASET_PATH="/mnt/vita-vldriving/scratch/datasets/DrivingVQA"
CROP_METHOD="normal_50"
MODEL_PATH="/mnt/vita-vldriving/scratch/users/corbiere/models/riv-cot/aokvqa/llava-onevision-siglip384-qwen2-7b-instruct-ft-interleaved-riv-cot-crops_${CROP_METHOD}-10ep-run${RUN_NUMBER}"
echo "MODEL_PATH: ${MODEL_PATH}"

# === RUN EVALUATION ===
python3 llava/eval/eval_riv_cot.py \
    --model_path ${MODEL_PATH} \
    --image_folder ${DATASET_PATH}/images/original \
    --llava_format_path ${DATASET_PATH}/llava_format/llava-test-interleaved-riv-cot-crops_${CROP_METHOD}.json \
    --crop_method ${CROP_METHOD} \
    --conv_mode qwen_1_5 \
    --temperature 0. \
    --max_new_tokens 4096

