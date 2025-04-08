#!/bin/bash

# ================================
# Zero-shot Evaluation
# e.g. Qwen2.5-VL-7B-Instruct, Qwen2.5-VL-72B-Instruct
#      Llama-3.2-11B-Vision-Instruct, Llama-3.2-90B-Vision-Instruct
# ================================

# === CONFIGURATION ===
DATASET_PATH="/mnt/vita-vldriving/scratch/datasets/DrivingVQA"
VARIANT_TYPE="qp-a"
MODEL="Llama-3.2-11B-Vision-Instruct"

# === RUN EVALUATION ===
python3 llava/eval/eval_zeroshot.py \
    --mode ${MODEL} \
    --image_folder ${DATASET_PATH}/images/original \
    --llava_format_path ${DATASET_PATH}/llava_format/llava-test-original-${VARIANT_TYPE}-original.json \
    --temperature 0. \
    --max_new_tokens 4096

