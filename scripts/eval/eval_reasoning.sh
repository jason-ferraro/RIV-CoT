#!/bin/bash

# =====================================
# Reasoning Evaluation (quality of explanations)
# =====================================

# === CONFIGURATION ===
MODEL_PATH="/mnt/vita-vldriving/scratch/users/corbiere/models/riv-cot/llava-onevision-siglip384-qwen2-7b-instruct-ft-original-qp-rb-rv-ea-crops_normal50-run1"
DATASET_PATH="/mnt/vita-vldriving/scratch/datasets/DrivingVQA"
JUDGE_MODEL="gpt-4o"

python3 llava/eval/evaluate_reasoning.py \
    --result-folder ${MODEL_PATH} \
    --test_json_path ${DATASET_PATH}/splits/test.json \
    --judge_model ${JUDGE_MODEL}
