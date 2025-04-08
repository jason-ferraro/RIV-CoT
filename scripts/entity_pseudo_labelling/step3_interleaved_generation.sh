#!/bin/bash

# ================================
# Step 3: Interleaved Generation
# ================================

# === CONFIGURATION ===
DATASET_PATH="/mnt/vita-vldriving/scratch/datasets/DrivingVQA"
INPUT_JSON="${DATASET_PATH}/splits/train.json"
IMAGE_FOLDER="${DATASET_PATH}/images/original"
MODEL="gpt-4o"
SAVE_EVERY=5

# Output Paths
DATE=$(date +"%Y%m%d")
GPT_OUTPUT_JSON="${DATASET_PATH}/splits/train-interleaved-${DATE}.json"

# === STEP 1: Run interleaved generation ===
echo "Step 3: Running interleaved generation..."
python3 entity_pseudo_labelling/interleaved_generation.py \
    --input "${INPUT_JSON}" \
    --output "${GPT_OUTPUT_JSON}" \
    --image_folder "${IMAGE_FOLDER}" \
    --model "${MODEL}" \
    --save_every "${SAVE_EVERY}"
