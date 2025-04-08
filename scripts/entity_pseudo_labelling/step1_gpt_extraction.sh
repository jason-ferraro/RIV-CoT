#!/bin/bash

# ================================
# Step 1: Entity Extraction
# ================================

# === CONFIGURATION ===
DATASET="drivingvqa"
DATASET_PATH="/mnt/vita-vldriving/scratch/datasets/DrivingVQA"
INPUT_JSON="${DATASET_PATH}/splits/train.json"
IMAGE_FOLDER="${DATASET_PATH}/images/original"
SAVE_EVERY=5

# Output Paths
DATE=$(date +"%Y%m%d")
GPT_OUTPUT_JSON="${DATASET_PATH}/splits/train-gpt4o-predicted-${DATE}.json"

# === STEP 1: Run Entity Extraction ===
echo "Step 1: Running entity extraction..."
python3 entity_pseudo_labelling/gpt_entity_extraction.py \
    --input "${INPUT_JSON}" \
    --output "${GPT_OUTPUT_JSON}" \
    --dataset "${DATASET}" \
    --image_folder "${IMAGE_FOLDER}" \
    --save_every "${SAVE_EVERY}"
