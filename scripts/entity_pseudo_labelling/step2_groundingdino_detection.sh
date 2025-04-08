#!/bin/bash

# ================================
# Step 2: GroundingDINO Detection
# ================================

# === CONFIGURATION ===
DATASET_PATH="/mnt/vita-vldriving/scratch/datasets/DrivingVQA"
IMAGE_FOLDER="${DATASET_PATH}/images/original"
GPT_OUTPUT_JSON="${DATASET_PATH}/splits/train-gpt4o-predicted-20250324.json"

# Output Paths
DATE=$(date +"%Y%m%d")
DINO_OUTPUT_JSON="${DATASET_PATH}/splits/train-gpt4o-predicted-groundingdino-detected-${DATE}.json"

# === STEP 2: Run GroundingDINO Detection ===
echo "Step 2: Running GroundingDINO detection..."
python3 entity_pseudo_labelling/groundingdino_detection.py \
    --input "${GPT_OUTPUT_JSON}" \
    --output "${DINO_OUTPUT_JSON}" \
    --image_folder "${IMAGE_FOLDER}"
