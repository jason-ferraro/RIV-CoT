#!/bin/bash

# ================================
# Preprocess Image Patches
# ================================

# === CONFIGURATION ===
DATASET_PATH="/mnt/vita-vldriving/scratch/datasets/DrivingVQA"
SPLIT="trainval"
CROP_METHOD="normal"
EXTEND_PERCENT=50

# === RUN IMAGE CROP EXTRACTION ===
echo "Running image crop extraction..."
python3 llava/preprocessing/extract_image_crops.py \
    --input ${DATASET_PATH}/test/splits/${SPLIT}.json \
    --image_folder ${DATASET_PATH}/images/original/ \
    --output_dir ${DATASET_PATH}/test/images \
    --crop_method ${CROP_METHOD} \
    --extend_by_percent ${EXTEND_PERCENT}
