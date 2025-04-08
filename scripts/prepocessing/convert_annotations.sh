#!/bin/bash

# ================================
# Convert Annotations to LLaVA Format
# ================================

# === CONFIGURATION ===
DATASET="drivingvqa"
DATASET_PATH="/mnt/vita-vldriving/scratch/datasets/DrivingVQA-v2"
VARIANT_TYPE="qp-a"
EXPLANATION_TYPE="original"
IMAGE_FOLDER="original"
SPLIT="test"

# Convert variant type to uppercase, preserving "D" as lowercase
CAPITALIZED_VARIANT_TYPE=$(echo "$VARIANT_TYPE" | sed 's/\(.*\)/\U\1/;s/D/d/g')

# === RUN CONVERSION SCRIPT ===
echo "Running annotation conversion..."
python3 llava/preprocessing/convert_annotations_to_llava.py \
    --input ${DATASET_PATH}/splits/${SPLIT}.json \
    --dataset ${DATASET} \
    --image_folder ${DATASET_PATH}/images/${IMAGE_FOLDER} \
    --explanation_type ${EXPLANATION_TYPE} \
    --prompt_format ${CAPITALIZED_VARIANT_TYPE} \
    --output_dir ${DATASET_PATH}/llava_format/
