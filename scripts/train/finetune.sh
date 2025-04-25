#!/bin/bash

# ================================
# Fine-Tuning Configuration
# ================================

# Export environment variables for optimal performance
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

# === CONFIGURATION ===
VARIANT_TYPE="qp-a"
RUN_NUMBER=1
IMAGE_FOLDER="original"
EXPLANATION_TYPE="original"
PROMPT_VERSION="qwen_1_5"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
DATASET_PATH="/mnt/vita-vldriving/scratch/datasets/DrivingVQA"
OUTPUT_FOLDER="/mnt/vita-vldriving/scratch/users/corbiere/models/llava-next/drivingvqa"
RUN_NAME="llava-onevision-siglip384-qwen2-7b-instruct-ft-${EXPLANATION_TYPE}-${VARIANT_TYPE}-${IMAGE_FOLDER}-10ep-run${RUN_NUMBER}"
echo "RUN_NAME: ${RUN_NAME}"

# Get the number of GPUs available
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs available: ${NUM_GPUS}"

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node=${NUM_GPUS} --nnodes=1 --node_rank=0  \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path lmms-lab/llava-onevision-qwen2-7b-ov \
    --version ${PROMPT_VERSION} \
    --data_path ${DATASET_PATH}/llava_format/llava-train-${EXPLANATION_TYPE}-${VARIANT_TYPE}-${IMAGE_FOLDER}.json \
    --image_folder ${DATASET_PATH}/images/${IMAGE_FOLDER} \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio pad \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name ${RUN_NAME} \
    --output_dir ${OUTPUT_FOLDER}/${RUN_NAME} \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --seed $((RUN_NUMBER * 42))
exit 0;

# You can delete the sdpa attn_implementation if you want to use flash attn
