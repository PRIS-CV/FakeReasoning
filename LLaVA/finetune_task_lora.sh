#!/bin/bash
# global_batchsize = 128

export WANDB_MODE=offline

deepspeed --include="localhost:0,1,2,3,4,5,6,7" llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --output_dir ./checkpoints/fakereasoning-lora \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --mm_vision_select_layer -2 \
    --data_path path_to_forgery_reasoning_cot.json \
    --model_name_or_path path_to_MoF_Models \
    --image_folder path_to_MMFR-Dataset \
    --vision_tower path_to_clip-vit-large-patch14-336 \
    --version v1 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
