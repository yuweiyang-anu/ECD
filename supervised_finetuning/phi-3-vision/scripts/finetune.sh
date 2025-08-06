#!/bin/bash

MODEL_NAME="microsoft/Phi-3-vision-128k-instruct"

export PYTHONPATH=src:$PYTHONPATH
export WANDB_PROJECT="VLM-SFT-on-ECD"
export WANDB_API_KEY="your_wandb_api_key"  # need more setup

deepspeed src/training/train.py \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path ECD/datasets/ECD_qa_data_all_formatted.json \
    --image_folder ECD/datasets/images/ \
    --tune_img_projector True \
    --freeze_vision_tower True \
    --freeze_llm False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/full_sft_phi3v_on_ECD \
    --num_crops 16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 24 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-6 \
    --projector_lr 5e-8 \
    --vision_lr 0.0 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to wandb \
    --run_name full_sft_phi3v_on_ECD \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 4