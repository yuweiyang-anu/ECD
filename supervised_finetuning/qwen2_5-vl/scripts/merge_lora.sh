#!/bin/bash

MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
export PYTHONPATH=src:$PYTHONPATH

python src/merge_lora_weights.py \
    --model-path ./output/lora_qwen2_5_vl_on_ECD \
    --model-base $MODEL_NAME  \
    --save-model-path ./output/lora_qwen2_5_vl_on_ECD-merged \
    --safe-serialization