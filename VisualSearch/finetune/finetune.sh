#!/bin/bash

# Set environment variables (adjust paths as needed)
export DATA_DIR="/home/a3ilab01/dataset/test_janus/deepseek-janus-pro-lora/trump"
export PRETRAINED_MODEL_PATH="/home/a3ilab01/Project/pretrain_models/Janus-Pro-7B"
export OUTPUT_DIR="./test1"
export BATCH_SIZE=2
export MAX_EPOCHS=1
export LR=3e-4
export USER_QUESTION="请描述一下这张图片的内容." 
export OPTIMIZER_NAME="AdamW"
export LORA_CONFIG='{
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj"],
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}'
export TRAINING_ARGS='{
    "fp16": True,
    "max_grad_norm": 1.0,
    "save_strategy": "epoch",
    "evaluation_strategy": "no",
    "logging_steps": 50,
    "save_total_limit": 2,
    "remove_unused_columns": False
}'

# Set path to DeepSpeed ZeRO-3 configuration file
export DEEPSPEED_CONFIG_PATH="zero3.json"  # Update with the actual path

# Launch training using DeepSpeed
deepspeed --num_gpus=2 finetune.py \
    --pretrained_model_path $PRETRAINED_MODEL_PATH \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --max_epochs $MAX_EPOCHS \
    --lr $LR \
    --user_question "$USER_QUESTION" \
    --optimizer_name $OPTIMIZER_NAME \
    --lora_config "$LORA_CONFIG" \
    --training_args "$TRAINING_ARGS" \
    --deepspeed $DEEPSPEED_CONFIG_PATH
