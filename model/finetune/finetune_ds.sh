#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

# MODEL="/mnt/bn/automl-aigc/yatai/Qwen-VL/weight/Qwen-VL-Chat"/
MODEL="/mnt/bn/automl-aigc/yatai/Qwen-VL/result/qwen_alpha_full_llava_mini_1/checkpoint-3000"

# DATA="/mnt/bn/automl-aigc/yatai/Qwen-VL/data/alpha_vcr_ref_f30k_llava.json"
# DATA="/mnt/bn/automl-aigc/yatai/Qwen-VL/data/beta_gpt4v_mix_mini_llava_share.json"
DATA="/mnt/bn/automl-aigc/yatai/Qwen-VL/data/beta_gpt4v_mix_mini.json"

# for i in $range; do
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --bf16 True \
    --fix_vit True \
    --use_llava True \
    --output_dir result/qwen_beta_idadapter_full_llavashare_mini_2 \
    --num_train_epochs 8 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 10 \
    --learning_rate 5e-6 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed finetune/ds_config_zero3.json


