#!/bin/bash

# python3 train/tulu/open-instruct/open_instruct/finetune.py train/tulu/scripts/configs/config_lora_tulu3.yaml

# source ~/.bashrc


# conda activate difflora

nvidia-smi

export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

cd ./train/tulu/open-instruct

NUM_GPUS=1
echo "Number of GPUs allocated: ${NUM_GPUS}"

config_file_name=$1

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes ${NUM_GPUS} \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage2.conf \ 
    # --main_process_port $((12000 + $SLURM_JOB_ID % 1000)) \
    open_instruct/finetune.py \
    ../scripts/configs/${config_file_name}.yaml \
    "$@" 
