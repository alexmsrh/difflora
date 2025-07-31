#!/bin/bash

# source ~/.bashrc

conda activate difflora

nvidia-smi

export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
# export PYTHONPATH="/scratch/1/user/vnikouli/projects/bergen:$PYTHONPATH"

open_instruct_dir=./tulu/open-instruct
eval_dir=./tulu/evals

# export TRITON_CACHE_DIR=/scratch/1/user/vnikouli/triton_cache/
model_name=$1
task=$2
suffix=$3
echo "Evaluating $model_name"
echo "Task: $task"


if [[ "$model_name" == *"meta-"* || "$model_name" == *"allenai"* ]]; then
    echo "Running with meta or allenai model"
        olmes --model ${model_name} --task $task --output-dir $eval_dir/"${task//:/_}_${model_name//\//-}${suffix}" --model-args '{"trust_remote_code": true, "max-length": 4096}' --batch-size 16
        olmes --model ${model_name} --task $task --output-dir $eval_dir/"${task//:/_}_${model_name//\//-}${suffix}" --model-args '{"trust_remote_code": true, "max-length": 4096}' --batch-size 4
        olmes --model ${model_name} --task $task --output-dir $eval_dir/"${task//:/_}_${model_name//\//-}${suffix}" --model-args '{"trust_remote_code": true, "max-length": 4096}' --batch-size 1
    else
       echo "Running with other model"
        olmes --model $open_instruct_dir/output/${model_name} --task $task --output-dir $eval_dir/${task//:/_}_${model_name}${suffix} --model-args '{"trust_remote_code": true, "max-length": 4096}' --batch-size 16
        olmes --model $open_instruct_dir/output/${model_name} --task $task --output-dir $eval_dir/${task//:/_}_${model_name}${suffix} --model-args '{"trust_remote_code": true, "max-length": 4096}' --batch-size 4
        olmes --model $open_instruct_dir/output/${model_name} --task $task --output-dir $eval_dir/${task//:/_}_${model_name}${suffix} --model-args '{"trust_remote_code": true, "max-length": 4096}' --batch-size 1
    
fi
