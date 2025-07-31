#!/bin/bash
#SBATCH -p gpu,calmar
#SBATCH -A calmar
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --time=15-00:00:00
#SBATCH --constraint="gpu_40g+"
#SBATCH --exclude=homer-017,homer-011
#SBATCH --output=/scratch/1/user/vnikouli/calmar/log/%j.log
#SBATCH --error=/scratch/1/user/vnikouli/calmar/log/%j.err

#scl enable gcc-toolset-9 bash

#scontrol show job ${SLURM_JOB_ID}   

# source ~/.bashrc

# conda activate /nfs/data/calmar/mlouis/miniconda3/envs/tulu/


#nvidia-smi

export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export PYTHONPATH="/scratch/1/user/vnikouli/projects/bergen:$PYTHONPATH"

open_instruct_dir=/scratch/1/user/vnikouli/projects/bergen/difflora/tulu/open-instruct
eval_dir=/scratch/1/user/vnikouli/projects/bergen/difflora/tulu/evals

export TRITON_CACHE_DIR=/scratch/1/user/vnikouli/triton_cache/


# Default values for flags
model_name=""
suffix=""
merge_flag=false
basename="meta-llama/Llama-3.2-1B-Instruct"
# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --model_name)
      model_name="$2"
      shift 2 # Remove the flag and its value
      ;;
    --merge)
      merge_flag=true
      shift # Remove the flag
      ;;
    --save_suffix)
      suffix="$2"
      shift 2;;
    --basename)
      basename="$2"
      shift 2;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 --model_name <model_name> [--merge]"
      exit 1
      ;;
  esac
done

# Validate required arguments
if [ -z "$model_name" ]; then
  echo "Error: --model_name is required."
  echo "Usage: $0 --model_name <model_name> [--merge]"
  exit 1
fi

# Print received arguments
echo "Model name: $model_name"
echo "Merge flag: $merge_flag"

# For LoRA models we need to merge weights first (open-instruct stuff before going to olmes)
# The tokenizer needs also to be cped. let's go.
if [ "$merge_flag" = true ]; then
  if test -d "$open_instruct_dir/output/${model_name}_MERGED"; then # Check if the directory already exists
    echo "Directory already exists. Skipping merge."
  else
    echo "Merge flag is set. Executing merge-related tasks..."
    cd $open_instruct_dir/open_instruct/
    python merge_lora.py --lora_model_name_or_path $open_instruct_dir/output/${model_name} --base_model_name_or_path $basename --tokenizer_name_or_path $open_instruct_dir/output/${model_name}  --output_dir $open_instruct_dir/output/${model_name}_MERGED
    cp $open_instruct_dir/output/${model_name}/tokenizer.json $open_instruct_dir/output/${model_name}_MERGED/tokenizer.json 
    cp $open_instruct_dir/output/${model_name}/tokenizer_config.json $open_instruct_dir/output/${model_name}_MERGED/tokenizer_config.json 
  fi
  model_name=${model_name}_MERGED

fi




tasks=("tulu_3_dev" "bigcodebench_hard::tulu" "bigcodebench::tulu")
tasks=("drop::olmes" "ifeval::tulu", "olmo_2_heldout::olmes" "arc_challenge::olmes"  "mmlu_pro::olmes" "piqa::olmes" "hellaswag::olmes" "winogrande::olmes" "boolq::olmes" "popqa::tulu" "truthfulqa::tulu"  "winogrande::olmes" "piqa::olmes" "hellaswag::olmes" "arc_easy::olmes" "boolq::olmes" "codex_humaneval::tulu" "gsm8k::tulu")
#tasks=("olmo_2_heldout::olmes")
#tasks=("truthfulqa::tulu", "hellaswag::olmes", "arc_challenge::olmes", "arc")

#tasks=("codex_humaneval::tulu" "arc_challenge::olmes")
#tasks=("codex_humaneval::tulu" "arc_challenge::olmes"  "popqa::tulu" "truthfulqa::tulu"  "winogrande::olmes" "piqa::olmes" "hellaswag::olmes" "arc_easy::olmes" "boolq::olmes")
for task in "${tasks[@]}"; do
  echo "$eval_dir/${task//:/_}_${model_name}${suffix}/metrics.json" 
  if test -e "$eval_dir/${task//:/_}_${model_name}${suffix}/metrics.json"; then
      echo "Metrics already exist for ${task//:/_}_${model_name}${suffix}"
  else
      echo "run eval"
      sbatch -p gpu-be run_olmes.sh $model_name $task $suffix
  fi
done