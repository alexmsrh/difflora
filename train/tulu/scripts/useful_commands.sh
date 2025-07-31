sbatch --gres=gpu:1 -J tulu_debug ./launch_tulu_sft.sh

###########################################
#### LLAMA 8B CLASSSICO #########
###########################################
 # LORA RUN 8B Lora rank=64
sbatch --gres=gpu:1 -J llama_3_8B_lora_r64 ./launch_tulu_sft.sh config_lora \
 --model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct \
 --tokenizer_name=meta-llama/Meta-Llama-3-8B-Instruct \
 --output_dir=output/llama_3_8B_lora_r64 \
 --lora_rank=64 \
 --lora_alpha=128


 # LORA RUN 8B Lora rank=512
sbatch --gres=gpu:1 -J llama_3_8b_lora_r512 ./launch_tulu_sft.sh config_lora \
 --model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct \
 --tokenizer_name=meta-llama/Meta-Llama-3-8B-Instruct \
 --output_dir=output/llama_3_8b_lora_r512 \
 --lora_rank=512 \
 --lora_alpha=1024 

###########################################
#### LLAMA 1B CLASSSICO #########
###########################################
# LORA RUN 1B Lora rank=64
sbatch --gres=gpu:1 -J llama_32_1B_lora_r64 ./launch_tulu_sft.sh config_lora \
 --model_name_or_path=meta-llama/Llama-3.2-1B-Instruct \
  --tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
  --output_dir=output/llama_32_1B_lora_r64 \
  --per_device_train_batch_size=2 \
  --gradient_accumulation_steps=32 \
  --lora_rank=64 \
  --lora_alpha=128 # 2017 DONE

# LORA RUN 1B Lora rank=64, Lora on Q and K only AS SHOULD BE
sbatch --gres=gpu:1 -J llama_32_1B_lora_r64_qk ./launch_tulu_sft.sh config_lora \
 --model_name_or_path=meta-llama/Llama-3.2-1B-Instruct \
  --tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
  --output_dir=output/llama_32_1B_lora_r64_qk \
  --per_device_train_batch_size=2 \
  --gradient_accumulation_steps=32 \
  --lora_rank=64 \
  --lora_alpha=128 \
  --lora_modules="['q_proj','k_proj']" # 2889 DONE

  # LORA RUN 1B Lora rank=64, Lora on Q, K, O and V, HALF RANK
  sbatch --gres=gpu:1 -J llama_32_1B_lora_r64_qkov ./launch_tulu_sft.sh config_lora \
   --model_name_or_path=meta-llama/Llama-3.2-1B-Instruct \
    --tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
    --output_dir=output/llama_32_1B_lora_r64_qkov \
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=32 \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_modules="['q_proj','k_proj', 'o_proj', 'v_proj']" # 2891 DONE


  # LORA RUN 1B Lora rank=64, tulu3, Lora on Q and K only AS SHOULD BE
sbatch --gres=gpu:1 -J llama_32_1B_lora_r64_qk_tulu3 ./launch_tulu_sft.sh config_lora_tulu3 \
 --model_name_or_path=meta-llama/Llama-3.2-1B-Instruct \
  --tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
  --output_dir=output/llama_32_1B_lora_r64_qk_tulu3 \
  --per_device_train_batch_size=2 \
  --gradient_accumulation_steps=32 \
  --lora_rank=64 \
  --lora_alpha=128 \
  --lora_modules="['q_proj','k_proj']" # 2890 DONE


# LORA RUN 1B Lora rank=64, tulu3, Lora on Q and K only AS SHOULD BE
sbatch --gres=gpu:1 -J llama_32_1B_full_tulu3 ./launch_tulu_sft.sh config_lora_tulu3 \
--model_name_or_path=meta-llama/Llama-3.2-1B-Instruct \
 --tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
 --output_dir=output/llama_32_1B_full_tulu3 \
 --per_device_train_batch_size=2 \
 --gradient_accumulation_steps=32 \
 --use_lora=false \
 --learning_rate=5e-6 # 3869
  
  # 2890 DONE


  # LORA RUN 1B Lora rank=64, tulu3, Lora on Q, K, O and V, HALF RANK
sbatch --gres=gpu:1 -J llama_32_1B_lora_r64_qkov_tulu3 ./launch_tulu_sft.sh config_lora_tulu3 \
--model_name_or_path=meta-llama/Llama-3.2-1B-Instruct \
 --tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
 --output_dir=output/llama_32_1B_lora_r64_qkov_tulu3 \
 --per_device_train_batch_size=2 \
 --gradient_accumulation_steps=32 \
 --lora_rank=32 \
 --lora_alpha=64 \
 --lora_modules="['q_proj','k_proj', 'o_proj', 'v_proj']" # 2892 DONE


# LORA RUN 1B Lora rank=512
sbatch --gres=gpu:1 -J llama_32_1B_lora_r512 ./launch_tulu_sft.sh config_lora \
  --model_name_or_path=meta-llama/Llama-3.2-1B-Instruct \
  --tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
  --output_dir=output/llama_32_1B_lora_r512 \
  --per_device_train_batch_size=2 \
  --gradient_accumulation_steps=32 \
  --lora_rank=512 \
  --lora_alpha=1024 # 2046 DONE

# LORA RUN 1B Lora rank=512, Lora on Q and K only AS SHOULD BE
sbatch --gres=gpu:1 -J llama_32_1B_lora_r512_qk ./launch_tulu_sft.sh config_lora \
--model_name_or_path=meta-llama/Llama-3.2-1B-Instruct \
--tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
--output_dir=output/llama_32_1B_lora_r512_qk \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=32 \
--lora_rank=512 \
--lora_alpha=1024 \
--lora_modules="['q_proj','k_proj']" # 2909 DONE


##################################################
# DIFF  TRANSFORMER ON 1B model ##################
##################################################

# LAMBDA = 0.1, r=64
sbatch --gres=gpu:1 -J llama_32_1B_lamb_0.1_r64_fix ./launch_tulu_sft.sh config_lora \
--tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
--diff_transformer_base_model_name=meta-llama/Llama-3.2-1B-Instruct \
--output_dir=output/llama_32_1B_lamb_0.1_r64_fix \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=32 \
--diff_transformer=true \
--lora_alpha=128 \
--lora_rank=64 \
--groupnorm=false \
--diff_attn_lambda=0.1 \
--gradient_checkpointing=true \
--use_lora=false # 2082, 2734 (fix) DONE

# LAMBDA = 0.1, r=64, V and O trained as well, HALF RANK
sbatch --gres=gpu:1 -J llama_32_1B_lamb_0.1_r64_qkov_fix ./launch_tulu_sft.sh config_lora \
--tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
--diff_transformer_base_model_name=meta-llama/Llama-3.2-1B-Instruct \
--output_dir=output/llama_32_1B_lamb_0.1_r64_qkov_fix \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=32 \
--diff_transformer=true \
--lora_alpha=64 \
--lora_rank=32 \
--lora_v=true \
--lora_o=true \
--groupnorm=false \
--diff_attn_lambda=0.1 \
--gradient_checkpointing=true \
--use_lora=false # 2903 DONE


# LAMBDA = 0.1, r=512
sbatch --gres=gpu:1 -J llama_32_1B_lamb_0.1_r512_fix ./launch_tulu_sft.sh config_lora \
--tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
--diff_transformer_base_model_name=meta-llama/Llama-3.2-1B-Instruct \
--output_dir=output/llama_32_1B_lamb_0.1_r512_fix \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=32 \
--diff_transformer=true \
--lora_alpha=1024 \
--lora_rank=512 \
--groupnorm=false \
--diff_attn_lambda=0.1 \
--gradient_checkpointing=true \
--use_lora=false # 2083, 2735 DONE

# LAMBDA = 0.1, r=256, both terms
sbatch --gres=gpu:1 -J llama_32_1B_lamb_0.1_b256_fix ./launch_tulu_sft.sh config_lora \
--tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
--diff_transformer_base_model_name=meta-llama/Llama-3.2-1B-Instruct \
--output_dir=output/llama_32_1B_lamb_0.1_b256_fix \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=32 \
--diff_transformer=true \
--lora_alpha=512 \
--lora_rank=256 \
--groupnorm=false \
--diff_attn_lambda=0.1 \
--gradient_checkpointing=true \
--use_lora=false \
--lora_negative_term_only=false # 2736 DONE

# LAMBDA = 0.1, r=64, TULU 3
sbatch --gres=gpu:1 -J llama_32_1B_lamb_0.1_r64_tulu3_fix ./launch_tulu_sft.sh config_lora_tulu3 \
--tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
--diff_transformer_base_model_name=meta-llama/Llama-3.2-1B-Instruct \
--output_dir=output/llama_32_1B_lamb_0.1_r64_tulu3_fix \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=32 \
--diff_transformer=true \
--lora_alpha=128 \
--lora_rank=64 \
--groupnorm=false \
--diff_attn_lambda=0.1 \
--gradient_checkpointing=true \
--use_lora=false # 3002 DONE

# LAMBDA = 0.1, r=64, TULU 3
sbatch --gres=gpu:1 -J llama_32_1B_lamb_0.1_b32_tulu3_fix ./launch_tulu_sft.sh config_lora_tulu3 \
--tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
--diff_transformer_base_model_name=meta-llama/Llama-3.2-1B-Instruct \
--output_dir=output/llama_32_1B_lamb_0.1_b32_tulu3_fix \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=32 \
--diff_transformer=true \
--lora_alpha=64 \
--lora_rank=32 \
--groupnorm=false \
--diff_attn_lambda=0.1 \
--gradient_checkpointing=true \
--use_lora=false \
--lora_negative_term_only=false # 4665


# SAME AS ABOVE BUT 'EAGER'
sbatch --gres=gpu:1 -J llama_32_1B_lamb_0.1_b32_tulu3_eager ./launch_tulu_sft.sh config_lora_tulu3 \
--tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
--diff_transformer_base_model_name=meta-llama/Llama-3.2-1B-Instruct \
--output_dir=output/llama_32_1B_lamb_0.1_b32_tulu3_eager \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=32 \
--diff_transformer=true \
--diff_attn_implementation='eager' \
--lora_alpha=64 \
--lora_rank=32 \
--groupnorm=false \
--diff_attn_lambda=0.1 \
--gradient_checkpointing=true \
--use_lora=false \
--lora_negative_term_only=false # 5508


sbatch --gres=gpu:1 -J llama_32_1B_lamb_0.1_fullR_tulu3_fix ./launch_tulu_sft.sh config_lora_tulu3 \
--tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
--diff_transformer_base_model_name=meta-llama/Llama-3.2-1B-Instruct \
--output_dir=output/llama_32_1B_lamb_0.1_fullR_tulu3_fix \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=32 \
--diff_transformer=true \
--groupnorm=false \
--diff_attn_lambda=0.1 \
--gradient_checkpointing=true \
--use_lora=false \
--negative_term_lora_only=false \
--negative_term_full_dim=true # 3870

# LAMBDA = 0.5, r=64, TULU 3
sbatch --gres=gpu:1 -J llama_32_1B_lamb_0.5_r64_tulu3_fix ./launch_tulu_sft.sh config_lora_tulu3 \
--tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
--diff_transformer_base_model_name=meta-llama/Llama-3.2-1B-Instruct \
--output_dir=output/llama_32_1B_lamb_0.5_r64_tulu3_fix \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=32 \
--diff_transformer=true \
--lora_alpha=128 \
--lora_rank=64 \
--groupnorm=false \
--diff_attn_lambda=0.5 \
--gradient_checkpointing=true \
--use_lora=false # 3256

# LAMBDA = 0.9, r=64, TULU 3
sbatch --gres=gpu:1 -J llama_32_1B_lamb_0.9_r64_tulu3_fix ./launch_tulu_sft.sh config_lora_tulu3 \
--tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
--diff_transformer_base_model_name=meta-llama/Llama-3.2-1B-Instruct \
--output_dir=output/llama_32_1B_lamb_0.9_r64_tulu3_fix \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=32 \
--diff_transformer=true \
--lora_alpha=128 \
--lora_rank=64 \
--groupnorm=false \
--diff_attn_lambda=0.9 \
--gradient_checkpointing=true \
--use_lora=false # 3257

# LAMBDA = 0.1, r=64, TULU 3, V and O trained as well, HALF RANK
sbatch --gres=gpu:1 -J llama_32_1B_lamb_0.1_r64_qkov_tulu3_fix ./launch_tulu_sft.sh config_lora_tulu3 \
--tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
--diff_transformer_base_model_name=meta-llama/Llama-3.2-1B-Instruct \
--output_dir=output/llama_32_1B_lamb_0.1_r64_qkov_tulu3_fix \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=32 \
--diff_transformer=true \
--lora_alpha=64 \
--lora_rank=32 \
--lora_o=true \
--lora_v=true \
--groupnorm=false \
--diff_attn_lambda=0.1 \
--gradient_checkpointing=true \
--use_lora=false # 2906

# LAMBDA = 0.5, r=64
sbatch --gres=gpu:1 -J llama_32_1B_lamb_0.5_r64_fix ./launch_tulu_sft.sh config_lora \
--tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
--diff_transformer_base_model_name=meta-llama/Llama-3.2-1B-Instruct \
--output_dir=output/llama_32_1B_lamb_0.5_r64_fix \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=32 \
--diff_transformer=true \
--lora_alpha=128 \
--lora_rank=64 \
--groupnorm=false \
--diff_attn_lambda=0.5 \
--gradient_checkpointing=true \
--use_lora=false # 2084, 2738 DONE


# LAMBDA = 0.5, r=64, V and O trained as well, HALF RANK
sbatch --gres=gpu:1 -J llama_32_1B_lamb_0.5_r64_qkov_fix ./launch_tulu_sft.sh config_lora \
--tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
--diff_transformer_base_model_name=meta-llama/Llama-3.2-1B-Instruct \
--output_dir=output/llama_32_1B_lamb_0.5_r64_qkov_fix \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=32 \
--diff_transformer=true \
--lora_alpha=64 \
--lora_rank=32 \
--lora_v=true \
--lora_o=true \
--groupnorm=false \
--diff_attn_lambda=0.5 \
--gradient_checkpointing=true \
--use_lora=false # 2907 DONE

# LAMBDA = 0.5, r=512
sbatch --gres=gpu:1 -J llama_32_1B_lamb_0.5_r512_fix ./launch_tulu_sft.sh config_lora \
--tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
--diff_transformer_base_model_name=meta-llama/Llama-3.2-1B-Instruct \
--output_dir=output/llama_32_1B_lamb_0.5_r512_fix \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=32 \
--diff_transformer=true \
--lora_alpha=1024 \
--lora_rank=512 \
--groupnorm=false \
--diff_attn_lambda=0.5 \
--gradient_checkpointing=true \
--use_lora=false # 2085, 2739 DONE

# LAMBDA = 0.9, r=512, groupnorm
sbatch --gres=gpu:1 -J llama_32_1B_lamb_0.9_r512_fix ./launch_tulu_sft.sh config_lora \
--tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
--diff_transformer_base_model_name=meta-llama/Llama-3.2-1B-Instruct \
--output_dir=output/llama_32_1B_lamb_0.9_r512_fix \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=32 \
--diff_transformer=true \
--lora_alpha=1024 \
--lora_rank=512 \
--groupnorm=true \
--diff_attn_lambda=0.9 \
--gradient_checkpointing=true \
--use_lora=false # 2086, 2740 DONE

# LAMBDA = 0.9, groupnorm, r=64, V and O trained as well, HALF RANK
sbatch --gres=gpu:1 -J llama_32_1B_lamb_0.9_r64_qkov_fix ./launch_tulu_sft.sh config_lora \
--tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
--diff_transformer_base_model_name=meta-llama/Llama-3.2-1B-Instruct \
--output_dir=output/llama_32_1B_lamb_0.9_r64_qkov_fix \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=32 \
--diff_transformer=true \
--lora_alpha=64 \
--lora_rank=32 \
--lora_v=true \
--lora_o=true \
--groupnorm=true \
--diff_attn_lambda=0.9 \
--gradient_checkpointing=true \
--use_lora=false # 2908 DONE

#######################################################
############ LR/BS OPTIM 
######################################################
# LAMBDA = 0.1, r=64, TULU 3
sbatch --gres=gpu:1 -J llama_32_1B_lamb_0.1_b32_lr_1e-4 ./launch_tulu_sft.sh config_lora_tulu3 \
--tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
--diff_transformer_base_model_name=meta-llama/Llama-3.2-1B-Instruct \
--output_dir=output/llama_32_1B_lamb_0.1_b32_lr_1e-4 \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=32 \
--diff_transformer=true \
--lora_alpha=64 \
--lora_rank=32 \
--groupnorm=false \
--diff_attn_lambda=0.1 \
--gradient_checkpointing=true \
--use_lora=false \
--lora_negative_term_only=false \
--learning_rate=1e-4 # 10598 DONE

sbatch --gres=gpu:1 -J llama_32_1B_lamb_0.1_b32_lr_5e-4 ./launch_tulu_sft.sh config_lora_tulu3 \
--tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
--diff_transformer_base_model_name=meta-llama/Llama-3.2-1B-Instruct \
--output_dir=output/llama_32_1B_lamb_0.1_b32_lr_5e-4 \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=32 \
--diff_transformer=true \
--lora_alpha=64 \
--lora_rank=32 \
--groupnorm=false \
--diff_attn_lambda=0.1 \
--gradient_checkpointing=true \
--use_lora=false \
--lora_negative_term_only=false \
--learning_rate=5e-4 # 10599 DONE

sbatch --gres=gpu:1 -J llama_32_1B_lamb_0.1_b32_lr_1e-4_bs_128 ./launch_tulu_sft.sh config_lora_tulu3 \
--tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
--diff_transformer_base_model_name=meta-llama/Llama-3.2-1B-Instruct \
--output_dir=output/llama_32_1B_lamb_0.1_b32_lr_1e-4_bs_128 \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=64 \
--diff_transformer=true \
--lora_alpha=64 \
--lora_rank=32 \
--groupnorm=false \
--diff_attn_lambda=0.1 \
--gradient_checkpointing=true \
--use_lora=false \
--lora_negative_term_only=false \
--learning_rate=1e-4 # 10600 DONE

sbatch --gres=gpu:1 -J llama_32_1B_lamb_0.1_b32_lr_5e-4_bs_128 ./launch_tulu_sft.sh config_lora_tulu3 \
--tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
--diff_transformer_base_model_name=meta-llama/Llama-3.2-1B-Instruct \
--output_dir=output/llama_32_1B_lamb_0.1_b32_lr_5e-4_bs_128 \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=64 \
--diff_transformer=true \
--lora_alpha=64 \
--lora_rank=32 \
--groupnorm=false \
--diff_attn_lambda=0.1 \
--gradient_checkpointing=true \
--use_lora=false \
--lora_negative_term_only=false \
--learning_rate=5e-4 # 10747


sbatch --gres=gpu:1 -J llama_32_1B_lamb_0.1_b32_lr_1e-3_bs_128 ./launch_tulu_sft.sh config_lora_tulu3 \
--tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
--diff_transformer_base_model_name=meta-llama/Llama-3.2-1B-Instruct \
--output_dir=output/llama_32_1B_lamb_0.1_b32_lr_1e-3_bs_128 \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=64 \
--diff_transformer=true \
--lora_alpha=64 \
--lora_rank=32 \
--groupnorm=false \
--diff_attn_lambda=0.1 \
--gradient_checkpointing=true \
--use_lora=false \
--lora_negative_term_only=false \
--learning_rate=1e-3 # 11929


sbatch --gres=gpu:1 -J llama_32_1B_lamb_0.1_b32_lr_5e-5_bs_128 ./launch_tulu_sft.sh config_lora_tulu3 \
--tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
--diff_transformer_base_model_name=meta-llama/Llama-3.2-1B-Instruct \
--output_dir=output/llama_32_1B_lamb_0.1_b32_lr_5e-4_bs_128 \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=64 \
--diff_transformer=true \
--lora_alpha=64 \
--lora_rank=32 \
--groupnorm=false \
--diff_attn_lambda=0.1 \
--gradient_checkpointing=true \
--use_lora=false \
--lora_negative_term_only=false # 10748 DONE

# LATER !
sbatch --gres=gpu:1 -J llama_32_1B_lamb_0.1_b32_bs_128 ./launch_tulu_sft.sh config_lora_tulu3 \
--tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
--diff_transformer_base_model_name=meta-llama/Llama-3.2-1B-Instruct \
--output_dir=output/llama_32_1B_lamb_0.1_b32_bs_128 \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=64 \
--diff_transformer=true \
--lora_alpha=64 \
--lora_rank=32 \
--groupnorm=false \
--diff_attn_lambda=0.1 \
--gradient_checkpointing=true \
--use_lora=false \
--lora_negative_term_only=false # 11435 DONE

sbatch --gres=gpu:1 -J llama_32_1B_lamb_0.1_b32_bs_256 ./launch_tulu_sft.sh config_lora_tulu3 \
--tokenizer_name=meta-llama/Llama-3.2-1B-Instruct \
--diff_transformer_base_model_name=meta-llama/Llama-3.2-1B-Instruct \
--output_dir=output/llama_32_1B_lamb_0.1_b32_bs_128 \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=64 \
--diff_transformer=true \
--lora_alpha=64 \
--lora_rank=32 \
--groupnorm=false \
--diff_attn_lambda=0.1 \
--gradient_checkpointing=true \
--use_lora=false \
--lora_negative_term_only=false # 11436 DONE


##################################################
## MERGIN FOR CLASSI MODELS LORA########
######################################
python merge_lora.py --lora_model_name_or_path /scratch/1/user/mlouis/calmar/diff_transformer/code/open-instruct/output/llama_32_1B_lora_r64 --base_model_name_or_path meta-llama/Llama-3.2-1B-Instruct --tokenizer_name_or_path /scratch/1/user/mlouis/calmar/diff_transformer/code/open-instruct/output/llama_32_1B_lora_r64 --output_dir /scratch/1/user/mlouis/calmar/diff_transformer/code/open-instruct/output/llama_32_1B_lora_r64_MERGED
python merge_lora.py --lora_model_name_or_path /scratch/1/user/mlouis/calmar/diff_transformer/code/open-instruct/output/llama_32_1B_lora_r512 --base_model_name_or_path meta-llama/Llama-3.2-1B-Instruct --tokenizer_name_or_path /scratch/1/user/mlouis/calmar/diff_transformer/code/open-instruct/output/llama_32_1B_lora_r512 --output_dir /scratch/1/user/mlouis/calmar/diff_transformer/code/open-instruct/output/llama_32_1B_lora_r512_MERGED

######################
### EVAL COMMANDS#####
#####################
sbatch --gres=gpu:1 -J ev_llama_32_1B_lamb_0.1_r64 ./eval.sh llama_32_1B_lamb_0.1_r64 # 2814
sbatch --gres=gpu:1 -J ev_llama_32_1B_lamb_0.1_r512 ./eval.sh llama_32_1B_lamb_0.1_r512 # 2207
sbatch --gres=gpu:1 -J ev_llama_32_1B_lamb_0.5_r512 ./eval.sh llama_32_1B_lamb_0.5_r512 # 2713
sbatch --gres=gpu:1 -J ev_llama_32_1B_lamb_0.5_r64 ./eval.sh llama_32_1B_lamb_0.5_r64 # 2714
sbatch --gres=gpu:1 -J ev_llama_32_1B_lamb_0.9_r512 ./eval.sh llama_32_1B_lamb_0.9_r512 # 2715
sbatch --gres=gpu:1 -J ev_llama_32_1B_lora_r512_MERGED ./eval.sh llama_32_1B_lora_r512_MERGED # 2716
sbatch --gres=gpu:1 -J ev_llama_32_1B_lora_r64_MERGED ./eval.sh llama_32_1B_lora_r64_MERGED # 2717
sbatch --gres=gpu:1 -J ev_meta-llama/Llama-3.2-1B-Instruct ./eval.sh meta-llama/Llama-3.2-1B-Instruct # 2718
    
sbatch --gres=gpu:1 -J ev_llama_32_1B_lamb_0.5_r64_fix ./eval.sh llama_32_1B_lamb_0.5_r64_fix # 2911
sbatch --gres=gpu:1 -J ev_llama_32_1B_lamb_0.1_r64_fix ./eval.sh llama_32_1B_lamb_0.1_r64_fix # 2912
sbatch --gres=gpu:1 -J ev_llama_32_1B_lamb_0.1_r512_fix ./eval.sh llama_32_1B_lamb_0.1_r512_fix # 2913
sbatch --gres=gpu:1 -J ev_llama_32_1B_lamb_0.1_b256_fix ./eval.sh llama_32_1B_lamb_0.1_b256_fix # 2914


# Need merge lora before ...
sbatch --gres=gpu:1 -J ev_llama_32_1B_lora_r64_qk ./eval.sh --model_name llama_32_1B_lora_r64_qk --merge # 2971
sbatch --gres=gpu:1 -J ev_llama_32_1B_lora_r64_qkov ./eval.sh --model_name  llama_32_1B_lora_r64_qkov --merge # 2974
sbatch --gres=gpu:1 -J ev_llama_32_1B_lora_r512_qk ./eval.sh --model_name  llama_32_1B_lora_r512_qk --merge # 2975


sbatch --gres=gpu:1 -J ev_llama_32_1B_lamb_0.1_r64_qkov_fix ./eval.sh --model_name llama_32_1B_lamb_0.1_r64_qkov_fix # 3003
sbatch --gres=gpu:1 -J ev_llama_32_1B_lamb_0.5_r64_qkov_fix ./eval.sh --model_name llama_32_1B_lamb_0.5_r64_qkov_fix #  3004
sbatch --gres=gpu:1 -J ev_llama_32_1B_lamb_0.9_r512_fix ./eval.sh --model_name llama_32_1B_lamb_0.9_r512_fix #  3005
sbatch --gres=gpu:1 -J ev_llama_32_1B_lamb_0.9_r64_qkov_fix ./eval.sh --model_name llama_32_1B_lamb_0.9_r64_qkov_fix #  3006


sbatch --gres=gpu:1 -J ev_llama_32_1B_lora_r64_qk_tulu3 ./eval.sh --model_name llama_32_1B_lora_r64_qk_tulu3 --merge  # 3066
sbatch --gres=gpu:1 -J ev_llama_32_1B_lora_r64_qkov_tulu3 ./eval.sh --model_name llama_32_1B_lora_r64_qkov_tulu3 --merge  # 3067
sbatch --gres=gpu:1 -J ev_llama_32_1B_lamb_0.1_r64_tulu3_fix ./eval.sh --model_name llama_32_1B_lamb_0.1_r64_tulu3_fix # 3068
sbatch --gres=gpu:1 -J ev_llama_32_1B_lamb_0.5_r512_fix ./eval.sh --model_name llama_32_1B_lamb_0.5_r512_fix # 3070

sbatch --gres=gpu:1 -J ev_llama_32_1B_lamb_0.1_r64_qkov_tulu3_fix ./eval.sh --model_name llama_32_1B_lamb_0.1_r64_qkov_tulu3_fix # 3210



#To check:

 3257
sbatch --gres=gpu:1 -J ev_llama_32_1B_lamb_0.5_r64_tulu3_fix ./eval.sh --model_name llama_32_1B_lamb_0.5_r64_tulu3_fix # 3531
sbatch --gres=gpu:1 -J ev_llama_32_1B_lamb_0.9_r64_tulu3_fix ./eval.sh --model_name llama_32_1B_lamb_0.9_r64_tulu3_fix  # 3532


# Checking load embedding properly:
sbatch --gres=gpu:1 -J ev_llama_32_1B_lamb_0.1_r64_tulu3_fix_pl ./eval.sh --model_name llama_32_1B_lamb_0.1_r64_tulu3_fix --save_suffix _proper_loading # 3068




# RUNS TODO/UNDONE  
# NB: qkov runs NOT DONE for diff r512, qkov runs NOT DONE for tulu3 (anywhere)
# No runs launched with mlp so far (wait on Alex feedback on that)
sbatch --gres=gpu:1 -J ev_llama_32_1B_lamb_0.1_fullR_tulu3_fix ./eval.sh --model_name llama_32_1B_lamb_0.1_fullR_tulu3_fix # DONE
sbatch --gres=gpu:1 -J ev_llama_32_1B_full_tulu3 ./eval.sh --model_name llama_32_1B_full_tulu3 # DONE

sbatch --gres=gpu:1 -J ev_llama_32_1B_lamb_0.1_b32_tulu3_fix ./eval.sh --model_name llama_32_1B_lamb_0.1_b32_tulu3_fix # 11927

# RAG eval
sbatch --gres=gpu:1 -J rag_ev_llama_32_1B_lamb_0.1_b32_tulu3_fix ./rag_eval_diff.sh "/scratch/1/user/mlouis/calmar/diff_transformer/code/open-instruct/output/llama_32_1B_lamb_0.1_b32_tulu3_fix" # 10750


sbatch --gres=gpu:1 -J ev_llama_32_1B_lamb_0.1_b32_lr_1e-4 ./eval.sh --model_name llama_32_1B_lamb_0.1_b32_lr_1e-4 # 11437, 11636
sbatch --gres=gpu:1 -J ev_llama_32_1B_lamb_0.1_b32_lr_5e-4 ./eval.sh --model_name llama_32_1B_lamb_0.1_b32_lr_5e-4 # 11438
sbatch --gres=gpu:1 -J ev_llama_32_1B_lamb_0.1_b32_lr_1e-4_bs_128 ./eval.sh --model_name llama_32_1B_lamb_0.1_b32_lr_1e-4_bs_128 # 11439
sbatch --gres=gpu:1 -J ev_llama_32_1B_lamb_0.1_b32_lr_5e-4_bs_128 ./eval.sh --model_name llama_32_1B_lamb_0.1_b32_lr_5e-4_bs_128 # 11450
sbatch --gres=gpu:1 -J ev_llama_32_1B_lamb_0.1_b32_lr_5e-5_bs_128   ./eval.sh --model_name llama_32_1B_lamb_0.1_b32_lr_5e-5_bs_128 # 12689

sbatch --gres=gpu:1 -J ev_llama_32_1B_lamb_0.1_b32_bs_256   ./eval.sh --model_name llama_32_1B_lamb_0.1_b32_bs_256 # 12690
sbatch --gres=gpu:1 -J ev_llama_32_1B_lamb_0.1_b32_bs_128   ./eval.sh --model_name llama_32_1B_lamb_0.1_b32_bs_128 # 12691
sbatch --gres=gpu:1 -J ev_llama_32_1B_lamb_0.1_b32_tulu3_eager  ./eval.sh --model_name llama_32_1B_lamb_0.1_b32_tulu3_eager # 12686


