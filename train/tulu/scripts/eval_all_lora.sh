
# to evaluate (1b):
# - fulllora-r8, qklora-r64, difflora-bothsides-r32 (lambda=0.1, learnable lambda, lambda=0.5, lambda=0.9)
models_dir=/scratch/1/user/vnikouli/projects/bergen/difflora/tulu/open-instruct/output/
m_models_dir=/scratch/1/user/mlouis/calmar/diff_transformer/code/open-instruct/output/
models=(
    #"llama_32_1B_lora_r64_qk" #- ongoing
   # "llama_32_1B_lora_r64_qk_lr5e5" #- ongoing
   # "llama_32_1B_lora_r8_lr5e5"
    "llama_32_1B_lora_r8"
    #"llama_32_1B_lora_r64_qk_tulu3"
)

models_8b=(
    "llama_3_8b_loraqk_r128"
    "llama_3_8B_lora_r16"
)
models_diff_8b=(
    #"llama_3_8b_difflora_r64_definit"
    "llama_3_8B_difflora_r64_groupnorm"
)
diff_models=(
    "llama_32_1B_learnlambda_r64_tulu3_nogroupnorm_right_side_sum"
    # llama_32_1B_learnlambda_definit_r64_right_only_withlinit
    # "llama_32_1B_learnlambda_withdefinit_withlinit_r32"
    #"llama_32_1B_learnlambda_withdefinit_groupnorm_withlinit_r32"
    #"llama_32_1B_learnlambda_withinit01_r32_eager"
    #"llama_32_1B_learnlambda_init0_r32_tulu3_nogroupnorm"
    #"llama_32_1B_learnlambda_init0_r32_tulu3_groupnorm"
    #"llama_32_1B_learnlambda_definit_r32_tulu3_groupnorm"
    #"llama_32_1B_learnlambda_real_definit_r64_right_only_groupnorm"
    #"llama_32_1B_learnlambda_real_definit_r32_groupnorm"
    #"llama_32_1B_learnlambda_definit_r32_groupnorm"
    #"llama_32_1B_learnlambda_definit_r32_sum"
    # "llama_32_1B_lamb_0.1_b32_lr_1e-4" #bothsides, lr-1e-4
    #"llama_32_1B_learnlambda_definit_r32_groupnorm_sum"
    #"llama_32_1B_learnlambda_definit_r64_right_only_groupnorm_sum"
    #"llama_32_1B_learnlambda_definit_r32_withfixedtok"
    #"llama_32_1B_learnlambda_definit_r64_right_only_groupnorm"
    # "llama_32_1B_lamb_0.1_r64" #rightside only, fixed lambda
    #"llama_32_1B_learnLambda_rightonly_r64" #rightside only, learnable lambda
    #  "llama_32_1B_learnlambda_withinit01_r32"
     #"llama_32_1B_learnlambda_definit_r32"
    # "llama_32_1B_learnlambda_definit_r64_right_only"
    # #"llama_32_1B_lamb_0.1_b32_lr_5e-4_bs_128"
    # "llama_32_1B_lamb_0.1_b32_tulu3_fix"
    # "llama_32_1B_lamb_0.1_r64_tulu3_fix"
    #"llama_32_1B_lamb_0.1_b32_tulu3_eager" 
    # "llama_32_1B_lamb_0.1_fullR_tulu3_fix"
    # "llama_32_1B_lamb_0.5_r64_tulu3_fix"
    # "llama_32_1B_lamb_0.1_r64_fix"
    #"llama_32_1B_lamb_0.5_r64_fix"
)

for model in ${models_diff_8b[@]}; do
    echo "Evaluating $model"
    #sbatch eval_1b.sh --model_name $model --save_suffix "fixedtok"
    #sbatch --gres=gpu:1 -J rag_ev ./rag_eval_diff.sh "${models_dir}/${model}" #35540
done
for model in ${models_8b[@]}; do
    
    echo "Evaluating $model"
    #sbatch eval_1b.sh --model_name $model --merge --basename "meta-llama/Meta-Llama-3-8B-Instruct"
    #sbatch --gres=gpu:1 -J rag_ev ./rag_eval_lora.sh "${models_dir}/${model}_MERGED" llama-3-8b-instruct
done
#sh eval_1b.sh --model_name "meta-llama/Llama-3.2-1B-Instruct"

for model in ${diff_models[@]}; do
    echo "Evaluating $models_dir/$model"
    if test -e "$models_dir/$model"; then
        echo "$models_dir/$model exists"
    else
       echo "$models_dir/$model does not exist"
       ln -s $m_models_dir/$model $models_dir/$model
    fi
    echo "Running evaluation for $model"
    sh eval_1b.sh --model_name $model --save_suffix "fixcode" #36935
    sbatch --gres=gpu:1 -J rag_ev ./rag_eval_diff.sh "${models_dir}/${model}" 
done
exit
for model in ${models[@]}; do
    echo "Evaluating $model"

    sh eval_1b.sh --model_name $model --merge 
    #sbatch --gres=gpu:1 -J rag_ev ./rag_eval_lora.sh "${models_dir}/${model}_MERGED" llama-32-1b-instruct
done

exit
sbatch --gres=gpu:1 -J rag_ev ./rag_eval_baseline.sh llama-32-1b-instruct #34864

exit
sbatch eval_1b.sh --model_name "meta-llama/Meta-Llama-3-8B-Instruct" 
sbatch eval_1b.sh --model_name "allenai/Llama-3.1-Tulu-3-8B-SFT" #35945
sbatch eval_1b.sh --model_name "allenai/llama-3-tulu-2-8b" #35996
exit
diff_models=(
    "llama_32_1B_lamb_0.1_b32_lr_1e-4" #bothsides, lr-1e-4
    "llama_32_1B_lamb_0.1_r64" #rightside only, fixed lambda
    "llama_32_1B_learnLambda_rightonly_r64" #rightside only, learnable lambda
    "llama_32_1B_learnlambda_withinit01_r32"
    "llama_32_1B_learnlambda_definit_r32"
    "llama_32_1B_learnlambda_definit_r64_right_only"
    "llama_32_1B_learnlambda_init01_r64_right_only"
)


# eval baseline model 
llama-32-1b-instruct

sbatch --gres=gpu:1 -J rag_ev ./rag_eval_baseline.sh llama-3-8b-instruct #34864

sbatch --gres=gpu:1 -J rag_ev ./rag_eval_baseline.sh llama-32-1b-instruct #34864
for model in ${diff_models[@]}; do
    echo "Evaluating $models_dir/$model"
    sh eval_1b.sh --model_name $model 
    #sh rag_eval_diff.sh "${models_dir}/${model}" 
    #sbatch --gres=gpu:1 -J rag_ev ./rag_eval_diff.sh "${models_dir}/${model}" 
done
exit
for model in ${models[@]}; do
    echo "Evaluating $model"

    sbatch eval_1b.sh --model_name $model --merge 
    sbatch --gres=gpu:1 -J rag_ev ./rag_eval_diff.sh "${models_dir}/{$model}" 

done

#RAG eval 
#sbatch --gres=gpu:1 -J rag_ev ./rag_eval_diff.sh "${models_dir}/{$model}" # 10750
