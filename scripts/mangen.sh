# generations

# declare -a models=(meta-llama/Meta-Llama-3-8B-Instruct Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct)

# declare -a modes=(arc coord)
# # declare -a modes=(coord)

# for model in "${models[@]}"; do
#     for mode in "${modes[@]}"; do
#         python src/mangen-eval.py --instruct --model $model --mode $mode --results-dir data/results/mangen/${mode}
#     done
# done


declare -a models=(meta-llama/Meta-Llama-3-8B Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B)

declare -a modes=(arc coord)
# declare -a modes=(coord)

for model in "${models[@]}"; do
    for mode in "${modes[@]}"; do
        python src/mangen-eval.py --model $model --mode $mode --results-dir data/results/mangen/${mode}
    done
done