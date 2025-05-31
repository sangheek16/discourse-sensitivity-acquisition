# generations

declare -a models=(meta-llama/Meta-Llama-3-8B-Instruct Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct)

declare -a modes=(arc coord)
# declare -a modes=(coord)

for model in "${models[@]}"; do
    for mode in "${modes[@]}"; do
        python src/dgrc-eval.py --instruct --model $model --mode $mode
    done
done