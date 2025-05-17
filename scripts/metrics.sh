
# instruct models

declare -a models=(meta-llama/Meta-Llama-3-8B-Instruct Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct)
# declare -a models=(Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct)

for model in "${models[@]}"; do
    python src/metrics.py \
        --model $model \
        --batch_size 16 \
        --instruct \
        --eval-path data/stimuli/kim22-arc.csv \
        --results-dir data/results/kim22-arc-metrics/

    python src/metrics.py \
        --model $model \
        --batch_size 16 \
        --instruct \
        --eval-path data/stimuli/kim22-coord.csv \
        --results-dir data/results/kim22-coord-metrics/
done



# non instruct models

declare -a models=(meta-llama/Meta-Llama-3-8B Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B)
# declare -a models=(Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B)

for model in "${models[@]}"; do
    python src/metrics.py \
        --model $model \
        --batch_size 16 \
        --eval-path data/stimuli/kim22-arc.csv \
        --results-dir data/results/kim22-arc-metrics/

    python src/metrics.py \
        --model $model \
        --batch_size 16 \
        --eval-path data/stimuli/kim22-coord.csv \
        --results-dir data/results/kim22-coord-metrics/
done
