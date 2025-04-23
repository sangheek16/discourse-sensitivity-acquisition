# large instruction-tuned models, small batch size
declare -a models=(
    # -- COMPLETED
    meta-llama/Meta-Llama-3-8B-Instruct
        # NOTE: Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
        # NOTE: Below are the messages that showed up:
        # * -- Saving meta-llama/Meta-Llama-3-8B-Instruct data to: data/results/generations -- *
        # scripts/run-generations.sh: line 57: h-size: command not found
        # scripts/run-generations.sh: line 59: syntax error near unexpected token `done'
        # scripts/run-generations.sh: line 59: `done'

    # -- CANDIDATES
    # mistralai/Mistral-7B-Instruct-v0.3 
    #     # NOTE: Not authorized
    # google/gemma-7b-it
    #     # NOTE: Out of memory
)

for model in "${models[@]}"; do
    echo "Running $model"
    python src/generations.py \
        --eval-path "data/used_items.csv" \
        --results-dir "data/results/generations" \
        --model $model \
        --batch-size 16 \
        --template "\$name1 said, \"\$subj \$vp\", and \$name2 replied, "
done

## small instruction-tuned models, high batch size
declare -a models=(
    # -- COMPLETED
    HuggingFaceTB/SmolLM2-135M-Instruct

    # -- CANDIDATES
)

for model in "${models[@]}"; do
    echo "Running $model"
    python src/generations.py \
        --eval-path "data/used_items.csv" \
        --results-dir "data/results/generations" \
        --model $model \
        --batch-size 64 \
        --template "\$name1 said, \"\$subj \$vp\", and \$name2 replied, "
done

## medium instruction-tuned models, medium batch size
# declare -a models=(
#     # -- CANDIDATES
#         # facebook/opt-iml-1.3b 
#             ## NOTE: CUDA out of memory
#             ## This process has 34.12 GiB memory in use;
#             ## GPU 0 has a total capacity of 47.50 GiB of which 13.37 GiB is free
#         # tiiuae/falcon-rw-1b-instruct
#         # mosaicml/mpt-1b-redpajama-200b-instruct
#         # togethercomputer/RedPajama-INCITE-Instruct-7B-v0.1nvidia-smi
#         # google/gemma-2b-it
#         # allenai/OLMo-7B-Instruct
# )

# for model in "${models[@]}"; do
#     echo "Running $model"
#     python src/generations.py \
#         --eval-path "data/used_items.csv" \
#         --results-dir "data/results/generations" \
#         --model $model \
#         --batch-size 32 \
#         --template "\$name1 said, \"\$subj \$vp\", and \$name2 replied, "
# done