# small models, high batch size
declare -a models=(gpt2 gpt2-medium gpt2-large EleutherAI/pythia-70m-deduped EleutherAI/pythia-160m-deduped
    EleutherAI/pythia-410m-deduped facebook/opt-125m facebook/opt-350m)

for model in "${models[@]}"; do
    echo "Running $model"
    python src/basic-eval.py --model $model --batch_size 128 --device cuda:0
done

# medium models, medium batch size
declare -a models=(EleutherAI/pythia-1.4b-deduped EleutherAI/pythia-2.8b-deduped
    EleutherAI/pythia-6.9b-deduped google/Gemma-2-2B allenai/OLMo-2-1124-7B facebook/opt-1.3b facebook/opt-2.7b)

for model in "${models[@]}"; do
    echo "Running $model"
    python src/basic-eval.py --model $model --batch_size 64 --device cuda:0
done

declare -a models=(EleutherAI/pythia-12b-deduped google/Gemma-2-9B facebook/opt-6.7b)

for model in "${models[@]}"; do
    echo "Running $model"
    python src/basic-eval.py --model $model --batch_size 16 --device cuda:0
done
