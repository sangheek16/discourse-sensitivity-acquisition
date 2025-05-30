MODEL=$1
SAVENAME=$2


declare -a ps=(0 0.9 0.95)
declare -a ks=(50 0)
declare -a temps=(0.7 1.0)

for k in "${ks[@]}"; do
    for temp in "${temps[@]}"; do

        # for p = None
        # echo "p: None, t: $temp, k: $k"
        # python src/collect-generations.py \
        #     --device cuda:0 \
        #     -p -1 \
        #     -t $temp \
        #     -k $k \
        #     --instruct \
        #     --response "No, that's not true!" \
        #     --model $MODEL \
        #     --outdir data/results/generations/$SAVENAME \
        #     --outfile gens_None_${k}_${temp}_rejection.json

        # python src/collect-generations.py \
        #     --device cuda:0 \
        #     -p -1 \
        #     -t $temp \
        #     -k $k \
        #     --instruct \
        #     --model $MODEL \
        #     --outdir data/results/generations/$SAVENAME \
        #     --outfile gens_None_${k}_${temp}_freeform.json

        for p in "${ps[@]}"; do
            python src/collect-generations.py \
                --device cuda:0 \
                -p $p \
                -t $temp \
                -k $k \
                --instruct \
                --response "No, that's not true!" \
                --model $MODEL \
                --outdir data/results/generations/$SAVENAME \
                --outfile gens_${p}_${k}_${temp}_rejection.json

            python src/collect-generations.py \
                --device cuda:0 \
                -p $p \
                -t $temp \
                -k $k \
                --instruct \
                --model $MODEL \
                --outdir data/results/generations/$SAVENAME \
                --outfile gens_${p}_${k}_${temp}_freeform.json
        done
    done
done



# declare -a models=(meta-llama/Meta-Llama-3-8B)

# for model in "${models[@]}"; do
#     for k in "${ks[@]}"; do
#         for temp in "${temps[@]}"; do

#             # for p = None
#             echo "p: None, t: $temp, k: $k"
#             python src/collect-generations.py \
#                     --device cuda:0 \
#                     -p -1 \
#                     -t $temp \
#                     -k $k \
#                     --response "No, that's not true!" \
#                     --model $model \
#                     --outdir data/results/generations/llama-3-8b \
#                     --outfile gens_None_${k}_${temp}_rejection.json
#                     # --model meta-llama/Meta-Llama-3-8B-Instruct \
#                     # --outdir data/results/generations/llama-3-8b-instruct \

#             python src/collect-generations.py \
#                 --device cuda:0 \
#                 -p -1 \
#                 -t $temp \
#                 -k $k \
#                 --model meta-llama/Meta-Llama-3-8B \
#                 --outdir data/results/generations/llama-3-8b \
#                 --outfile gens_None_${k}_${temp}_freeform.json
#                 # --model HuggingFaceTB/SmolLM2-360M-Instruct \
#                 # --outdir data/results/generations/smollm2-360m-instruct \

#             for p in "${ps[@]}"; do
#                 echo "p: $p, t: $temp, k: $k"
#                 python src/collect-generations.py \
#                     --device cuda:0 \
#                     -p $p \
#                     -t $temp \
#                     -k $k \
#                     --response "No, that's not true!" \
#                     --model meta-llama/Meta-Llama-3-8B\
#                     --outdir data/results/generations/llama-3-8b\
#                     --outfile gens_${p}_${k}_${temp}_rejection.json

#                 python src/collect-generations.py \
#                     --device cuda:0 \
#                     -p $p \
#                     -t $temp \
#                     -k $k \
#                     --model meta-llama/Meta-Llama-3-8B \
#                     --outdir data/results/generations/llama-3-8b\
#                     --outfile gens_${p}_${k}_${temp}_freeform.json
#             done
#         done
#     done
# done