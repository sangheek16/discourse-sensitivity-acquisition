MODEL=$1
SAVENAME=$2


declare -a ps=(0 0.9 0.95)
declare -a ks=(50 0)
declare -a temps=(0.7 1.0)

for k in "${ks[@]}"; do
    for temp in "${temps[@]}"; do

        # for p = None
        echo "p: None, t: $temp, k: $k"
        python src/collect-generations.py \
            --device cuda:0 \
            -p -1 \
            -t $temp \
            -k $k \
            --response "No, that's not true!" \
            --model $MODEL \
            --outdir data/results/generations/$SAVENAME \
            --outfile gens_None_${k}_${temp}_rejection.json

        python src/collect-generations.py \
            --device cuda:0 \
            -p -1 \
            -t $temp \
            -k $k \
            --model $MODEL \
            --outdir data/results/generations/$SAVENAME \
            --outfile gens_None_${k}_${temp}_freeform.json

        for p in "${ps[@]}"; do
            python src/collect-generations.py \
                --device cuda:0 \
                -p $p \
                -t $temp \
                -k $k \
                --response "No, that's not true!" \
                --model $MODEL \
                --outdir data/results/generations/$SAVENAME \
                --outfile gens_${p}_${k}_${temp}_rejection.json

            python src/collect-generations.py \
                --device cuda:0 \
                -p $p \
                -t $temp \
                -k $k \
                --model $MODEL \
                --outdir data/results/generations/$SAVENAME \
                --outfile gens_${p}_${k}_${temp}_freeform.json
        done
    done
done
