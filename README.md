# discourse-sensitivity-acquisition


## Data

```bash
python src/stimuli-constructor.py --save-dir data/stimuli
```

## Basic Kim et al., 2022 type eval


Basic usage:
```bash
usage: src/basic-eval.py [-h] [--eval-path EVAL_PATH] [--model MODEL] [--results-dir RESULTS_DIR]
                     [--batch-size BATCH_SIZE] [--device DEVICE]

options:
  -h, --help            show this help message and exit
  --eval-path EVAL_PATH
  --model MODEL
  --results-dir RESULTS_DIR
  --batch-size BATCH_SIZE
  --device DEVICE
```

For example:

```bash
python src/basic-eval.py --model allenai/OLMo-7B-0724-hf --device cuda:0 --batch-size 32 --eval_path data/stimuli/kim22-full.csv
```

or run the full script with lots of models (requires GPU):

```bash
bash scripts/run-kim22-evals.sh
```