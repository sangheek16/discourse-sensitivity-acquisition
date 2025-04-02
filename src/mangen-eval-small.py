'''
-----------------------------------------------------------
NOTE on workflow:

    1. get logprobs for each sequence
    2. compute average logprobs by item & continuation
    3. compare [move on to R script]

-----------------------------------------------------------
'''

import argparse
import pathlib
import utils
import pandas as pd
from tqdm import tqdm
from minicons import scorer

# initialize model
model = "distilgpt2" # TODO: change this

model_name = model.replace("", "") # TODO: change this
lm = scorer.IncrementalLMScorer(model, device='cpu') # TODO: change this

# load dataset
eval_path = '../data/stimuli/manual-full.csv' # TODO: change this
results_dir = '../data/results/mangen' # TODO: change this
df = pd.read_csv(eval_path)

# get logprobs
tqdm.pandas()
df['logprob'] = df['stimuli'].progress_apply(lambda x: lm.sequence_score(x)[0])

# compute average logprobs
df_avg = df.groupby(['idx', 'cont_type', 'cont_target'], as_index=False)['logprob'].mean()
df_avg.rename(columns={'logprob': 'avg_logprob'}, inplace=True)

# save results
pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True) # creates if it does not exist
df_avg.to_csv(f"{results_dir}/{model_name}.csv", index=False) # save'