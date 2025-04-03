'''
-----------------------------------------------------------
NOTE:   this is a code that calculates average logprobs
        of the manually generated sentences.
        it averages the 15 continuations given the MC;
        and the 15 continuations given the ARC content.

NOTE:   workflow

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

def main(args):
    # get paths 
    eval_path = args.eval_path
    results_dir = args.results_dir

    # load the model
    model = args.model
    model_name = model.replace("/", "_")

    # load dataset
    df = pd.read_csv(eval_path)

    # load scorer
    lm = scorer.IncrementalLMScorer(args.model, device=args.device)
    
    # compute log probabilities
    tqdm.pandas()
    df['logprob'] = df['stimuli'].progress_apply(lambda x: lm.sequence_score(x)[0])
    
    # compute average log probabilities
    df_avg = df.groupby(['idx', 'cont_type', 'cont_target'], as_index=False)['logprob'].mean()
    df_avg.rename(columns={'logprob': 'avg_logprob'}, inplace=True)

    # add model name
    df_avg['model_name'] = model_name
    
    # save results
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)
    df_avg.to_csv(f"{results_dir}/{model_name}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--eval-path", type=str, required=True, help="csv file")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0", help="choose cuda:0 or cpu")
    
    args = parser.parse_args()
    main(args)