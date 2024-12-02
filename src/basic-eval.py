"""
basic script to do log-probability based evaluation of language models.

So far I am only implementing this for decoder only LMs, and not models like BERT.
"""

import argparse
import pathlib
import utils

from minicons import scorer
from torch.utils.data import DataLoader
from tqdm import tqdm  # loading bar


def main(args):

    eval_path = args.eval_path  # csv?
    model = args.model  # usually a huggingface identifier
    results_dir = args.results_dir

    model_name = model.replace("/", "_")

    # load the model
    lm = scorer.IncrementalLMScorer(model, device=args.device)

    # read eval
    eval_data = utils.read_csv_dict(eval_path)

    eval_dl = DataLoader(eval_data, batch_size=args.batch_size)

    results = []
    for batch in tqdm(eval_dl):
        # idx, items, types, prefixes, stimuli = batch
        idx = batch['idx']
        items = batch['item']
        types = batch['type']
        sentences = batch['sentence']
        prefixes = batch['prefix']
        stimuli = batch['stimulus']

        # compute conditional log probs 
        if "gpt2" in model_name:
            # gpt2 doesn't automatically add bos tokens at the start of the sequence by default.
            log_probs = lm.conditional_score(prefixes, stimuli, bos_token=True, bow_correction=True)
            # log_probs = lm.sequence_score(sentences, bos_token=True, bow_correction=True)
        else:
            log_probs = lm.conditional_score(prefixes, stimuli, bow_correction=True)
            # log_probs = lm.sequence_score(sentences, bow_correction=True)

        for i, item, typ, logprob in zip(idx, items, types, log_probs):
            results.append({
                "idx": i,
                "item": item,
                "type": typ,
                "logprob": logprob
            })

    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True) # creates if it does not exist
    utils.write_dict_list_to_csv(results, f"{results_dir}/{model_name}.csv") # save

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--eval-path", type=str, default="data/stimuli/kim22-full.csv")
    parser.add_argument("--model", type=str, default="gpt2-medium")
    parser.add_argument("--results-dir", type=str, default="data/results/kim22")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    main(args)
