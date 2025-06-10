import argparse
import pathlib
import config
import torch
import utils

from string import Template
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from minicons import scorer
from tqdm import tqdm

INSTRUCTION = "Please respond to the following message as naturally as possible, using a single sentence, as if we were talking to each other. Please keep it short."

NO_HEADER = "No, that's not true!"
HEYWAIT_HEADER = "Hey, wait a minute!"

TEMPLATE = Template('$name1 said, "$preamble", and $name2 replied, "')


def chat_template(sentence, tok, response_prompt=NO_HEADER):
    """
    A function that applies the model's chat template to simulate
    an interaction environment. Two possible options
    """
    if response_prompt is None:
        return tok.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": INSTRUCTION,
                },
                {"role": "user", "content": sentence},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        return tok.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": INSTRUCTION,
                },
                {"role": "user", "content": sentence},
                {"role": "assistant", "content": response_prompt},
            ],
            tokenize=False,
            continue_final_message=True,
        )


def dialog_template(name1, name2, preamble, response_prompt=NO_HEADER):
    substituted = TEMPLATE.substitute(name1=name1, name2=name2, preamble=preamble)
    if response_prompt is None:
        return substituted
    else:
        return f"{substituted}{response_prompt}"


def main(args):
    model = args.model
    # results_dir = args.results_dir
    instruct = args.instruct
    mode = args.mode

    model_name = model.replace("/", "_")

    # load the model
    lm = scorer.IncrementalLMScorer(model, device=args.device, trust_remote_code=True)

    eval_file = config.MODELS[model]
    eval_path = f"data/results/sorted-generations/rejection/{eval_file}-{mode}.csv"
    eval = utils.read_csv_dict(eval_path)

    eval_preprocessed = []
    for entry in eval:
        if instruct:
            no_stimulus = chat_template(
                entry["preamble"],
                tok=lm.tokenizer,
                response_prompt=f'{NO_HEADER} {entry["continuation"]}',
            )
            wait_stimulus = chat_template(
                entry["preamble"],
                tok=lm.tokenizer,
                response_prompt=f'{HEYWAIT_HEADER} {entry["continuation"]}',
            )
        else:
            no_stimulus = dialog_template(
                name1=entry["name1"],
                name2=entry["name2"],
                preamble=entry["preamble"],
                response_prompt=f'{NO_HEADER} {entry["continuation"]}',
            )
            wait_stimulus = dialog_template(
                name1=entry["name1"],
                name2=entry["name2"],
                preamble=entry["preamble"],
                response_prompt=f'{HEYWAIT_HEADER} {entry["continuation"]}',
            )
        eval_preprocessed.append(("no", no_stimulus))
        eval_preprocessed.append(("wait", wait_stimulus))

    print(eval_preprocessed[:4])

    batches = DataLoader(eval_preprocessed, batch_size=args.batch_size)

    scores = []
    types, stimuli = list(zip(*eval_preprocessed))

    for batch in tqdm(batches):
        headers, stimuli = batch
        score = lm.sequence_score(stimuli, bow_correction=True)
        scores.extend(score)

    pathlib.Path(args.results_dir).mkdir(exist_ok=True, parents=True)

    # scores = [(s,) for s in scores]
    scores = list(zip(types, scores))

    utils.write_csv(
        data=scores,
        path=f"{args.results_dir}/{config.MODELS[model]}.csv",
        header=["header", "score"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
    )
    parser.add_argument(
        "--results-dir", type=str, default="data/results/dgrc/rejection-arc"
    )
    parser.add_argument("--mode", type=str, default="arc")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--instruct", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    main(args)
