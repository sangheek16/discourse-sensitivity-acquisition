import argparse
import pathlib
import utils

from copy import deepcopy
from minicons import scorer
from string import Template
from torch.utils.data import DataLoader
from tqdm import tqdm


DIALOG_TEMPLATE = Template('$name1 said, "$sentence" $name2 replied, "')


def construct_items(no, wait, cont1, cont2):

    n_c1 = [f"{n} {c1}" for n, c1 in zip(no, cont1)]
    n_c2 = [f"{n} {c2}" for n, c2 in zip(no, cont2)]
    w_c1 = [f"{w} {c1}" for w, c1 in zip(wait, cont1)]
    w_c2 = [f"{w} {c2}" for w, c2 in zip(wait, cont2)]

    return n_c1, n_c2, w_c1, w_c2


def chat_template(sentence, instruction, tok, response_prompt=None):
    """
    A function that applies the model's chat template to simulate
    an interaction environment. Two possible options
    """
    if response_prompt is None:
        if instruction is None:
            return tok.apply_chat_template(
                [
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
                        "content": instruction,
                    },
                    {"role": "user", "content": sentence},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
        if instruction is None:
            return tok.apply_chat_template(
                [
                    {"role": "user", "content": sentence},
                    {"role": "assistant", "content": response_prompt},
                ],
                tokenize=False,
                continue_final_message=True,
            )
        else:
            return tok.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": instruction,
                    },
                    {"role": "user", "content": sentence},
                    {"role": "assistant", "content": response_prompt},
                ],
                tokenize=False,
                continue_final_message=True,
            )


def dialog_template(name1, name2, sentence, **kwargs):
    substituted = DIALOG_TEMPLATE.substitute(
        name1=name1, name2=name2, sentence=sentence
    )
    return substituted


def main(args):
    eval_path = args.eval_path
    model = args.model
    results_dir = args.results_dir
    instruct = args.instruct

    model_name = model.replace("/", "_")

    # load the model
    lm = scorer.IncrementalLMScorer(model, device=args.device, trust_remote_code=True)

    # read eval
    eval_data = utils.read_csv_dict(eval_path)

    formatted_stimuli = []
    for item in eval_data:
        item_copy = deepcopy(item)
        if instruct:
            item_copy.update(
                {
                    "no_prefix": chat_template(
                        item_copy["preamble"], None, lm.tokenizer, item["no"]
                    ),
                    "wait_prefix": chat_template(
                        item_copy["preamble"], None, lm.tokenizer, item["wait"]
                    ),
                }
            )
        else:
            item_copy.update(
                {
                    "no_prefix": dialog_template(
                        item["name1"], item["name2"], item_copy["preamble"]
                    )
                    + item["no"],
                    "wait_prefix": dialog_template(
                        item["name1"],
                        item["name2"],
                        item_copy["preamble"],
                    )
                    + item["wait"],
                }
            )

        formatted_stimuli.append(item_copy)

    print(formatted_stimuli[:2])

    eval_dl = DataLoader(formatted_stimuli, batch_size=args.batch_size)

    results = []

    for batch in tqdm(eval_dl):
        idx = batch["idx"]

        no_prefix = batch["no_prefix"]
        wait_prefix = batch["wait_prefix"]

        no = batch["no"]
        wait = batch["wait"]

        continuation1 = batch["continuation1"]
        continuation2 = batch["continuation2"]

        if "gpt2" in model_name or "pythia" in model_name:
            bos_tok = True
        else:
            bos_tok = False

        no_prefix_c1 = lm.conditional_score(
            no_prefix, continuation1, bos_token=bos_tok, bow_correction=True
        )
        no_prefix_c2 = lm.conditional_score(
            no_prefix, continuation2, bos_token=bos_tok, bow_correction=True
        )
        wait_prefix_c1 = lm.conditional_score(
            wait_prefix, continuation1, bos_token=bos_tok, bow_correction=True
        )
        wait_prefix_c2 = lm.conditional_score(
            wait_prefix, continuation2, bos_token=bos_tok, bow_correction=True
        )

        no_c1 = lm.conditional_score(
            no, continuation1, bos_token=bos_tok, bow_correction=True
        )
        no_c2 = lm.conditional_score(
            no, continuation2, bos_token=bos_tok, bow_correction=True
        )
        wait_c1 = lm.conditional_score(
            wait, continuation1, bos_token=bos_tok, bow_correction=True
        )
        wait_c2 = lm.conditional_score(
            wait, continuation2, bos_token=bos_tok, bow_correction=True
        )

        for item in zip(
            idx,
            no_prefix_c1,
            no_prefix_c2,
            wait_prefix_c1,
            wait_prefix_c2,
            no_c1,
            no_c2,
            wait_c1,
            wait_c2,
        ):
            results.append([*item])

    pathlib.Path(results_dir).mkdir(exist_ok=True, parents=True)

    utils.write_csv(
        results,
        path=f"{results_dir}/{model_name}.csv",
        header=[
            "idx",
            "no_prefix_c1",
            "no_prefix_c2",
            "wait_prefix_c1",
            "wait_prefix_c2",
            "no_c1",
            "no_c2",
            "wait_c1",
            "wait_c2",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval-path", type=str, default="data/stimuli/kim22-arc-dcpmi.csv"
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
    )
    parser.add_argument(
        "--results-dir", type=str, default="data/results/kim22-arc-dcpmi.csv"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--instruct", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    main(args)
