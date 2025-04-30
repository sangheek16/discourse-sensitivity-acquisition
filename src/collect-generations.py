"""
Script to collect generations from a model

Allows two types of models:
1. Instruct-tuned -- this will come with a chat template
2. Non-instruct tuned -- this will come with a dialogue format

With two settings:
1. rejection -- with the response starting with "No, that's not true"
2. freeform -- with only a response prompt, letting the model generated whatever.

Accepts the following generation parameters:
- Top-k
- Top-p
- Temperature

Results are saved as a big json file.
"""

import argparse
import pathlib
import torch
import utils

from string import Template
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from minicons import scorer
from tqdm import tqdm


INSTRUCTION = "Please respond to the following message as naturally as possible, using a single sentence, as if we were talking to each other. Please keep it short."

REJECTION = "No, that's not true!"

TEMPLATE = Template('$name1 said, "$subj $vp", and $name2 replied, "')


def chat_template(sentence, tok, response_prompt=None):
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


"""
default:

System: blah
User: blah <end>


add_generation_prompt = True

System: blah
User: blah
Assistant:

continue_final_message=False,
System: blah
User: blah
Assistant: blah <end_assistant_turn>

continue_final_message=True,
System: blah
User: blah
Assistant: blah 

"""


def dialog_template(name1, name2, subj, vp, response_prompt=None):
    substituted = TEMPLATE.substitute(name1=name1, name2=name2, subj=subj, vp=vp)
    if response_prompt is None:
        return substituted
    else:
        return f"{substituted}{response_prompt}"


def generate_and_decode(lm, batch, p, k, t, num_gen=10, max_new=20, device="cuda:0"):
    encoded = lm.tokenizer(
        batch, return_tensors="pt", add_special_tokens=False, padding=True
    )
    encoded = encoded.to(device)

    # store input lengths
    # encoded.input_ids' shape = [batch_size, input_length]
    input_length = encoded.input_ids.shape[-1]

    """
    inputs = ["sentence1", "sentence2", ..., "sentence8"]
    encoded.input_ids = [[1x50], [1x50], ..., ]

    shape of encoded.input_ids:
    [8, 50]

    Let's say these are the lengths of the generations: 
    [[1x100], [1x100], ]

    """

    set_seed(1024)

    generations = lm.model.generate(
        **encoded,
        num_return_sequences=num_gen,
        max_new_tokens=max_new,
        do_sample=True,
        tokenizer=lm.tokenizer,
        top_p=p,
        temperature=t,
        top_k=k,
        repetition_penalty=1.2,
    )

    decoded = []

    """
    decoded = [
        [generation1_1, generation1_2, ..., generation1_10], 
        [generation2_1, generation2_2, ..., generation2_10],  
        ...
    ]    
    """

    for i, gen in enumerate(generations[:, input_length:].split([num_gen] * len(batch))):
        decoded_sentences = [
            x for x in lm.tokenizer.batch_decode(gen, skip_special_tokens=True)
        ]

        joined = [f"{batch[i]}{d}" if d.startswith(" ") else f"{batch[i]} {d}" for d in decoded_sentences]

        lm.tokenizer.padding_side = "right"
        decoded_scores = lm.sequence_score(joined)
        lm.tokenizer.padding_side = "left"
        
        decoded.append(list(zip(decoded_sentences, decoded_scores)))

    return decoded


def main(args):
    model = args.model
    model_name = model.replace("/", "_")

    # lm = AutoModelForCausalLM.from_pretrained(model)
    # tokenizer = AutoTokenizer.from_pretrained(model)
    # tokenizer.padding_side = "left"
    # lm.to("cuda:0")
    lm = scorer.IncrementalLMScorer(model, device=args.device)
    lm.tokenizer.padding_side = "left"

    if lm.tokenizer.pad_token is None:
        lm.tokenizer.pad_token_id = lm.tokenizer.eos_token_id

    topp = args.topp
    topk = args.topk
    temp = args.temp
    num_generations = args.num_gen
    max_gen = args.max_gen

    if topp == -1:
        topp = None

    batch_size = args.batch_size
    device = args.device

    instruct = args.instruct
    response = args.response  # none or actual string

    results = {
        "model": model,
        "instruct": instruct,
        "top_p": topp,
        "top_k": topk,
        "temperature": temp,
        "num_generations": num_generations,
        "max_gen": max_gen,
        "response": response,
        "generation_vp1": [],  # {id, list}
        "generation_vp2": [],  # {id, list}
    }

    # read eval/analysis data
    analysis_data = utils.read_csv_dict(args.analysis_data)

    # stimuli generation:
    stimuli = []
    for i, entry in enumerate(analysis_data):
        idx = i + 1
        sentence1 = f"{entry['subj']} {entry['vp1']}"
        sentence2 = f"{entry['subj']} {entry['vp2']}"
        if args.instruct:
            sentence1, sentence2 = f"{sentence1}.", f"{sentence2}."
            stimulus1 = chat_template(
                sentence1, tok=lm.tokenizer, response_prompt=response
            )
            stimulus2 = chat_template(
                sentence2, tok=lm.tokenizer, response_prompt=response
            )
        else:
            stimulus1 = dialog_template(
                name1=entry["name1"],
                name2=entry["name2"],
                subj=entry["subj"],
                vp=entry["vp1"],
                response_prompt=response,
            )
            stimulus2 = dialog_template(
                name1=entry["name1"],
                name2=entry["name2"],
                subj=entry["subj"],
                vp=entry["vp2"],
                response_prompt=response,
            )

        stimuli.append((idx, stimulus1, stimulus2))

    # print(stimuli[0][-1])

    batches = DataLoader(stimuli, batch_size=batch_size)
    for j, batch in enumerate(tqdm(batches)):
        idx, stimuli1, stimuli2 = batch
        idx = idx.tolist()

        decoded1 = generate_and_decode(
            lm,
            stimuli1,
            p=topp,
            k=topk,
            t=temp,
            num_gen=num_generations,
            max_new=max_gen,
            device=device,
        )

        decoded2 = generate_and_decode(
            lm,
            stimuli2,
            p=topp,
            k=topk,
            t=temp,
            num_gen=num_generations,
            max_new=max_gen,
            device=device,
        )

        # idxes = [[i]*num_generations for i in idx]

        if j == 0:
            print(decoded1[:5])

        for i, d1 in zip(idx, decoded1):
            results["generation_vp1"].append({"idx": i, "sentences": d1})

        for i, d2 in zip(idx, decoded2):
            results["generation_vp2"].append({"idx": i, "sentences": d2})

    pathlib.Path(args.outdir).mkdir(exist_ok=True, parents=True)
    utils.write_json(results, f"{args.outdir}/{args.outfile}")


"""
topp = args.topp
topk = args.topk
temp = args.temp
num_generations = args.num_gen
max_gen = args.max_gen

batch_size = args.batch_size
device = args.device

instruct = args.instruct
response = args.response  # none or actual string
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="HuggingFaceTB/SmolLM2-360M-Instruct"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_gen", type=int, default=10)
    parser.add_argument("--max_gen", type=int, default=20)
    parser.add_argument("--topp", "-p", type=float, default=None)
    parser.add_argument("--topk", "-k", type=int, default=0)
    parser.add_argument("--temp", "-t", type=float, default=1.0)
    parser.add_argument("--instruct", action="store_true")
    parser.add_argument("--response", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="data/generations/smolm2-360m")
    parser.add_argument("--outfile", type=str, default="gens_0_0_1-0.json")
    parser.add_argument("--analysis_data", type=str, default="data/used_items.csv")

    args = parser.parse_args()
    main(args)
