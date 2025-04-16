import argparse
import pathlib
import pandas as pd
from tqdm import tqdm
from string import Template
import torch
from torch.utils.data import DataLoader
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM
import utils

def generate_batch_output(contexts, n_return, tokenizer, model, device):
    prompts = [f"Please respond to the following message as naturally as possible.\nUser: {ctx}" for ctx in contexts]
    
    # breakpoint()
    
    encoded = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

    set_seed(1024)

    generations = model.generate(
        **encoded, 
        num_return_sequences=n_return, 
        max_length=50, 
        do_sample=True,
        top_p=0.8, 
        temperature=0.9
    )

    # breakpoint()

    decoded = tokenizer.batch_decode(generations, skip_special_tokens=True)

    # breakpoint()

    # Group outputs using slicing
    grouped_outputs = []
    for i, context in enumerate(contexts):
        start = i * n_return
        end = start + n_return
        outputs = [s.split(context)[-1].strip() for s in decoded[start:end]]
        grouped_outputs.append(outputs)

    # breakpoint()

    return grouped_outputs


def main(args, template):
    eval_path = args.eval_path
    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    df = pd.read_csv(eval_path)

    eval_data = utils.read_csv_dict(eval_path)
    eval_dl = DataLoader(eval_data[:4], batch_size=args.batch_size)

    all_generations_vp1 = []
    all_generations_vp2 = []

    for batch in tqdm(eval_dl):
        # print(batch)
        contexts_vp1 = [
            template.substitute(name1=batch['name1'][i], name2=batch['name2'][i], subj=batch['subj'][i], vp=batch['vp1'][i])
            for i in range(args.batch_size)
        ]
        contexts_vp2 = [
            template.substitute(name1=batch['name1'][i], name2=batch['name2'][i], subj=batch['subj'][i], vp=batch['vp2'][i])
            for i in range(args.batch_size)
        ]
        # print(contexts_vp1)
        # breakpoint()

        gens_vp1 = generate_batch_output(contexts_vp1, args.num_return, tokenizer, model, args.device)
        print(gens_vp1)
        gens_vp2 = generate_batch_output(contexts_vp2, args.num_return, tokenizer, model, args.device)

        all_generations_vp1.extend(gens_vp1)
        all_generations_vp2.extend(gens_vp2)

    df['generations_vp1'] = all_generations_vp1
    df['generations_vp2'] = all_generations_vp2

    output_path = results_dir / f"{args.model.replace('/', '_')}.csv"
    df.to_csv(output_path, index=False)
    print(f"* -- saving {args.model} data to: {output_path} -- *")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument("--eval-path", type=str, required=True)
    # parser.add_argument("--results-dir", type=str, required=True)
    # parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--eval-path", type=str, required=False, default='data/used_items.csv', help="CSV file path")
    parser.add_argument("--results-dir", type=str, required=False, default='data/results/generations')
    parser.add_argument("--model", type=str, required=False, default='meta-llama/Meta-Llama-3-8B-Instruct')

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-return", type=int, default=15)

    template = Template('$name1 said, "$subj $vp", and $name2 replied, ')
    args = parser.parse_args()
    main(args, template)
