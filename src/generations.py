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

    decoded = tokenizer.batch_decode(generations, skip_special_tokens=True)

    # Group outputs using slicing
    grouped_outputs = []
    for i, context in enumerate(contexts):
        start = i * n_return
        end = start + n_return
        outputs = [s.split(context)[-1].strip() for s in decoded[start:end]]
        grouped_outputs.append(outputs)

    return grouped_outputs

def main(args):
    eval_path = args.eval_path
    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    
    df = pd.read_csv(eval_path)

    eval_data = utils.read_csv_dict(eval_path)
    eval_dl = DataLoader(eval_data, batch_size=args.batch_size)

    all_generations_vp1 = []
    all_generations_vp2 = []

    template = Template(args.template)

    for batch in tqdm(eval_dl):
        try:
            batch = {k: [str(v) for v in batch[k]] for k in batch}

            # Use actual length of this batch
            actual_batch_size = len(next(iter(batch.values())))

            contexts_vp1 = [
                template.substitute(
                    name1=batch['name1'][i],
                    name2=batch['name2'][i],
                    subj=batch['subj'][i],
                    vp=batch['vp1'][i]
                )
                for i in range(actual_batch_size)
            ]
            contexts_vp2 = [
                template.substitute(
                    name1=batch['name1'][i],
                    name2=batch['name2'][i],
                    subj=batch['subj'][i],
                    vp=batch['vp2'][i]
                )
                for i in range(actual_batch_size)
                # for i in range(batch_size) will give an error
                # if batch_size (e.g., 128) > remaining items to loop over
            ]

            gens_vp1 = generate_batch_output(contexts_vp1, args.num_return, tokenizer, model, args.device)
            gens_vp2 = generate_batch_output(contexts_vp2, args.num_return, tokenizer, model, args.device)

            all_generations_vp1.extend(gens_vp1)
            all_generations_vp2.extend(gens_vp2)

        except Exception as e:
            print(batch)
            raise e

    df['generations_vp1'] = all_generations_vp1
    df['generations_vp2'] = all_generations_vp2

    print(f"* -- Saving {args.model} data to: {results_dir} -- *")
    df.to_csv(results_dir / f"{args.model.replace('/', '_')}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # === When using a script ===
    parser.add_argument("--eval-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--num-return", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--template", type=str, required=False, default='$name1 said, "$subj $vp", and $name2 replied, ')

    # === One try-out (using GPU) ===
    # parser.add_argument("--eval-path", type=str, required=False, default='data/used_items.csv', help="CSV file path")
    # parser.add_argument("--results-dir", type=str, required=False, default='data/results/generations')
    # parser.add_argument("--batch-size", type=int, default=5)
    # parser.add_argument("--num-return", type=int, default=30)
    # parser.add_argument("--model", type=str, required=False, default='meta-llama/Meta-Llama-3-8B-Instruct')
    # parser.add_argument("--device", type=str, default="cuda:0")
    # parser.add_argument("--template", type=str, required=False, default='$name1 said, "$subj $vp", and $name2 replied, ')

    # === One try-out (using CPU) ===
    # parser.add_argument("--eval-path", type=str, required=False, default='data/used_items.csv', help="CSV file path")
    # parser.add_argument("--results-dir", type=str, required=False, default='data/results/generations')
    # parser.add_argument("--batch-size", type=int, default=2)
    # parser.add_argument("--num-return", type=int, default=10)
    # parser.add_argument("--model", type=str, required=False, default="HuggingFaceTB/SmolLM2-135M-Instruct")
    # parser.add_argument("--device", type=str, default="cpu")

    # === Global options ===
    args = parser.parse_args()
    main(args)

    # template = Template('$name1 said, "$subj $vp", and $name2 replied, ')
    # main(args, template)