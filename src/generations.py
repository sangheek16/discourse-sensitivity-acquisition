import argparse
import pathlib
import pandas as pd
from tqdm import tqdm
from string import Template
import torch
from torch.utils.data import DataLoader
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM
import utils
import re

# TODO: this function either doesn't seem to properly work
#       or it's already at the generation stage that a full
#       sentence is not showing up.

def trim_to_sentence_end(text):
    text = text.strip().strip('"').strip()  # remove double quotes
    # Cut at last full stop punctuation
    match = re.search(r'([.!?])(\s*["\']?\s*)$', text)
    if match:
        end_idx = match.end()
        return text[:end_idx].strip()
    return text.strip()

def clean_response(s, prefix):
    # TODO: Somehow this doesn't work for \nAssistant:
    s = s.strip()
    clean_s = s.split(prefix)[-1]
    return clean_s

def generate_batch_output(contexts, n_return, tokenizer, model, device):
    # instruction_prefix = "Then the other person said:"
    # prompts = [f'{ctx} {instruction_prefix}' for ctx in contexts]

    # instruction_prefix = "\nAssistant:"
    # prompts = [f"The following is a natural, friendly conversation.\nUser: {ctx} \nAssistant:" for ctx in contexts]

    instruction_prefix = "\nUser:"
    prompts = [f"Please respond to the following message as naturally as possible. \nUser: {ctx}" for ctx in contexts]
     
    encoded = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

    set_seed(1024)

    # TODO: Is this ensuring that the response ends as a full sentence?
    eos_tok_id = tokenizer.eos_token_id 

    generations = model.generate(
        **encoded, 
        num_return_sequences=n_return, 

        # max_length=50, 
        # top_p=0.8, 
        # temperature=0.9,

        max_length=60,
        # top_p = None,
        temperature=0.7,
        top_k=50,

        eos_token_id=eos_tok_id,
        repetition_penalty=1.2,  # >1.0 penalizes repeated generations

        do_sample=True,
        length_penalty=1.0
    )

    decoded = tokenizer.batch_decode(generations, skip_special_tokens=True)

    # Group outputs using slicing
    grouped_outputs = []
    for i, context in enumerate(contexts):
        start = i * n_return
        end = start + n_return
        outputs = [s.split(context)[-1].strip() for s in decoded[start:end]]

        outputs = [trim_to_sentence_end(s) for s in outputs]
        outputs = [clean_response(s, instruction_prefix) for s in outputs]
        
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