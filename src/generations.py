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

# Regex to split on sentence-final punctuation
re_sentend = re.compile(
    r'(?<!\b[A-Z]\.)'
    r'(?<!\b[Mm]rs\.)'
    r'(?<!\b[MmDdSsJj]r\.)'
    r'(?<=[\.\?\!])'
    r'[ \n\t](?!["\'])'
    r'|'
    r'(?<!\b[A-Z]\.)'
    r'(?<!\b[Mm]rs\.)'
    r'(?<!\b[MmDdSsJj]r\.)'
    r'(?<=[\.\?\!] ["\'])'
    r'[ \n\t]+'
)

def is_complete_sentence(sent):
    return bool(re.match(r'^[A-Z0-9"“].*[\.\?\!]$', sent.strip()))

def sent_tokenize(chunk):
    def get_last_complete_idx(split_sents, last_complete_sent):
        for idx in reversed(range(len(split_sents))):
            if split_sents[idx].strip() == last_complete_sent:
                return idx
        return len(split_sents) - 1  # fallback in case not found

    if isinstance(chunk, list):
        results = []
        for text in chunk:
            split_sents = re.split(re_sentend, text)
            complete = [s.strip() for s in split_sents if is_complete_sentence(s)]
            if complete:
                last_complete = get_last_complete_idx(split_sents, complete[-1])
                combined = ' '.join(split_sents[:last_complete + 1]).strip()
                results.append(combined)
            else:
                results.append('')
        return results
    else:
        split_sents = re.split(re_sentend, chunk)
        complete = [s.strip() for s in split_sents if is_complete_sentence(s)]
        if complete:
            last_complete = get_last_complete_idx(split_sents, complete[-1])
            return [' '.join(split_sents[:last_complete + 1]).strip()]
        else:
            return ['']

# def sent_tokenize(chunk):
#     if isinstance(chunk, list):
#         results = []
#         for text in chunk:
#             split_sents = re.split(re_sentend, text)
#             complete = [s.strip() for s in split_sents if is_complete_sentence(s)]

#             if complete:
#                 # include all sentences up to and including the last complete one
#                 last_complete = split_sents.index(complete[-1])
#                 combined = ' '.join(split_sents[:last_complete + 1]).strip()
#                 results.append(combined)

#             else:
#                 results.append('')
#         return results
#     else:
#         split_sents = re.split(re_sentend, chunk)
#         complete = [s.strip() for s in split_sents if is_complete_sentence(s)]
#         if complete:
#             last_complete = split_sents.index(complete[-1])
#             return [' '.join(split_sents[:last_complete + 1]).strip()]
#         else:
#             return ['']

'''
def generate_batch_output(contexts, n_return, tokenizer, model, device):
    # instruction_prefix = "Then the other person said:"
    # prompts = [f'{ctx} {instruction_prefix}' for ctx in contexts]

    # instruction_prefix = "\nAssistant:"
    # prompts = [f"The following is a natural, friendly conversation.\nUser: {ctx} \nAssistant:" for ctx in contexts]

    instruction_prefix = "\nUser:"
    prompts = [f"Please respond to the following message as naturally as possible. \nUser: {ctx}" for ctx in contexts]
     
    encoded = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

    set_seed(1024)

    generations = model.generate(
        **encoded, 
        num_return_sequences=n_return, 

        # max_length=50, 
        # top_p=0.8, 
        # temperature=0.9,

        max_length=60,
        top_p = None,
        temperature=0.7,
        top_k=50,

        repetition_penalty=1.2,  # >1.0 penalizes repeated generations

        do_sample=True,
        length_penalty=1.0
    )

    decoded = tokenizer.batch_decode(generations, skip_special_tokens=True)
    breakpoint()

    # Group outputs using slicing
    grouped_outputs = []
    for i, context in enumerate(contexts):
        start = i * n_return
        end = start + n_return

        breakpoint()

        outputs = [s.split(context)[-1].strip() for s in decoded[start:end]]

        breakpoint()

        outputs_complete_sents = [sent_tokenize(s.split(context)[-1].strip()) for s in decoded[start:end]]
        
        grouped_outputs.append(outputs)

    return grouped_outputs
'''

def generate_batch_output(contexts, n_return, tokenizer, model, device):
    instruction_prefix = "\nUser:"
    prompts = [f"Please respond to the following message as naturally as possible. \nUser: {ctx}" for ctx in contexts]
     
    encoded = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

    # Initialize grouped outputs
    grouped_outputs = []
    seed = 1024  # Starting seed value

    for i, context in enumerate(contexts):
        # Initialize the complete sentences list for this context
        outputs_complete_sents = []

        while True:
            # Set the seed and increment it for the next round
            set_seed(seed)
            seed -= 1  # Lower the seed number

            # Generate responses
            generations = model.generate(
                **encoded, 
                num_return_sequences=n_return, 
                max_length=60,
                top_p=None,
                temperature=0.7,
                top_k=50,
                repetition_penalty=1.2,
                do_sample=True,
                length_penalty=1.0
            )

            decoded = tokenizer.batch_decode(generations, skip_special_tokens=True)

            # Group outputs using slicing
            start = i * n_return
            end = start + n_return

            # Process and split the responses
            outputs = [s.split(context)[-1].strip() for s in decoded[start:end]]

            # Tokenize to sentences and filter out empty elements
            new_complete_sents = [
                sent for s in [sent_tokenize(s.split(context)[-1].strip()) for s in decoded[start:end]]
                for sent in s if sent.strip() != ''
            ]
            outputs_complete_sents.extend(new_complete_sents)

            # Check if we have enough valid sentences
            if len(outputs_complete_sents) >= (end - start):
                # If we have more than enough non-empty sentences, trim to (end - start)
                if len(outputs_complete_sents) > (end - start):
                    outputs_complete_sents = outputs_complete_sents[:(end - start)]
                break

        # Append the valid outputs
        grouped_outputs.append(outputs_complete_sents)

    return grouped_outputs

def main(args):
    eval_path = args.eval_path
    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    template = Template(args.template)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    
    df = pd.read_csv(eval_path)

    eval_data = utils.read_csv_dict(eval_path)
    eval_dl = DataLoader(eval_data, batch_size=args.batch_size)

    all_generations_vp1 = []
    all_generations_vp2 = []
    is_first_batch = True

    for batch_idx, batch in enumerate(tqdm(eval_dl)):
        try:
            batch = {k: [str(v) for v in batch[k]] for k in batch}
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
            ]

            gens_vp1 = generate_batch_output(contexts_vp1, args.num_return, tokenizer, model, args.device)
            gens_vp2 = generate_batch_output(contexts_vp2, args.num_return, tokenizer, model, args.device)

            # Create a temp df for this batch
            df_batch = pd.DataFrame({k: batch[k] for k in batch})
            df_batch['generations_vp1'] = gens_vp1
            df_batch['generations_vp2'] = gens_vp2

            # Save or append to CSV
            output_path = results_dir / f"{args.model.replace('/', '_')}.csv"
            df_batch.to_csv(output_path, mode='w' if is_first_batch else 'a', index=False, header=is_first_batch)
            is_first_batch = False

        except Exception as e:
            print(f"\nError in batch {batch_idx}:")
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