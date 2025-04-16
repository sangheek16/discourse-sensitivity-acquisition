import argparse
import pathlib
import pandas as pd
from tqdm import tqdm
from string import Template
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM

def generate_context(input_template, **kwargs):
    return input_template.substitute(**kwargs)

def generate_output(input_context, tokenizer, model, device):
    chat_prompt = f"Please respond to the following message as naturally as possible.\nUser: {input_context}"
    encoded = tokenizer(chat_prompt, return_tensors="pt").to(device)
    
    set_seed(1024)

    generations = model.generate(
        **encoded, 
        num_return_sequences=2, max_length=50, 
        do_sample=True,
        top_p=0.8, temperature=0.9
    )

    decoded = tokenizer.batch_decode(generations, skip_special_tokens=True)
    output = [s.split(input_context)[-1] for s in decoded]
    output = [s.strip() for s in output]

    return output


def main(args, template):
    eval_path = args.eval_path
    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)

    df = pd.read_csv(eval_path)
    # df = df.head(5)

    tqdm.pandas()
    df['context_vp1'] = df.progress_apply(lambda x: generate_context(template, name1=x['name1'], name2=x['name2'], subj=x['subj'], vp=x['vp1']), axis=1)
    df['context_vp2'] = df.progress_apply(lambda x: generate_context(template, name1=x['name1'], name2=x['name2'], subj=x['subj'], vp=x['vp2']), axis=1)
    df['generations_vp1'] = df['context_vp1'].progress_apply(lambda x: generate_output(x, tokenizer, model, args.device))
    df['generations_vp2'] = df['context_vp2'].progress_apply(lambda x: generate_output(x, tokenizer, model, args.device))

    # df.to_csv(f"results_dir}/{args.model.replace("/", "_")}.csv", index=False)

    print(f"* -- saving {args.model} data to: {results_dir} -- *")
    df.to_csv(f"{results_dir}/{args.model.replace('/', '_')}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval-path", type=str, required=True, help="csv file")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0", help="choose cuda:0 or cpu")

    # parser.add_argument("--eval-path", type=str, required=False, default='data/used_items.csv', help="CSV file path")
    # parser.add_argument("--results-dir", type=str, required=False, default='data/results/generations')
    # parser.add_argument("--model", type=str, required=False, default='meta-llama/Meta-Llama-3-8B-Instruct')
    # parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0 or cpu")

    template = Template('$name1 said, "$subj $vp", and $name2 replied, ')
    args = parser.parse_args()
    main(args, template)