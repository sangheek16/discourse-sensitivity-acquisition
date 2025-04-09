import argparse
import pathlib
import utils
import pandas as pd
from tqdm import tqdm
from string import Template 

from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_context(my_template, name1, name2, subj, vp):
    return my_template.substitute(
        name1=name1, name2=name2,
        subj=subj, vp=vp)

def generate_output(input_context, my_tokenizer, my_model):
    # chat_prompt = my_tokenizer.apply_chat_template(
    #     [   {"role": "system", "content": "Please respond to the following message as naturally as possible."},
    #         { "role": "user", "content": input_context}],
    #     tokenize=False,
    #     add_generation_prompt=True
    #     )
    chat_prompt = "Please respond to the following message as naturally as possible.\nUser: " + input_context

    encoded = my_tokenizer(chat_prompt, return_tensors="pt")
    # encoded = encoded.to(args.device)
    encoded = encoded.to("cuda:0")

    set_seed(1024)

    generations = my_model.generate(
        **encoded, 
        num_return_sequences=10, 
        max_length=50, 
        do_sample=True, 
        top_p=0.8,
        temperature=0.9,
        # typical_p=0.9,
        # min_p=0.2,
        # top_p=0.8,
        # top_k = 5000,
        # force_words_ids=[[857]],
        stop_strings=["."],
        tokenizer=my_tokenizer
        )
    
    # TODO: decoded
    # decoded = my_tokenizer.batch_decode(generations,skip_special_tokens=True)
    
    return generations


def main(args, my_template):

    # get paths
    eval_path = args.eval_path
    results_dir = args.results_dir
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)

    # load the model
    model_name = args.model
    # model_name = model_name.replace("/", "_")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generation_model = AutoModelForCausalLM.from_pretrained(model_name)
    generation_model.to(args.device)

    # load dataset
    df = pd.read_csv(eval_path)

    # add the generated context to df
    tqdm.pandas()
    df['gencontext_vp1'] = df.progress_apply(
        lambda x: generate_context(
            my_template,
            x['name1'], x['name2'], x['subj'], x['vp1']
        ),
        axis=1)
    df['gencontext_vp2'] = df.progress_apply(
        lambda x: generate_context(
            my_template,
            x['name1'], x['name2'], x['subj'], x['vp2']
        ),
        axis=1)
    
    # generate
    breakpoint()

    # TODO: decode and save the output into dictionary format in a dataframe
    x = generate_output(df.iloc[0]['gencontext_vp1'], tokenizer, generation_model)
    tokenizer.batch_decode(x,skip_special_tokens=True)


if __name__ == "__main__":
    # define my args
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval-path", type=str, required=True, default="data/used_items.csv", help="csv file")
    parser.add_argument("--results-dir", type=str, required=True, default="data/results/generations")
    parser.add_argument("--model", type=str, required=True, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0", help="choose cuda:0 or cpu")
    
    # define my template
    context_template = Template('$name1 said, "$subj $vp", and $name2 replied, ')

    # call
    args = parser.parse_args()
    main(args, context_template)