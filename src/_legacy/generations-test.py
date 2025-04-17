# pip install -U transformers
# pip install -U jinja2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import pandas as pd
import src.utils as utils

# === Load model and tokenizer ===
model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token

# === Custom chat template prompt ===
def chat_prompt(sentence, tok=tokenizer):
    return tok.apply_chat_template(
        [
            {"role": "system", "content": "Please respond to the following message as naturally as possible."},
            {"role": "user", "content": sentence}
        ],
        tokenize=False,
        add_generation_prompt=True
    )

# === Set generation parameters ===
set_seed(1024)
my_batch_size = 5  # Adjust as needed
my_num_return_sequences = 5
my_max_length = 50

# === Example input contexts ===
# === Try 1: Using the entire dataframe
df = pd.read_csv('data/used_items.csv')
eval_df = df.head(30).to_dict(orient="records")

# === Try 2: Using a list of sentences
# contexts = [
#     "Ellie said, 'The nurse adopted a rescue dog,' and Marco replied, ",
#     "Tina whispered, 'The teacher saw the shooting star,' and Paul answered, ",
#     "Jamie remarked, 'The doctor canceled the appointment,' and Riley responded, ",
#     "Alex said, 'The artist painted a beautiful mural,' and Jordan replied, ",
#     "Sophie exclaimed, 'The chef won the cooking contest,' and Leo responded, ",
#     "Mia noted, 'The scientist discovered a new element,' and Noah said, ",
#     "Lily mentioned, 'The author signed the books,' and Jack answered, ",
#     "Ethan commented, 'The gardener trimmed the roses,' and Ava responded, ",
#     "Olivia observed, 'The firefighter rescued the kitten,' and Lucas replied, ",
#     "Grace remarked, 'The engineer fixed the machine,' and Henry said, ",
#     "Nathan said, 'The pilot landed in the storm,' and Emma responded, ",
#     "Aria whispered, 'The dancer slipped on stage,' and Max replied, ",
#     "Isla noted, 'The director postponed the movie,' and Owen said, ",
#     "Ella said, 'The student solved the riddle,' and Caleb replied, ",
#     "Charlotte exclaimed, 'The baby smiled at me,' and Dylan answered, ",
#     "Harper commented, 'The driver missed the turn,' and Mason said, ",
#     "Zoe mentioned, 'The swimmer broke the record,' and Liam responded, ",
#     "Scarlett observed, 'The lawyer presented the case,' and Isaac replied, ",
#     "Chloe said, 'The actor forgot his lines,' and Elijah responded, ",
#     "Sienna noted, 'The barista spilled the coffee,' and Aiden replied, "
# ]

# === Use DataLoader to create batches of contexts ===
context_dl = DataLoader(eval_df, batch_size=my_batch_size)
# context_dl = DataLoader(contexts, batch_size=my_batch_size)

# === Store final outputs ===
all_outputs = []

for batch in tqdm(context_dl, desc="Generating responses"):
    # === Format prompts
    ''' 
    ==================================================================
        NOTE:   batch is currently taking ALL rows in the dataset
                so the output will look a little weird for now
    ==================================================================
    ''' 

    prompts = [chat_prompt(ctx) for ctx in batch]

    # === Tokenize
    encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    # === Generate
    generations = model.generate(
        **encoded,
        num_return_sequences=my_num_return_sequences,
        max_length=my_max_length,
        do_sample=True,
        top_p=0.8,
        temperature=0.9
    )

    # === Decode
    decoded = tokenizer.batch_decode(generations, skip_special_tokens=True)
    
    # === Remove the prompt part (ver. for multiple generations)
    for i, prompt in enumerate(batch):
        start = i * my_num_return_sequences
        end = start + my_num_return_sequences
        cleaned = [decoded[j].split(prompt)[-1].strip('\nassistant\n') for j in range(start, end)]
        all_outputs.append(cleaned)

    breakpoint()

print(all_outputs)