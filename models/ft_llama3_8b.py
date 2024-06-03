import sys
import subprocess
import os

# subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "accelerate"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "peft"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "bitsandbytes"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "trl"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])

import time
from random import randrange, sample, seed
import os
import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
import numpy as np
from rouge_score import rouge_scorer

seed(42)


# Add your Hugging Face token to the environment variables
os.environ['HF_TOKEN'] = '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'


# Load dataset from the hub
dataset = load_dataset("ankitagr01/dynamic_topic_modeling_arxiv_abstracts", split="train")

print(f"Dataset size: {len(dataset)}")
# print(dataset[randrange(len(dataset))])

# #Reduce dataset to size N
# n_samples = sample(range(len(dataset)), k=1000)
# print(f"First 5 samples: {n_samples[:5]}")
# dataset = dataset.select(n_samples)
# print(f"Reduced dataset size: {len(dataset)}")



def format_instruction(sample):
	
    instructions = 'You are a helpful and respectful AI assistant for for labeling topics. Given the input abstract below, generate an appropriate small topic relevant to the input abstract. Give me response without explaination.'

    return f"""    
    Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {instructions}

    ### Input:
    {sample['Abstract']}

    ### Response:
    {sample['Topic']}
    """


print(format_instruction(dataset[randrange(len(dataset))]))



# Hugging Face model id
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = "mistralai/Mistral-7B-v0.1"

# BitsAndBytesConfig int-4 config 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    use_cache=False, 
    device_map="auto",
    token = os.environ["HF_TOKEN"], # if model is gated like llama or mistral
)
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token = os.environ["HF_TOKEN"], # if model is gated like llama or mistral
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# Load PEFT model
peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj", 
            "up_proj", 
            "down_proj",
        ]
)

# Prepare model for training
model = prepare_model_for_kbit_training(model)

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    # Replace -100 in the preds as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(pred.strip()) for pred in decoded_preds]

    decoded_labels = ["\n".join(label.strip()) for label in decoded_labels]
    # Compute ROUGscores

    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    result = scorer.score(decoded_preds, decoded_labels)
    # result = rouge.get_scores(
    #     predictions=decoded_preds, references=decoded_labels, use_stemmer=True, avg=True
    # )
    # Extract the median scores
    result = {key: value * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


#     per_device_train_batch_size=4,

# Training arguments
args = TrainingArguments(
    output_dir="./results/llama3_8b_ft",
    num_train_epochs=3,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=False,
    fp16=True,
    tf32=False,
    max_grad_norm=0.3,
    warmup_steps=5,
    lr_scheduler_type="linear",
    disable_tqdm=False,
    auto_find_batch_size=True,
    report_to="none",
    ddp_find_unused_parameters=False,
)

model = get_peft_model(model, peft_config)



trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="Abstract",
    max_seq_length=2048,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=format_instruction, 
    args=args,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics = preprocess_logits_for_metrics,
)

# train
trainer.train()

# save model
trainer.save_model()


# Save the fine-tuned model
trainer.save_model("llama3_8b_ft_model_new")
tokenizer.save_pretrained("llama3_8b_ft_model_new")
trainer.model.push_to_hub("ankitagr01/llama_3_8b_ft_topic_new", token=os.environ["HF_TOKEN"])
tokenizer.push_to_hub("ankitagr01/llama_3_8b_ft_topic_new", token=os.environ["HF_TOKEN"])

