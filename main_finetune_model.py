import sys
import subprocess
import os

# below code is used to install the required packages in GPU cluster
# subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "accelerate"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "peft"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "bitsandbytes"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "trl"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])

import time
import os
import torch
import random
import wandb
import numpy as np
from datasets import load_dataset
from random import randrange, sample, seed
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
from rouge_score import rouge_scorer
from utils.argument_parser import ArgumentParser


# setting up seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Add your Hugging Face token to the environment variables
os.environ['HF_TOKEN'] = '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'


def initialize_wandb():
    wandb.init(project="topic_modeling", name="llama3_8b_ft")


def load_and_prepare_dataset(config):
    print("Loading dataset...")
    dataset = load_dataset(config['data']['train_data_name'], split="train")
    print(f"Dataset size: {len(dataset)}")
    return dataset


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


def load_model_and_tokenizer(model_id, bnb_config):
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        use_cache=False, 
        device_map="auto",
        token=os.environ["HF_TOKEN"]
    )
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=os.environ["HF_TOKEN"]
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Model and tokenizer loaded successfully!")
    
    return model, tokenizer


def prepare_model_for_training(model, peft_config):
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    return model


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds):
    """
    Compute the Rouge-1 score between the predicted and target topics.
    This custom metric will be used for training
    """

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(pred.strip()) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.strip()) for label in decoded_labels]

    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    result = scorer.score(decoded_preds, decoded_labels)

    result = {key: value * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


def train_and_save_model(model, tokenizer, dataset, peft_config, args):
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
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.train()
    print("Training completed successfully!")

    model_dir = "llama3_8b_ft_model_new"
    # save to local
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    # push to huhhingface hub
    trainer.model.push_to_hub("ankitagr01/llama_3_8b_ft_topic_new", token=os.environ["HF_TOKEN"])
    tokenizer.push_to_hub("ankitagr01/llama_3_8b_ft_topic_new", token=os.environ["HF_TOKEN"])
    print("Model saved successfully!")


if __name__ == "__main__":

    # Parse arguments and load config
    arg_parser = ArgumentParser()
    args = arg_parser.parse_args()
    config = arg_parser.load_config(args)

    dataset = load_and_prepare_dataset(config)

    # Hugging Face model id
    model_id = config['model']['student_model_name']

    # TODO: add parameters to config file
    # BitsAndBytesConfig int-4 config 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model, tokenizer = load_model_and_tokenizer(model_id, bnb_config)

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", 
            "up_proj", "down_proj",
        ]
    )

    model = prepare_model_for_training(model, peft_config)

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
        report_to="wandb",
        ddp_find_unused_parameters=False,
    )

    train_and_save_model(model, tokenizer, dataset, peft_config, args)
    wandb.finish()