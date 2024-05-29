# !pip install bertopic datasets accelerate bitsandbytes xformers adjustText
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps xformers trl peft accelerate bitsandbytes

import datasets
import torch

from datasets import load_dataset
# from unsloth import FastLanguageModel

dataset = load_dataset("CShorten/ML-ArXiv-Papers")["train"]

# Extract abstracts to train on and corresponding titles
abstracts = dataset["abstract"]
titles = dataset["title"]

# total length of the dataset
print('Total articles: ', len(abstracts))

# from huggingface_hub import notebook_login
# notebook_login()




