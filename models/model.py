#from unsloth import FastLanguageModel 
#from unsloth import is_bfloat16_supported
import os
from random import randrange
from functools import partial
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          HfArgumentParser,
                          Trainer,
                          TrainingArguments,
                          DataCollatorForLanguageModeling,
                          EarlyStoppingCallback,
                          pipeline,
                          logging,
                          set_seed)

import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, AutoPeftModelForCausalLM
from trl import SFTTrainer

class TopicModel:
    """
    Class for handling the LLaMA-3 8B model.

    Attributes:
        model: The loaded LLaMA-3 8B model.
    """

    def __init__(self, model_name: str, config: dict):
        # self.model = unslot.FastLanguageModel.from_pretrained(
        #     model_name=model_name,
        #     max_seq_length=config['model']['max_seq_length'],
        #     dtype="float16",
        #     load_in_4bit=config['model']['load_in_4bit']
        # )

        model_name = "meta-llama/Llama-2-7b-hf"
        # model_name = "meta-llama/Llama-2-13b-chat-hf"
        # model_name = "meta-llama/Llama-2-7b-chat-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")


    def create_bnb_config(self, config: dict, load_in_4bit, bnb_4bit_use_double_quant, bnb_4bit_quant_type, bnb_4bit_compute_dtype):
        """
        Configures model quantization method using bitsandbytes to speed up training and inference

        :param load_in_4bit: Load model in 4-bit precision mode
        :param bnb_4bit_use_double_quant: Nested quantization for 4-bit model
        :param bnb_4bit_quant_type: Quantization data type for 4-bit model
        :param bnb_4bit_compute_dtype: Computation data type for 4-bit model
        """

        bnb_config = BitsAndBytesConfig(
            load_in_4bit = config['model']['load_in_4bit'],
            bnb_4bit_use_double_quant = config['model'][bnb_4bit_use_double_quant],
            bnb_4bit_quant_type = config['model'][bnb_4bit_quant_type],
            bnb_4bit_compute_dtype = config['model'][bnb_4bit_compute_dtype]
        )

        return bnb_config

    # def add_lora_weights(self,  config: dict):
    #     """
    #     Adds the LoRA weights to the model.

    #     Args:
    #         config : config.
    #     """

    #     # TODO: put lora parameters inside config
        
    #     self.model = FastLanguageModel.get_peft_model(
    #         self.model,
    #         r = 16,
    #         target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
    #                         "gate_proj", "up_proj", "down_proj",],
    #         lora_alpha = 16,
    #         lora_dropout = 0, # Supports any, but = 0 is optimized
    #         bias = "none",    # Supports any, but = "none" is optimized
    #         # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    #         use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    #         random_state = 3407,
    #         max_seq_length = config['model']['max_seq_length'],
    #         use_rslora = False,  # We support rank stabilized LoRA
    #         loftq_config = None, # And LoftQ
    #     )


    def save_model(self, output_dir: str):
        """
        Saves the model to the specified directory.

        Args:
            output_dir (str): The directory to save the model to.
        """
        self.model.save_pretrained(output_dir)

    def load_model(self, model_name: str):
        """
        Loads a model from the specified model name.

        Args:
            model_name (str): The name of the model to load.
        """
        self.model = unslot.FastLanguageModel.from_pretrained(model_name)
