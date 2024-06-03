# import unslot
import transformers
import torch
# from unsloth import FastLanguageModel
from typing import List
from transformers import BitsAndBytesConfig
import csv
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class TopicGenerator:
    """
    Class for generating topics using LLaMA-3 70B model.

    Attributes:
        model: The loaded LLaMA-3 70B model.
        few_shot_examples (str): Few-shot examples for prompting the model.
    """

    def __init__(self, model_name: str):
        # self.model = unslot.UnsloTModel.from_pretrained(model_name)   # fix: this cant be from unsloth, no llama3-79b in unsloth
        # self.model = FastLanguageModel.from_pretrained(model_name)   # fix: this cant be from unsloth, no llama3-79b in unsloth

        # model_id = "meta-llama/Meta-Llama-3-70B"
        # pipeline = transformers.pipeline(
        #     "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
        # )

        # model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
        # Load the model with int8 precision
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,  # Set to True for int4
            device_map="auto"   # This automatically places the model on the GPU
        )
        # quantization_config=BitsAndBytesConfig(load_in_8bit=True)


    def extract_topic(self, output_text: str, start_marker: str, end_marker: str) -> str:

        start = output_text.find(start_marker) + len(start_marker)
        end = len(output_text) if end_marker is None else output_text.find(end_marker, start)
        
        if start == -1 + len(start_marker) or end == -1 or start >= end:
            return ""
        
        topic = output_text[start:end].strip()
        # Strip leading and trailing quotes
        topic = topic.strip('"').strip("'")

        # Remove 'Topic:' prefix if present
        if topic.startswith("Topic:"):
            topic = topic[len("Topic:"):].strip()
        
        return topic


    def generate_topic(self, abstract: str) -> str:
        """
        Generates a topic for a given abstract and introduction.

        Args:
            abstract (str): The abstract of the paper.
            introduction (str): The introduction of the paper.

        Returns:
            str: Generated topic.
        """

        prompt = f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful and respectful AI assistant for for labeling topics<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        
        Given the abstract below, generate an appropriate small topic relevant to the abstract. Give me response without explaination.
        Abstract: {abstract} <|eot_id|>

        <|start_header_id|>assistant<|end_header_id|>
        """
    
        inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')
        
        generated_ids = self.model.generate(inputs.input_ids.cuda(), max_new_tokens=256, do_sample=True, top_p=0.95, temperature=0.8)
        topic = self.tokenizer.batch_decode(generated_ids)[0]
        # print('topiccc:', topic)

        start_marker = '<|start_header_id|>assistant<|end_header_id|>'
        end_marker = '<|eot_id|>'
        topic = self.extract_topic(topic, start_marker, end_marker)
        print('topic:', topic)

        return topic
    

    def generate_topics(self, abstracts: List[str]) -> List[str]:
        """
        Generates topics for a list of abstracts and introductions.

        Args:
            abstracts (list): List of abstracts.
            introductions (list): List of introductions.

        Returns:
            list: List of generated topics.
        """
        topics = []
        for abstract in tqdm(abstracts):
            topic = self.generate_topic(abstract)
            topics.append(topic)

        # Create a DataFrame with abstracts and topics
        data = {'Abstract': abstracts, 'Topic': topics}
        df = pd.DataFrame(data)
        
        # Write the DataFrame to a CSV file
        output_csv_file = 'generated_topics_15k_16k.csv'
        df.to_csv(output_csv_file, index=False)

        return topics
