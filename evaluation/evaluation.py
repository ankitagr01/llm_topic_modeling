# import unslot
import transformers
import torch
# from unsloth import FastLanguageModel
from typing import List
from transformers import BitsAndBytesConfig
import csv
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from datasets import load_metric
import evaluate
from rouge import Rouge 

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
subprocess.check_call([sys.executable, "-m", "pip", "install", "evaluate"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "sacrebleu"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "rouge"])


from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class Evaluation_model:
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


        # model_name = "meta-llama/Llama-3-13b-hf"
        # model_name = 'meta-llama/Meta-Llama-3-70B'
        # model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
        model_name = "ankitagr01/llama_3_8b_topic"
        # model_name = "meta-llama/Llama-2-70b-chat-hf"
        #model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
        # model_name = "meta-llama/Llama-2-13b-chat-hf"
        # model_name = "meta-llama/Llama-2-7b-chat-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
        # Load the model with int8 precision
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=False,  # Set to True for int4 # Set to True for int8
            device_map="auto"   # This automatically places the model on the GPU
        )
        # quantization_config=BitsAndBytesConfig(load_in_8bit=True)

        # TODO: fix examples
        # self.few_shot_examples = """
        # Abstract: This paper presents a new method for image segmentation using deep learning techniques.
        # Topic: Deep Learning Image Segmentation


        # Abstract: The study explores the impact of climate change on marine biodiversity.
        # Topic: Climate Change Marine Biodiversity

        # Abstract: An efficient algorithm for large-scale data processing in cloud environments is proposed.
        # Topic: Large-Scale Data Processing
        # """

        # System prompt describes information given to all conversations
        self.system_prompt = """
        <s>[INST] <<SYS>>
        You are a helpful, respectful and honest assistant for labeling topics.
        <</SYS>>
        """

        # fix examples
        self.few_shot_examples = """
        Abstract: Mesh-based simulations are central to modeling complex physical systems in many disciplines across science and engineering. Mesh representations support powerful numerical integration methods and their resolution can be adapted to strike favorable trade-offs between accuracy and efficiency. However, high-dimensional scientific simulations are very expensive to run, and solvers and parameters must often be tuned individually to each system studied. Here we introduce MeshGraphNets, a framework for learning mesh-based simulations using graph neural networks. Our model can be trained to pass messages on a mesh graph and to adapt the mesh discretization during forward simulation. Our results show it can accurately predict the dynamics of a wide range of physical systems, including aerodynamics, structural mechanics, and cloth. The model's adaptivity supports learning resolution-independent dynamics and can scale to more complex state spaces at test time. Our method is also highly efficient, running 1-2 orders of magnitude faster than the simulation on which it is trained. Our approach broadens the range of problems on which neural network simulators can operate and promises to improve the efficiency of complex, scientific modeling tasks.
        Topic: Mesh-Based Simulation with Graph Networks
        """

        self.main_prompt = """
        [INST]
        I want a topic based on the following text:
        {}

        Based on the text provided above, please propose a short topic appropriate for the text. Make sure you to only return the topic and nothing more.
        [/INST]
        """

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


    def generate_topic(self, abstract: str, fewshot: bool) -> str:
        """
        Generates a topic for a given abstract and introduction.

        Args:
            abstract (str): The abstract of the paper.
            introduction (str): The introduction of the paper.

        Returns:
            str: Generated topic.
        """
        # alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        # ### Instruction:
        # Generate a topic for the given abstract and introduction.

        # ### Input:
        # Abstract: {}

        # ### Response:
        # {}"""


        # prompt = f"""
        # ### System: You are an AI language model tasked with generating concise topics for academic paper abstracts.
        # ### Instruction: Given the abstract below, generate an appropriate topic.
        # ###
        # ### Abstract:
        # {abstract}
        # ###
        # ### Based on the text provided above, please propose a short topic appropriate for the text. Make sure you to only return the topic and nothing more.
        # ###
        # ### Topic:
        # """

        # prompt = f"""
        # <s>[INST] <<SYS>>
        # You are a helpful, respectful and honest assistant for labeling topics.
        # <</SYS>>

        # [INST]
        # See below example of how topic should be generated given an abstract:

        # Abstract: Mesh-based simulations are central to modeling complex physical systems in many disciplines across science and engineering. Mesh representations support powerful numerical integration methods and their resolution can be adapted to strike favorable trade-offs between accuracy and efficiency. However, high-dimensional scientific simulations are very expensive to run, and solvers and parameters must often be tuned individually to each system studied. Here we introduce MeshGraphNets, a framework for learning mesh-based simulations using graph neural networks. Our model can be trained to pass messages on a mesh graph and to adapt the mesh discretization during forward simulation. Our results show it can accurately predict the dynamics of a wide range of physical systems, including aerodynamics, structural mechanics, and cloth. The model's adaptivity supports learning resolution-independent dynamics and can scale to more complex state spaces at test time. Our method is also highly efficient, running 1-2 orders of magnitude faster than the simulation on which it is trained. Our approach broadens the range of problems on which neural network simulators can operate and promises to improve the efficiency of complex, scientific modeling tasks.
        # Topic: Mesh-Based Simulation with Graph Networks


        # [INST]
        # I want a short topic which appropriately describes the following text:
        # {abstract}

        # Based on the text provided above and the example, please propose a short topic appropriate for the text. Make sure you to only return the topic and nothing more.
        # [/INST]
        # """


        # prompt = f"Below is an instruction that describes a task. You are an AI language model tasked with generating concise topics for academic paper abstracts. Given the abstract below, generate an appropriate small topic. Give me response without explaination. \n\n### Instruction:\n Given the abstract below, generate an appropriate small topic. Dont provide any explanation, just return me the small topic. Make sure you to only return the topic and nothing more.\n\nInput:\n{abstract}\n\nMake sure you to only return the topic and nothing more.\n\n### Response:"


        # prompt = self.system_prompt + self.few_shot_examples + self.main_prompt.format(f"Abstract: {abstract}")
        # prompt = alpaca_prompt.format(f"Abstract: {abstract}")
        # response = self.model.generate(prompt, max_length=10, temperature=0.7)
        # return response.strip()

        prompt = f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful and respectful AI assistant for for labeling topics<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        
        Given the abstract below, generate an appropriate topic relevant to the abstract. Give me response without explaination.
        Abstract: {abstract} <|eot_id|>

        <|start_header_id|>assistant<|end_header_id|>
        """

        fewshot_prompt = f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful and respectful AI assistant for for labeling topics<|eot_id|>
        <|start_header_id|>user<|end_header_id|>

        Your task is to generate a topic for the input text, See below example of how topic should be generated given an abstract:
        Abstract: Mesh-based simulations are central to modeling complex physical systems in many disciplines across science and engineering. Mesh representations support powerful numerical integration methods and their resolution can be adapted to strike favorable trade-offs between accuracy and efficiency. However, high-dimensional scientific simulations are very expensive to run, and solvers and parameters must often be tuned individually to each system studied. Here we introduce MeshGraphNets, a framework for learning mesh-based simulations using graph neural networks. Our model can be trained to pass messages on a mesh graph and to adapt the mesh discretization during forward simulation. Our results show it can accurately predict the dynamics of a wide range of physical systems, including aerodynamics, structural mechanics, and cloth. The model's adaptivity supports learning resolution-independent dynamics and can scale to more complex state spaces at test time. Our method is also highly efficient, running 1-2 orders of magnitude faster than the simulation on which it is trained. Our approach broadens the range of problems on which neural network simulators can operate and promises to improve the efficiency of complex, scientific modeling tasks.
        Topic: Mesh-Based Simulation with Graph Networks

        Similarly, given the abstract below, generate an appropriate topic relevant to the abstract. Give me response without explaination.
        Abstract: {abstract} <|eot_id|>

        <|start_header_id|>assistant<|end_header_id|>
        """

        if fewshot:
            prompt = fewshot_prompt
    
        inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')
        
        generated_ids = self.model.generate(inputs.input_ids.cuda(), max_new_tokens=256, do_sample=True, top_p=0.95, temperature=0.8)
        topic = self.tokenizer.batch_decode(generated_ids)[0]
        print('topiccc:', topic)

        start_marker = '<|start_header_id|>assistant<|end_header_id|>'
        end_marker = '<|eot_id|>'
        topic = self.extract_topic(topic, start_marker, end_marker)
        print('topic:', topic)


        return topic

    def generate_topics(self, abstracts: List[str], fewshot: bool) -> List[str]:
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
            topic = self.generate_topic(abstract, fewshot)
            topics.append(topic)

        # # # Write abstracts and topics to a CSV file
        # output_csv_file = 'generated_topics_20k.csv'
        # with open(output_csv_file, mode='w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(['Abstract', 'Topic'])  # Writing the header
        #     writer.writerows(topics)  # Writing the data


        # # Create a DataFrame with abstracts and topics
        # data = {'Abstract': abstracts, 'Topic': topics}
        # df = pd.DataFrame(data)
        
        # # Write the DataFrame to a CSV file
        # output_csv_file = 'generated_topics_10k_11k.csv'
        # df.to_csv(output_csv_file, index=False)

        return topics



if __name__ == "__main__":
    # Load the arXiv dataset
    # Load dataset from the hub
    dataset = load_dataset("ankitagr01/dynamic_topic_modeling_arxiv_abstract_1k", split="test")

    print(f"Dataset size: {len(dataset)}")

    # model_name = "ankitagr01/abstract_topic_modeling_small100"
    # model_name = "ankitagr01/llama_3_8b_ft_topic"
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    fewshot = True

    topic_generator = Evaluation_model(model_name=model_name)
    pred_topics = topic_generator.generate_topics(dataset["Abstract"], fewshot=fewshot)

    # Create a DataFrame with abstracts and topics
    data = {'Abstract': dataset["Abstract"], 'Topic': dataset["Topic"], 'Prediction': pred_topics}
    df = pd.DataFrame(data)
    
    # Write the DataFrame to a CSV file
    output_csv_file = 'pred_topics_demo.csv'
    df.to_csv(output_csv_file, index=False)

    # bleu = load_metric("bleu")
    # rouge = load_metric("rouge")

    bleu = load_metric("sacrebleu")

    # Compute BLEU score
    bleu_score = bleu.compute(predictions=[pred.strip('"').strip("'").split() for pred in pred_topics],
                                    references=[[label.split()] for label in dataset["Topic"]], lowercase=True)

    rouge = Rouge()
    rouge_score = rouge.get_scores(pred_topics, dataset["Topic"], avg=True, lowercase=True)

    # Compute ROUGE score
    # rouge_score = rouge.compute(predictions=pred_topics, references=dataset["Topic"])

    print('bleu_score: ', bleu_score)
    print(f"BLEU score: {bleu_score['score']}")
    print(f"ROUGE score: {rouge_score}")
    #print(f"ROUGE score: {rouge_score['rouge-l']}")

    # "bleu": bleu_score["bleu"], "rouge": rouge_score["rouge-l"]}


    
    # Load the BLEU evaluation metric
    bleu = evaluate.load("bleu")

    # Compute the BLEU score
    results = bleu.compute(predictions=pred_topics, references=dataset["Topic"])

    # Print the results
    print(results)
