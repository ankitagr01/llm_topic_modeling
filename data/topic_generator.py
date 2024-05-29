# import unslot
import transformers
import torch
# from unsloth import FastLanguageModel
from typing import List

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


        model_name = "meta-llama/Llama-2-13b-hf"
        # model_name = "meta-llama/Llama-2-13b-chat-hf"
        # model_name = "meta-llama/Llama-2-7b-chat-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

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



    def generate_topic(self, abstract: str, introduction: str) -> str:
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


        alpaca_prompt = f"""
        ### System: You are an AI language model tasked with generating concise topics for academic paper abstracts.
        ### Instruction: Given the abstract below, generate an appropriate topic.
        ###
        ### Abstract:
        {abstract}
        ###
        ### Topic:
        """

        # prompt = self.system_prompt + self.few_shot_examples + self.main_prompt.format(f"Abstract: {abstract}")
        prompt = alpaca_prompt.format(f"Abstract: {abstract}")
        # response = self.model.generate(prompt, max_length=10, temperature=0.7)
        # return response.strip()
    
        inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')
        outputs = self.model.generate(input_ids=inputs["input_ids"].to("cuda"), attention_mask=inputs["attention_mask"], max_new_tokens=50, pad_token_id=self.tokenizer.eos_token_id)
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print('output_text', output_text)
        print('output_text.strip: ', output_text.strip())

        # Extract the topic from the output
        topic_marker = "### Topic:"
        topic_start = output_text.find(topic_marker) + len(topic_marker)
        topic = output_text[topic_start:].strip().split('\n')[0]

        #return output_text.strip()
        return topic

    def generate_topics(self, abstracts: List[str], introductions: List[str]) -> List[str]:
        """
        Generates topics for a list of abstracts and introductions.

        Args:
            abstracts (list): List of abstracts.
            introductions (list): List of introductions.

        Returns:
            list: List of generated topics.
        """
        topics = []
        for abstract, introduction in zip(abstracts, introductions):
            topic = self.generate_topic(abstract, introduction)
            topics.append(topic)
        return topics
