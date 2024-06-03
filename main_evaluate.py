import subprocess
import os
import evaluate
import pandas as pd
import wandb
from tqdm import tqdm
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from utils.argument_parser import ArgumentParser
from rouge import Rouge

# subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "accelerate"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "peft"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "bitsandbytes"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "trl"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "evaluate"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "sacrebleu"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "rouge"])


# Initialize wandb
def initialize_wandb():
    wandb.init(project="topic_modeling", name="llama_3_8b_ft")


# Load and prepare dataset
def load_and_prepare_dataset(config):
    print("Loading dataset...")
    dataset = load_dataset(config['data']['train_data_name'], split="train")
    print(f"Dataset size: {len(dataset)}")
    return dataset


# Data cleaning function
def clean_data(pred_topics: List[str], labels: List[str]) -> (List[str], List[str]):
    cleaned_pred_topics = [topic.strip('"').strip("'").lower() for topic in pred_topics]
    cleaned_labels = [label.strip('"').strip("'").lower() for label in labels]
    return cleaned_pred_topics, cleaned_labels


# Function to store results in CSV
def store_results_in_csv(data: dict, output_csv_file: str):
    df = pd.DataFrame(data)
    df.to_csv(output_csv_file, index=False)


# Evaluation function using BLEU and ROUGE
def evaluate_predictions(pred_topics: List[str], labels: List[str]):
    # bleu = evaluate.load("sacrebleu")
    # rouge = evaluate.load("rouge")
    bleu = load_metric("sacrebleu")
    rouge = Rouge()


    bleu_score = bleu.compute(predictions=[pred.split() for pred in pred_topics],
                              references=[[label.split()] for label in labels], lowercase=True)
    
    rouge_score = rouge.get_scores(pred_topics, labels, avg=True)
    # rouge_score = rouge.compute(predictions=pred_topics, references=labels)
    
    print('BLEU score: ', bleu_score)
    print(f"ROUGE score: {rouge_score}")

    wandb.log({"BLEU score": bleu_score['score'], "ROUGE score": rouge_score['rouge-l']['f']})
    
    return bleu_score, rouge_score


class EvaluationModel:
    """
    Class for eval model generating topics using eval model.

    Attributes:
        model (transformers.PreTrainedModel): The loaded model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
    """

    def __init__(self, model_name: str):
        """
        Initializes the EvaluationModel with the given model name.

        Args:
            model_name (str): The name of the model to be loaded.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=False,  # Set to True for int4
            device_map="auto"    # Automatically places the model on the GPU
        )

    def extract_topic(self, output_text: str, start_marker: str, end_marker: str) -> str:
        """
        Extracts the topic from the generated output text.

        Args:
            output_text (str): The generated text containing the topic.
            start_marker (str): The marker indicating the start of the topic.
            end_marker (str): The marker indicating the end of the topic.

        Returns:
            str: The extracted topic.
        """
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
        Generates a topic for a given abstract.

        Args:
            abstract (str): The abstract of the paper.
            fewshot (bool): Whether to use few-shot examples in the prompt.

        Returns:
            str: Generated topic.
        """
        prompt = f"""
        system

        You are a helpful and respectful AI assistant for for labeling topics
        user
        
        Given the abstract below, generate an appropriate topic relevant to the abstract. Give me response without explaination.
        Abstract: {abstract} 

        assistant
        """

        fewshot_prompt = f"""
        system

        You are a helpful and respectful AI assistant for for labeling topics
        user

        Your task is to generate a topic for the input text, See below example of how topic should be generated given an abstract:
        Abstract: Mesh-based simulations are central to modeling complex physical systems in many disciplines across science and engineering. Mesh representations support powerful numerical integration methods and their resolution can be adapted to strike favorable trade-offs between accuracy and efficiency. However, high-dimensional scientific simulations are very expensive to run, and solvers and parameters must often be tuned individually to each system studied. Here we introduce MeshGraphNets, a framework for learning mesh-based simulations using graph neural networks. Our model can be trained to pass messages on a mesh graph and to adapt the mesh discretization during forward simulation. Our results show it can accurately predict the dynamics of a wide range of physical systems, including aerodynamics, structural mechanics, and cloth. The model's adaptivity supports learning resolution-independent dynamics and can scale to more complex state spaces at test time. Our method is also highly efficient, running 1-2 orders of magnitude faster than the simulation on which it is trained. Our approach broadens the range of problems on which neural network simulators can operate and promises to improve the efficiency of complex, scientific modeling tasks.
        Topic: Mesh-Based Simulation with Graph Networks

        Similarly, given the abstract below, generate an appropriate topic relevant to the abstract. Give me response without explaination.
        Abstract: {abstract} 

        assistant
        """

        if fewshot:
            prompt = fewshot_prompt
    
        inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')
        
        generated_ids = self.model.generate(inputs.input_ids.cuda(), max_new_tokens=256, do_sample=True, top_p=0.95, temperature=0.8)
        topic = self.tokenizer.batch_decode(generated_ids)[0]
        print('Generated topic:', topic)

        start_marker = 'assistant'
        end_marker = ''
        topic = self.extract_topic(topic, start_marker, end_marker)
        print('Extracted topic:', topic)

        return topic

    def generate_topics(self, abstracts: List[str], fewshot: bool) -> List[str]:
        """
        Generates topics for a list of abstracts.

        Args:
            abstracts (List[str]): List of abstracts.
            fewshot (bool): Whether to use few-shot examples in the prompts.

        Returns:
            List[str]: List of generated topics.
        """
        topics = []
        for abstract in tqdm(abstracts):
            topic = self.generate_topic(abstract, fewshot)
            topics.append(topic)

        return topics


def main(config):
    # Initialize wandb
    initialize_wandb()

    # Load dataset from the hub
    dataset = load_and_prepare_dataset(config)

    # Define model name and fewshot flag
    model_name = config['eval']['eval_model_name']
    # model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    fewshot = config['eval']['fewshot']

    # Create an instance of EvaluationModel
    topic_generator = EvaluationModel(model_name=model_name)
    pred_topics = topic_generator.generate_topics(dataset["Abstract"], fewshot=fewshot)

    # Clean up the predicted and labels topics
    pred_topics, labels = clean_data(pred_topics, dataset["Topic"])

    # Create a DataFrame with abstracts and topics
    # Store results in a CSV file
    data = {'Abstract': dataset["Abstract"], 'Topic': dataset["Topic"], 'Prediction': pred_topics}
    store_results_in_csv(data, 'pred_topics_demo.csv')
    
    # Evaluate predictions
    bleu_score, rouge_score = evaluate_predictions(pred_topics, labels)


if __name__ == "__main__":
    
    # Parse arguments and load config
    arg_parser = ArgumentParser()
    args = arg_parser.parse_args()
    config = arg_parser.load_config(args)

    main(config)
