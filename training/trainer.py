import torch
from transformers import TrainingArguments, Trainer
from typing import Callable
import wandb

class TopicDataset(torch.utils.data.Dataset):
    """
    Custom dataset class for handling tokenized data.

    Attributes:
        encodings: The tokenized input data.
        labels: The tokenized target data.
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int):
        """
        Returns a single data point from the dataset.

        Args:
            idx (int): Index of the data point.

        Returns:
            dict: A dictionary containing input_ids, attention_mask, and labels.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.labels)

class ModelTrainer:
    """
    Class for training and evaluating the model.

    Attributes:
        model: The model to be trained.
        tokenizer: The tokenizer used for encoding the data.
        train_dataset: The training dataset.
        val_dataset: The validation dataset.
    """

    def __init__(self, model, tokenizer, train_dataset, val_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train(self, training_args: dict):
        """
        Trains the model on the training dataset.

        Args:
            training_args (dict): The training arguments for the Trainer.
        """
        wandb.init(project="llama-topic-modeling", config=training_args)
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(**training_args),
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
        )
        trainer.train()

    def evaluate(self, dataset, compute_metrics: Callable):
        """
        Evaluates the model on the given dataset.

        Args:
            dataset: The dataset to evaluate on.
            compute_metrics (Callable): A function to compute evaluation metrics.

        Returns:
            dict: The evaluation results.
        """
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir="./results",
                per_device_eval_batch_size=4,
            ),
        )
        return trainer.evaluate(eval_dataset=dataset, compute_metrics=compute_metrics)
