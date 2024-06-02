from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from typing import List, Tuple

class DataProcessor:
    """
    Class for preprocessing and tokenizing the data. 

    Attributes:
        tokenizer: The tokenizer used for encoding the data.
    """

    def __init__(self, model_name: str, config: dict):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_length = config["model"]["max_seq_length"]

    def preprocess_data(self, abstracts: List[str], introductions: List[str], topics: List[str]) -> Tuple[List[str], List[str]]:
        """
        Preprocesses the data by combining abstracts and introductions, and pairing them with topics.

        Args:
            abstracts (list): List of abstracts.
            introductions (list): List of introductions.
            topics (list): List of topics.

        Returns:
            inputs (list): List of combined abstracts and introductions.
            targets (list): List of topics.
        """
    
        # inputs = [f"{abstract} {introduction}" for abstract, introduction in zip(abstracts, introductions)]
        inputs = abstracts
        targets = topics
        return inputs, targets

    def tokenize_data(self, inputs: List[str], targets: List[str]) -> Tuple[dict, List[int]]:
        """
        Tokenizes the inputs and targets.

        Args:
            inputs (list): List of input texts.
            targets (list): List of target texts.

        Returns:
            encodings (dict): Tokenized input data.
            labels (list): Tokenized target data.
        """
        # tokenizer.pad_token 
        self.tokenizer.pad_token = self.tokenizer.eos_token

        encodings = self.tokenizer(inputs, truncation=True, padding=True, max_length=self.max_seq_length)
        labels = self.tokenizer(targets, truncation=True, padding=True, max_length=self.max_seq_length)["input_ids"]
        return encodings, labels

    def split_data(self, inputs: List[str], targets: List[str], test_size: float, val_size: float) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
        """
        Splits the data into training, validation, and test sets.

        Args:
            inputs (list): List of input texts.
            targets (list): List of target texts.
            test_size (float): Proportion of the data to include in the test split.
            val_size (float): Proportion of the data to include in the validation split (relative to training set).

        Returns:
            train_inputs (list): List of training inputs.
            val_inputs (list): List of validation inputs.
            test_inputs (list): List of test inputs.
            train_targets (list): List of training targets.
            val_targets (list): List of validation targets.
            test_targets (list): List of test targets.
        """
        train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs, targets, test_size=test_size)
        train_inputs, val_inputs, train_targets, val_targets = train_test_split(train_inputs, train_targets, test_size=val_size)
        return train_inputs, val_inputs, test_inputs, train_targets, val_targets, test_targets
