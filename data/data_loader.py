from datasets import load_dataset
import pandas as pd
import os
from typing import Tuple, List

class DataLoader:
    """
    Class for loading the arXiv dataset and handling saved topics.

    Attributes:
        dataset: The loaded dataset from the Hugging Face library.
        topics_file: Path to the CSV file storing generated topics.
    """

    def __init__(self, dataset_name: str, topics_file: str):
        self.dataset = load_dataset(dataset_name)["train"]
        # load only the first 10 samples for testing  # temp AA
        self.dataset = self.dataset.select(list(range(15000, 16000)))  # temp AA
        self.topics_file = topics_file

    def get_data(self) -> Tuple[List[str], List[str]]:
        """
        Extracts abstracts, titles, and introductions from the dataset.

        Returns:
            abstracts (list): List of abstracts.
            titles (list): List of titles.
            introductions (list): List of introductions (first sections).
        """
        abstracts = self.dataset["abstract"]
        titles = self.dataset["title"]
        # introductions = self.dataset["abstract"]   # temp AA
        # introductions = [section.split('\n')[0] for section in self.dataset["sections"]]  # Assuming the first section is the introduction  #temp AA
        return abstracts, titles

    def load_topics(self) -> List[str]:
        """
        Loads generated topics from a CSV file.

        Returns:
            list: List of topics.
        """
        if os.path.exists(self.topics_file):
            df = pd.read_csv(self.topics_file)
            return df['topic'].tolist()
        return []

    def save_topics(self, topics: List[str]):
        """
        Saves generated topics to a CSV file.

        Args:
            topics (list): List of topics to save.
        """
        df = pd.DataFrame({'topic': topics})
        df.to_csv(self.topics_file, index=False)
