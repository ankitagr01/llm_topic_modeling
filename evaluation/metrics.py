from datasets import load_metric
from typing import Tuple

class Evaluator:
    """
    Class for evaluating the model's performance.

    Attributes:
        tokenizer: The tokenizer used for decoding model outputs.
        bleu: The BLEU metric.
        rouge: The ROUGE metric.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bleu = load_metric("bleu")
        self.rouge = load_metric("rouge")

    def compute_metrics(self, eval_preds: Tuple) -> dict:
        """
        Computes BLEU and ROUGE metrics for model predictions.

        Args:
            eval_preds (Tuple): The model predictions and labels.

        Returns:
            dict: The computed metrics.
        """
        preds, labels = eval_preds
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Compute BLEU score
        bleu_score = self.bleu.compute(predictions=[pred.split() for pred in decoded_preds],
                                       references=[[label.split()] for label in decoded_labels])

        # Compute ROUGE score
        rouge_score = self.rouge.compute(predictions=decoded_preds, references=decoded_labels)

        return {"bleu": bleu_score["bleu"], "rouge": rouge_score["rouge-l"]}
