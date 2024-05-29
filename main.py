import os
from data.data_loader import DataLoader
from data.data_processor import DataProcessor
from data.topic_generator import TopicGenerator
from models.model import TopicModel
from training.trainer import TopicDataset, ModelTrainer
from evaluation.metrics import Evaluator
from utils.argument_parser import ArgumentParser
from huggingface_hub import notebook_login, login
import wandb

def main():
    # Parse arguments and load config
    arg_parser = ArgumentParser()
    args = arg_parser.parse_args()
    config = arg_parser.load_config(args)

    # !huggingface-cli login
    # TODO: token should be stored in a secure way
    login(token='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')

    # Load and preprocess data
    data_loader = DataLoader(dataset_name=config['data']['dataset_name'], topics_file=config['data']['generated_topics_file'])
    abstracts, titles, introductions = data_loader.get_data()

    # Generate or load topics
    if os.path.exists(config['data']['generated_topics_file']):
        topics = data_loader.load_topics()
    else:
        topic_generator = TopicGenerator(model_name=config['model']['teacher_model_name'])
        topics = topic_generator.generate_topics(abstracts, introductions)
        data_loader.save_topics(topics)

    # Preprocess data
    data_processor = DataProcessor(model_name="meta-llama/Llama-2-7b-hf", config=config)
    inputs, targets = data_processor.preprocess_data(abstracts, introductions, topics)
    train_inputs, val_inputs, test_inputs, train_targets, val_targets, test_targets = data_processor.split_data(inputs, targets, config['data']['test_size'], config['data']['val_size'])

    # Tokenize data
    train_encodings, train_labels = data_processor.tokenize_data(train_inputs, train_targets)
    val_encodings, val_labels = data_processor.tokenize_data(val_inputs, val_targets)
    test_encodings, test_labels = data_processor.tokenize_data(test_inputs, test_targets)

    # Create datasets
    train_dataset = TopicDataset(train_encodings, train_labels)
    val_dataset = TopicDataset(val_encodings, val_labels)
    test_dataset = TopicDataset(test_encodings, test_labels)

    # Initialize and train model
    topic_model = TopicModel(model_name=config['model']['student_model_name'], config=config)
    # add lora weights
    # topic_model.add_lora_weights(config=config)
    trainer = ModelTrainer(topic_model.model, data_processor.tokenizer, train_dataset, val_dataset)
    trainer.train(training_args=config['training'])

    # Evaluate model
    evaluator = Evaluator(data_processor.tokenizer)

    # Evaluate fine-tuned model
    eval_results_finetuned = trainer.evaluate(dataset=val_dataset, compute_metrics=evaluator.compute_metrics)
    print(f"Fine-tuned model evaluation: {eval_results_finetuned}")

    # Save the fine-tuned model
    topic_model.save_model("lora_model")
    data_processor.tokenizer.save_pretrained("lora_model")
    # topic_model.model.push_to_hub("your_name/lora_model", token="YOUR_HF_TOKEN")
    # data_processor.tokenizer.push_to_hub("your_name/lora_model", token="YOUR_HF_TOKEN")

    # Evaluate pre-trained (not fine-tuned) model
    topic_model.load_model(config['model']['student_model_name'])  # Reset to pre-trained state
    eval_results_pretrained = trainer.evaluate(dataset=val_dataset, compute_metrics=evaluator.compute_metrics)
    print(f"Pre-trained model evaluation: {eval_results_pretrained}")



if __name__ == "__main__":
    main()
