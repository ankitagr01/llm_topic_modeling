
# LLaMA-3 Topic Modeling

This project fine-tunes the LLaMA-3 8B model to generate topics dynamically from abstracts and introductions of arXiv papers. The fine-tuning process is modular, with separate components for data loading, preprocessing, training, and evaluation.

## Project Structure

llama_topic_modeling/
│
├── data/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_processor.py
│
├── models/
│   ├── __init__.py
│   ├── model.py
│
├── training/
│   ├── __init__.py
│   ├── trainer.py
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py
│
├── main.py
├── requirements.txt
└── README.md




## Installation

1. Clone the repository.
2. Install the required packages:

```
pip install -r requirements.txt
```

## Usage
Run the main.py script to fine-tune the LLaMA-3 8B model on the arXiv dataset and generate topics dynamically.

```
python main.py
```

## Evaluation
The model is evaluated using BLEU and ROUGE scores to compare the performance of the fine-tuned and pre-trained models.

This codebase should be structured, modular, and easy to manage. You can extend or modify any component as needed. Make sure to have the necessary API keys and access to the UnsloT library for this to work correctly.





# SUMM AI NLP Challenge - Topic Modeling with LLM 🚀

## Objective
Fine-tune an open-source Large Language Model (LLM) such as Llama3 to perform topic modeling on a collection of text paragraphs. Note that the method you develop should generate a meaningful topic for a given text **dynamically**, i.e. the outputs should not be limited to a specific class of topic names.

## Dataset
There is no dataset provided. Find a relevant dataset related to this task.

## Submission
Please clone this repository and upload it to a new private repo.
Implement a well-organized codebase along with a README documenting the setup, key findings, and challenges.  
Add me (@flowni) to the repo for submitting it.
You have one week to complete the assignment. ⏰

## Evaluation Criteria
The focus of this assignment is on the development process and the ability to fine-tune an LLM. Don't worry too much about the performance of the model as long as you know how to improve it.
The following criteria will be used to evaluate the assignment:

1. **Code Quality:**
   - Readability and organization of the code. 📦

2. **Documentation:**
   - Clarity and completeness of the documentation. 📚

3. **Improvement Ideas:**
   - Ideas how to evaluate and improve the model further. 🚀

4. **(Bonus for Docker environment)** 🐳

## Contact
Feel free to reach out to me (Nicholas) for any questions related to the assignment. 📧

Have fun! 😊