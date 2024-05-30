# Topic Modeling with LLMs

This project focuses on topic modeling for abstracts of papers from the CShorten/ML-ArXiv-Papers dataset using a knowledge distillation approach. We use a large teacher model (Llama-3-70B) to generate labels and a smaller student model (Llama-3-8B) fine-tuned on this data for inference.

---

## Approach

We employ a knowledge distillation approach to generate high-quality training data and use a smaller fine-tuned model for inference, which performs similarly to the large teacher model.

---

## Models

- **Teacher Model**: Llama-3-70B-Instruct
- **Student Model**: Llama-3-8B-Instruct

---
## Evaluation

The performance of the models is evaluated using BLEU-3 and ROUGE scores.

---
## Project Structure
```
lama_topic_modeling/  
├── ...    
├── config/  
|   ├── init.py  
|   └── config.yaml
|    
├── data/  
|   ├── init.py  
|   ├── data_loader.py  
|   ├── topic_generator.py  
|   └── data_processor.py 
|   
├── models/  
|   ├── init.py  
|   └── model.py 
|   
├── results/  
|   └── llama_finetuned.pth 
|   
├── training/  
|   ├── init.py  
|   └── trainer.py 
|   
├── evaluation/  
|   ├── init.py  
|   └── metrics.py 
|   
├── utils/  
|   ├── init.py  
|   └── argument_parser.py 
|  
├── main.py
├── requirements.txt
├── run.sh
├── Topic_modeling_documentation_ankit.pdf
└── README.md
```

---
## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/llama_topic_modeling.git
    cd llama_topic_modeling
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
---
## Usage

1. **HuggingFace token**: Put your huggingface token in /main.py which will be used to download the llama-3 models.

2. **Update config**; Update different training parameters if needed at config.yaml.

3. **Train the Model**: Generate label from teacher model and use it to train the student model by the following scripts: 
    ```bash
    python main.py
    ```

This will run the teacher model, save the labels in csv file (so that it can be directly used later without running the teacher model again). Next it will finetune the student model and generate inference for the test set. 
BLUE and ROUGE scores for the test set is computed.


---

## 4. Results
The table below presents the performance (BLEU and ROUGE scores) of our models:

Model                              |    BLUE    |  ROUGE  |
-----------------------------------|------------|---------|
Llama-3-8B-Instruct (Pre-trained)  |  xx.xx     |  xx.xx  |
Llama-3-8B-Instruct (Few-shot)     |  xx.xx     |  xx.xx  |
Llama-3-8B-Instruct (Fine-tuned)   |  xx.xx     |  xx.xx  |

---
## Resource Analysis (TBD)

### GPU Used
- **A100-80GB**
- **H100-80GB**

---
### GPU Utilization

- **Teacher Model (Llama-3-70B-Instruct)**
  - **GPU Memory Required**: XX GB
  - **Inference Time**: XX ms/sample

- **Student Model (Llama-3-8B-Instruct)**
  - **Fine-tuning**:
    - **GPU Memory Required**: XX GB
    - **Total Training Time**: XX hours
  - **Inference**:
    - **GPU Memory Required**: XX GB
    - **Inference Time**: XX ms/sample

---

## Improvements

- Hyperparameter tuning
- Prompt tuning (zero-shot/few-shot)
- Incorporating human evaluation and feedback
- Expanding training data to more domains for better generalization
- Enhancing documentation using Doxygen
- Scalability improvements using multiprocessing or multiple GPUs

---

## Tech Stack

- **Programming Language**: Python
- **Deep Learning Framework**: PyTorch
- **Tools**:
  - Gafarna
  - WandB
  - Slurm
  - Huggingface
- **Development Environment**: VSCode with SSH connection

---
#### Author: Ankit Agrawal (ankitagr.nitb@gmail.com)