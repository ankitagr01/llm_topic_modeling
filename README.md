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
├── main_generate_labels.py (Main code for generating labels for training)
├── main_finetune_model.py (Main code for finetuning)
├── main_evaluate.py (Main code for evaluating models)
├── requirements.txt
├── run.sh
├── Topic_modeling_documentation_ankit.pdf
├── Dockerfile
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

3. **Generate labels**: Generate label from teacher model 
    ```bash
    python main_generate_labels.py
    ```

4. **Finetune model**: Finetune the student model based on data generated using teacher model
    ```bash
    python main_finetune_model.py
    ```

5. **Evaluate models**: Evaluate the finetuned model for the test set using BLEU and ROUGE scores. 

  ```bash
  python main_evaliuate.py
  ```  


This will run the teacher model, save the labels in csv file and also upload to huggingface hub datasets (Step-3).   
Next in step-4, we start finetune the student model and save the model locally and also push model to huggingface hub. 

BLUE and ROUGE scores for the test set is computed.



6. ** Docker**: To use via docker, edit the Dockerfile as per the run requirement.
    ```bash
    docker build -t fine_tuning_image .
     
    docker run --gpus all -e HF_TOKEN=hugging_face_token -v /host/path:/container/path --rm -it fine_tuning_image
    ```


---

## 4. Results
The table below presents the performance (BLEU and ROUGE scores) of our models:

Model                              |    BLUE    |  ROUGE  |
-----------------------------------|------------|---------|
Llama-3-8B-Instruct (Pre-trained)  |  42.11     |  51.58  |
Llama-3-8B-Instruct (Few-shot)     |  39.83     |  53.91  |
Llama-3-8B-Instruct (Fine-tuned)   |  44.44     |  53.13  |

---
## Resource Analysis (TBD)

### GPU Used
- **A100-80GB**
- **H100-80GB**

---
### GPU Utilization

- **Teacher Model (Llama-3-70B-Instruct)**
  - **GPU Memory Required**: 42.8 GB
  - **Inference Time**: 1.2 s/sample

- **Student Model (Llama-3-8B-Instruct)**
  - **Fine-tuning**:
    - **GPU Memory Required**: 33.8 GB (batch size: 4)
    - **Total Training Time**: 90 mins (15k samples)
  - **Inference**:
    - **GPU Memory Required**: 32.7 GB
    - **Inference Time**: 0.4 s/sample

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