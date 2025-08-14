# Fine-Tuning Mistral 7B with LoRA PEFT

This project provides a comprehensive pipeline for fine-tuning the Mistral 7B large language model on a custom domain dataset. It leverages Hugging Face tools, including Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA), to optimize the training process for consumer-grade GPUs. The primary goal is to enhance the model's performance on specialized tasks, reduce domain-specific errors, and improve overall response quality for a given use case.

## ⬥ Key Features

- **Efficient Fine-Tuning**: Utilises LoRA and 4-bit quantization to significantly cut down VRAM usage and training duration.
- **Hugging Face Integration**: Built entirely on the Hugging Face ecosystem (`transformers`, `peft`, `datasets`, `trl`).
- **Custom Data Pipeline**: Easily adaptable to any custom, domain-specific dataset in a simple JSON Lines format.
- **Performance Validation**: Includes scripts for both qualitative and quantitative evaluation to validate the fine-tuned model's improved capabilities.
- **Reproducibility**: Defined dependencies and clear scripts ensure the workflow is straightforward to replicate.

## ⬥ Project Structure

```text
fine-tuning-Mistral-7B/
├── data/ # Contains custom dataset and format info
├── scripts/ # Main Python scripts for tuning and evaluation
├── models/ # Directory for saving trained LoRA adapters
├── .gitignore # Specifies files for Git to ignore
├── README.md # Project overview and instructions
└── requirements.txt # Python dependencies
```

## ⬥ Setup and Installation

Follow these steps to set up the project environment.

**1. Clone the Repository**
git clone https://github.com/Gamikant/fine-tuning-Mistral-7B.git
cd fine-tuning-Mistral-7B

**2. Create a Python Virtual Environment**
- It is highly recommended to use a virtual environment to manage dependencies.
- You can directly `Ctrl + Shift + P` -> Search for `Python: Create Environment` -> `Venv` -> Choose Python Version -> the `requirements.txt` file.
- This will create a virtual enviroment and directly install all the dependencies mentioned in `requirements.txt`

(OR)

```bash
python -m venv venv
source venv/bin/activate # On Windows, use venv\Scripts\activate
```

**3. Install Dependencies**
Install all the necessary libraries from `requirements.txt`.
```bash
pip install -r requirements.txt
```

**4. Prepare Your Dataset**
Place your custom dataset in the `data/` directory. Ensure it is named `custom_dataset.jsonl` and follows the format specified in `data/README.md`.

## ⬥ Usage

The fine-tuning and evaluation processes are handled by two main scripts.

**1. Fine-Tuning the Model**
Run the `fine_tune.py` script to start the training process. This will load the base Mistral 7B model, apply LoRA, and fine-tune it on your custom dataset.
cd scripts
python fine_tune.py

The script will save the resulting LoRA adapters to the `models/mistral-7b-custom-tuned/` directory.

**2. Evaluating the Model**
After fine-tuning, run the `evaluate.py` script to assess the model's performance.
python evaluate.py

This script provides a qualitative comparison between the base and fine-tuned models by generating responses to sample prompts. It also includes a placeholder for quantitative metrics, which should be tailored to your specific use case.