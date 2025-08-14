import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer

# --- Model & Tokenizer Configuration ---
base_model_name = "mistralai/Mistral-7B-v0.1"
new_model_name = "mistral-7b-custom-tuned"

# --- Quantization Configuration (for memory efficiency) ---
# Enables 4-bit quantization to significantly reduce VRAM usage.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

# --- LoRA Configuration (for parameter-efficient fine-tuning) ---
# LoRA (Low-Rank Adaptation) reduces the number of trainable parameters,
# which accelerates training and lowers memory overhead without sacrificing performance.
lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# --- Load Model & Tokenizer ---
print("Loading base model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto", # Automatically maps model layers to available devices (GPU/CPU)
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- Load Custom Dataset ---
print("Loading custom dataset...")
# Assumes the dataset is in the specified format in the data/ directory.
dataset = load_dataset("json", data_files="../data/custom_dataset.jsonl", split="train")

# --- Set up PEFT Model ---
print("Applying LoRA PEFT configuration...")
model = get_peft_model(model, lora_config)

# --- Training Arguments ---
# These arguments define the training process, including hyperparameters,
# logging, and saving strategies.
training_args = TrainingArguments(
    output_dir="../models/",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)

# --- Initialize SFTTrainer ---
# The SFTTrainer from TRL (Transformer Reinforcement Learning) simplifies supervised fine-tuning.
print("Initializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
)

# --- Start Fine-Tuning ---
print("Starting fine-tuning process...")
trainer.train()

# --- Save the Fine-Tuned Model ---
# This saves the trained LoRA adapters, not the full model, which is highly efficient.
print(f"Saving fine-tuned model to ../models/{new_model_name}")
trainer.model.save_pretrained(os.path.join("../models/", new_model_name))

print("Fine-tuning complete.")

