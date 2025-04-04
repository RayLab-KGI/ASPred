import os
import re
import random
import numpy as np
import pandas as pd
import json
import transformers

from transformers import TrainingArguments, Trainer
from datasets import Dataset
from models.pt5_model import PT5_classification_model
from models.save_load_model import save_model
from data_processing.preprocess import preprocess_sequences
from evaluate import load
from utils import set_seeds, create_dataset

# Load DeepSpeed configuration
CONFIG_PATH = os.path.join("scr", "configs", "deepspeed.json")
with open(CONFIG_PATH, "r") as file:
    ds_config = json.load(file)

# Main training function
def train_per_protein(
        train_df,         # Training data
        valid_df,         # Validation data
        num_labels=1,     # 1 for regression, >1 for classification
        batch=4,          # Training batch size
        accum=2,          # Gradient accumulation
        val_batch=16,     # Validation batch size
        epochs=10,        # Number of training epochs
        lr=3e-4,          # Learning rate
        seed=42,          # Random seed
        deepspeed=True,   # Use DeepSpeed if GPU is large enough
        mixed=False,      # Enable mixed precision training
        gpu=1):           # GPU selection

    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu - 1)

    # Set all random seeds
    set_seeds(seed)

    # Load model and tokenizer
    model, tokenizer = PT5_classification_model(num_labels=num_labels)

    # Preprocess inputs
    train_df = preprocess_sequences(train_df, sequence_col="sequence")
    valid_df = preprocess_sequences(valid_df, sequence_col="sequence")

    # Create datasets
    train_set = create_dataset(tokenizer, list(train_df['sequence']), list(train_df['label']))
    valid_set = create_dataset(tokenizer, list(valid_df['sequence']), list(valid_df['label']))

    # Define training arguments
    args = TrainingArguments(
        output_dir="./",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        learning_rate=lr,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=val_batch,
        gradient_accumulation_steps=accum,
        num_train_epochs=epochs,
        seed=seed,
        deepspeed=ds_config if deepspeed else None,
        fp16=mixed,
    )

    # Define compute metrics function for validation data
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if num_labels > 1:  # For classification
            metric = load("accuracy")
            predictions = np.argmax(predictions, axis=1)
        else:  # For regression
            metric = load("spearmanr")
        return metric.compute(predictions=predictions, references=labels)

    # Initialize trainer
    trainer = Trainer(
        model,
        args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    save_model(model, "./PT5_finetuned.pth")
    print("Model saved to ./PT5_finetuned.pth")

    return tokenizer, model, trainer.state.log_history

####
# CLI or main entry point
if __name__ == "__main__":
    # Example dataset paths
    TRAIN_PATH = "data/processed/training_set.csv"
    VALID_PATH = "data/processed/valid_set.csv"

    # Load data
    train_data = pd.read_csv(TRAIN_PATH)
    valid_data = pd.read_csv(VALID_PATH)

    # Run training
    tokenizer, model, history = train_per_protein(
        train_df=train_data,
        valid_df=valid_data,
        num_labels=2,
        lr=2e-4,
        batch=1,
        accum=8,
        epochs=8,
        seed=42
    )
