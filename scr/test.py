import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from models.save_load_model import load_model
from data_processing.preprocess import preprocess_sequences
from utils import create_dataset, evaluate
from torch.utils.data import DataLoader

def predict(model_path, test_df, batch_size=16):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer, model = load_model(model_path, num_labels=2, mixed=False)
    model.to(device)
    data = pd.read_csv(test_df)

    data_df = preprocess_sequences(data, sequence_col="sequence")
    test_set = create_dataset(tokenizer, list(data_df['sequence']), list(data_df['label']))
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    model.eval()

    predicted_probs, predicted_labels = [], []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits = model(input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            labels = (probs >= 0.5).long()
            predicted_probs.extend(probs.tolist())
            predicted_labels.extend(labels.tolist())
    return predicted_probs, predicted_labels

def evaluate_model(model_path, data_path, num_labels=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the fine-tuned model and tokenizer

    tokenizer, model = load_model(model_path, num_labels=num_labels, mixed=False)
    model.to(device)
    model.eval()

    # Load and preprocess the dataset
    data = pd.read_csv(data_path)
    data = preprocess_sequences(data, sequence_col="sequence")

    # Run predictions
    predicted_probs, predicted_labels = predict(model, tokenizer, data)

    # Calculate metrics
    true_labels = np.array(data['label'])
    roc_auc, conf_matrix, class_report = evaluate(true_labels, np.array(predicted_probs), np.array(predicted_labels))

    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)
    print("\nROC AUC Score:", roc_auc)

    return {"roc_auc": roc_auc, "conf_matrix": conf_matrix, "class_report": class_report}
