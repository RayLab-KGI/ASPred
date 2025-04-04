import random
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset
from transformers import set_seed
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

def create_dataset(tokenizer,seqs,labels):
    tokenized = tokenizer(seqs, max_length=256, padding=True, truncation=True)
    dataset = Dataset.from_dict(tokenized)
    dataset = dataset.add_column("labels", labels)

    return dataset

def set_seeds(s):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    set_seed(s)

def evaluate(true_labels, predicted_probs, predicted_labels):
    roc_auc = roc_auc_score(true_labels, predicted_probs)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    class_report = classification_report(true_labels, predicted_labels)
    return roc_auc, conf_matrix, class_report
