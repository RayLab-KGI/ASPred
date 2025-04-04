import os
import pandas as pd
from train import train_per_protein
from test import evaluate_model

def main():
    # Define dataset paths
    TRAIN_PATH = "./scr/datasets/training_set.csv"
    TEST_PATH = "./scr/datasets/test_set.csv"
    #ALL_DATA_PATH = "scr/datasets/training_and_test.csv"
    UNSEEN_VALID_PATH = "./scr/datasets/valid_set.csv"

    # Step 1: Train the model on training_set.csv
    print("Step 1: Training model on training_set.csv...")
    train_data = pd.read_csv(TRAIN_PATH)
    valid_data = pd.read_csv(TEST_PATH)
    tokenizer, model, history = train_per_protein(
        train_df=train_data,
	valid_df=valid_data,
        num_labels=2,  # Binary classification
        lr=2e-4,
        batch=4,
        accum=2,
        epochs=8,
        seed=42
    )

    # Step 2: Evaluate the model on test_set.csv
    print("\nStep 2: Evaluating model on test_set.csv...")
    test_metrics = evaluate_model(
        model_path="./PT5_finetuned.pth",
        tokenizer_path=None,  # Not needed as the tokenizer is bundled with the model
        data_path=TEST_PATH,
        num_labels=2
    )
    print("\nTest Set Metrics:")
    print(test_metrics)

    # Check metrics and decide whether to proceed
    # Simulating that the metrics are satisfactory and we proceed
    #print("\nMetrics are satisfactory. Proceeding to retrain the model with all data.")

    # Step 3: Retrain the model on valid_set.csv (all data)
    #print("\nStep 3: Retraining model on valid_set.csv...")
    #all_data = pd.read_csv(ALL_DATA_PATH)
    #tokenizer, model, history = train_per_protein(
    #    train_df=all_data,
    #    num_labels=2,
    #    lr=2e-4,
    #    batch=4,
    #    accum=2,
    #    epochs=8,
    #    seed=42
    #)

    # Step 4: Evaluate the retrained model on unseen_valid_set.csv
    print("\nStep 3: Evaluating trained model on unseen_valid_set.csv...")
    unseen_metrics = evaluate_model(
        model_path="./PT5_finetuned.pth",
        tokenizer_path=None,
        data_path=UNSEEN_VALID_PATH,
        num_labels=2
    )
    print("\nValidation Set Metrics:")
    print(unseen_metrics)

if __name__ == "__main__":
    main()
