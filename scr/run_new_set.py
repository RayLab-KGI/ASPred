import os
import sys
import torch
import pandas as pd
from test import predict

def main(model_path, csv_path):

    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
    print(f"\nRunning prediction on: {csv_path} using model: {model_path}")

    probs, labs = predict(model_path=model_path, test_df=csv_path)

    print("\nPredicted Probabilities:")
    print(probs)
    print("\nPredicted Labels:")
    print(labs)

    df = pd.read_csv(csv_path)
    df["predicted_probs"] = probs
    df["predicted_labels"] = labs

    output_csv_path=csv_path.replace(".csv", "_predictions.csv")
    df.to_csv(output_csv_path, index=False)
    print(f"\nPredictions saved to: {output_csv_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 run_prediction.py <model_path> <csv_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    csv_path = sys.argv[2]

    main(model_path, csv_path)
