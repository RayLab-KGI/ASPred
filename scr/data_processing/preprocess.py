import pandas as pd

def preprocess_sequences(df, sequence_col="sequence"):
    """Preprocess sequences by replacing uncommon amino acids with X and tokenizing."""
    df[sequence_col] = df[sequence_col].str.replace('|'.join(["O", "B", "U", "Z"]), "X", regex=True)
    df[sequence_col] = df.apply(lambda row: " ".join(row[sequence_col]), axis=1)
    return df