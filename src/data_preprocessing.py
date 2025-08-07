# src/data_preprocessing.py

import pandas as pd
import os
from data_ingestion import ingest  # relative import

def pre_process(output_x_path: str, output_y_path: str):
    # Call ingest and save cleaned data
    ingest(
        input_path="data/Breast_cancer_dataset.csv",
        output_path="data/interim/cleaned.csv"
    )

    # Read cleaned data
    df = pd.read_csv("data/interim/cleaned.csv")

    df = df.drop(['id', 'Unnamed: 32'], axis=1)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    os.makedirs(os.path.dirname(output_x_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_y_path), exist_ok=True)

    X.to_csv(output_x_path, index=False)
    y.to_csv(output_y_path, index=False)

if __name__ == "__main__":
    pre_process(
        output_x_path="data/processed/X.csv",
        output_y_path="data/processed/y.csv"
    )