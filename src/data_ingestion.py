# src/data_ingestion.py

import pandas as pd
import os

def ingest(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    print("✅ Data ingested:", df.shape)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save ingested data
    df.to_csv(output_path, index=False)
    print("✅ Data saved to:", output_path)

if __name__ == "__main__":
    ingest(
        input_path="data/Breast_cancer_dataset.csv",
        output_path="data/interim/cleaned.csv"
    )
