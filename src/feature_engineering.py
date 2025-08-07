# src/feature_engineering.py

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def feature_engineering(x_path: str, y_path: str, x_train_path: str, x_test_path: str, y_train_path: str, y_test_path: str):
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).values.ravel()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Ensure directories exist
    os.makedirs(os.path.dirname(x_train_path), exist_ok=True)

    # Save to CSV
    pd.DataFrame(X_train_scaled).to_csv(x_train_path, index=False)
    pd.DataFrame(X_test_scaled).to_csv(x_test_path, index=False)
    pd.DataFrame(y_train).to_csv(y_train_path, index=False)
    pd.DataFrame(y_test).to_csv(y_test_path, index=False)

    print("✅ Feature engineering completed and saved.")

if __name__ == "__main__":
    feature_engineering(
        x_path="data/processed/X.csv",
        y_path="data/processed/y.csv",
        x_train_path="data/processed/X_train.csv",
        x_test_path="data/processed/X_test.csv",
        y_train_path="data/processed/y_train.csv",
        y_test_path="data/processed/y_test.csv"
    )
