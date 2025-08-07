# src/model_training.py

import pandas as pd
import os
import joblib
import yaml
import mlflow
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import dagshub
dagshub.init(repo_owner='santosh-code', repo_name='mlops-breast-cancer', mlflow=True)
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_svm(X_train_path, y_train_path, X_test_path, y_test_path, model_path, metrics_path):
    # Load data
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()

    # Load config
    config = load_config()
    svm_params = config["model"]
    random_state = config["data"]["random_state"]

    # Train SVM model
    model = SVC(
        kernel=svm_params["kernel"],
        C=svm_params["C"],
        gamma=svm_params["gamma"],
        probability=svm_params["probability"],
        random_state=random_state
    )
    model.fit(X_train, y_train)

    # Predict & evaluate
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"🎯 Accuracy: {acc:.4f}")

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    # Save metrics
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        f.write(f"accuracy: {acc:.4f}")
    print(f"✅ Metrics written to: {metrics_path}")
    print(f"📁 Exists: {os.path.exists(metrics_path)}")

    # Log to MLflow/DagsHub
    mlflow.set_tracking_uri("https://dagshub.com/santosh-code/mlops-breast-cancer.mlflow")


    with mlflow.start_run():
        mlflow.log_param("kernel", svm_params["kernel"])
        mlflow.log_param("C", svm_params["C"])
        mlflow.log_param("gamma", svm_params["gamma"])
        mlflow.log_param("probability", svm_params["probability"])
        mlflow.log_metric("accuracy", acc)
        mlflow.log_artifact(model_path, artifact_path="model")

if __name__ == "__main__":
    train_svm(
        X_train_path="data/processed/X_train.csv",
        y_train_path="data/processed/y_train.csv",
        X_test_path="data/processed/X_test.csv",
        y_test_path="data/processed/y_test.csv",
        model_path="models/svm_model.pkl",
        metrics_path="metrics/metrics.txt"
    )
