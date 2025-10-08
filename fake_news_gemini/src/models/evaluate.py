import argparse
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from config import CONFIG
from src.data.preprocess import load_dataset


def evaluate_model(data_path: str, model_path: str) -> str:
    df = load_dataset(data_path)
    X = df["text"].values
    y = df["label"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=CONFIG.test_size, random_state=CONFIG.random_seed, stratify=y
    )

    model = joblib.load(model_path)
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_val, y_pred, labels=["fake", "real"]).tolist()

    report = classification_report(y_val, y_pred, digits=4)
    out = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix_labels": ["fake", "real"],
        "confusion_matrix": cm,
        "report": report,
    }

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()

    print(evaluate_model(args.data_path, args.model_path))


if __name__ == "__main__":
    main()