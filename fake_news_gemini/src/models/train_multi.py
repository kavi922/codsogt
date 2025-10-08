import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from config import CONFIG
from src.data.preprocess import load_dataset
from src.data.nltk_preprocessor import NLTKPreprocessor


def build_pipelines():
    common = [
        ("nltk", NLTKPreprocessor()),
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)),
    ]
    return {
        "logreg": Pipeline(common + [("clf", LogisticRegression(max_iter=1000))]),
        "nb": Pipeline(common + [("clf", MultinomialNB())]),
        "rf": Pipeline(common + [("clf", RandomForestClassifier(n_estimators=200, random_state=CONFIG.random_seed))]),
    }


def train_and_select(data_path: str, out_dir: str) -> str:
    df = load_dataset(data_path)
    X = df["text"].values
    y = df["label"].values
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=CONFIG.test_size, random_state=CONFIG.random_seed, stratify=y
    )

    os.makedirs(out_dir, exist_ok=True)

    best_name = None
    best_f1 = -1.0
    best_model = None

    for name, pipe in build_pipelines().items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        f1 = f1_score(y_val, preds, average="weighted")
        model_path = os.path.join(out_dir, f"fake_news_{name}.joblib")
        joblib.dump(pipe, model_path)
        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = pipe

    best_path = os.path.join(out_dir, f"fake_news_best.joblib")
    if best_model is not None:
        joblib.dump(best_model, best_path)
    print(f"Best model: {best_name} (F1={best_f1:.4f}) saved to {best_path}")
    return best_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--out_dir", default="models")
    args = parser.parse_args()

    train_and_select(args.data_path, args.out_dir)


if __name__ == "__main__":
    main()