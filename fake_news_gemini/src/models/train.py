import argparse
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import CONFIG
from src.data.preprocess import load_dataset
from src.data.nltk_preprocessor import NLTKPreprocessor


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("nltk", NLTKPreprocessor()),
            (
                "tfidf",
                TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1),
            ),
            ("clf", LogisticRegression(max_iter=1000, n_jobs=None)),
        ]
    )


def train_model(data_path: str, model_out: str) -> str:
    df = load_dataset(data_path)
    X = df["text"].values
    y = df["label"].values
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=CONFIG.test_size, random_state=CONFIG.random_seed, stratify=y
    )
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(pipe, model_out)
    return model_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_out", default="models/fake_news_lr.joblib")
    args = parser.parse_args()

    path = train_model(args.data_path, args.model_out)
    print(f"Model saved to: {path}")


if __name__ == "__main__":
    main()