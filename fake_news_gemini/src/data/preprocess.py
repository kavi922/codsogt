import re
from typing import List, Tuple

import pandas as pd


URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")
NON_ALNUM_RE = re.compile(r"[^a-zA-Z\s]")
MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = NON_ALNUM_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns:
        raise ValueError("Dataset must include a 'text' column")
    if "label" not in df.columns:
        raise ValueError("Dataset must include a 'label' column")
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).map(clean_text)
    df["label"] = df["label"].astype(str).str.lower()
    allowed = {"real", "fake"}
    df = df[df["label"].isin(allowed)].copy()
    return df.reset_index(drop=True)