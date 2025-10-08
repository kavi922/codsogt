# Fake News Detection using Gemini API

A full Python project for detecting whether a news article or banking/financial statement is real or fake using both classical machine learning (TF-IDF + Logistic Regression, Random Forest, Naive Bayes) and Google Gemini API for LLM analysis. Includes preprocessing (NLTK), optional spaCy NER for financial entities, model training/selection, evaluation, a weighted hybrid detector, and a Streamlit UI with charts, highlighting, and feedback.

## Features
- ML pipelines: TF-IDF + Logistic Regression / Random Forest / Naive Bayes (auto-select best by F1)
- Gemini API integration for text analysis and structured JSON outputs (domain + red flags)
- Weighted hybrid decision combining ML probabilities and Gemini reasoning
- Dataset utilities: merge general news and bank-specific datasets with labels
- Text preprocessing with NLTK: tokenization, lemmatization, stopword removal
- Optional spaCy NER to extract financial entities (MONEY, ORG, DATE, etc.)
- Streamlit UI: single/batch analysis, charts, keyword highlighting, feedback logging, retraining hints
- Secure API key handling with .env and python-dotenv

## Project Structure
```
fake_news_gemini/
  ├─ app.py
  ├─ config.py
  ├─ requirements.txt
  ├─ .env.example
  ├─ data/
  │   └─ sample_news.csv
  └─ src/
      ├─ __init__.py
      ├─ data/
      │   ├─ preprocess.py
      │   ├─ nltk_preprocessor.py
      │   ├─ keywords.py
      │   └─ merge_datasets.py
      ├─ inference/
      │   └─ predict.py
      ├─ models/
      │   ├─ train.py
      │   ├─ train_multi.py
      │   └─ evaluate.py
      └─ services/
          ├─ gemini_client.py
          ├─ hybrid_detector.py
          └─ spacy_ner.py
```

## Setup
1) Python 3.10–3.12 recommended.
2) venv + install requirements:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
3) Env vars:
- Copy `.env.example` → `.env`, set `GEMINI_API_KEY`.

### NLTK First-Run Downloads
If blocked, pre-download in Python REPL:
```python
import nltk
for p in ["punkt","stopwords","wordnet","omw-1.4"]:
    nltk.download(p)
```

### spaCy NER (optional)
Install the model (if desired):
```bash
python -m spacy download en_core_web_sm
```

## Datasets
- General news (Kaggle) and bank-specific dataset (custom/collected).
- Merge utility:
```python
from src.data.merge_datasets import merge_general_and_bank
merged = merge_general_and_bank("data/general.csv", "data/bank.csv")
merged.to_csv("data/merged.csv", index=False)
```
- Labels: `real`, `fake`, `bank_fake`.

## Training
- Single model (LogReg + NLTK + TF-IDF):
```bash
python -m src.models.train --data_path data/merged.csv --model_out models/fake_news_lr.joblib
```
- Multi-model (LR/RF/NB, pick best by weighted F1):
```bash
python -m src.models.train_multi --data_path data/merged.csv --out_dir models
```

## Evaluation
```bash
python -m src.models.evaluate --data_path data/merged.csv --model_path models/fake_news_best.joblib
```

## Run UI
```bash
streamlit run app.py
```

## Notes
- The UI shows ML prediction, Gemini analysis, and a hybrid decision with a simple confidence bar.
- Highlights financial/scam keywords and optionally shows spaCy NER entities.
- Feedback is appended to `data/feedback.csv` to support future retraining.

## Future Scope
- Real-time bank scam alert feed integration
- LIME/SHAP explainability
- Email/SMS alerts for detected fraud patterns
- Active learning loop and admin retraining tools