# Fake News Detection using Gemini API

A full Python project for detecting whether a news article is real or fake using both classical machine learning (TF-IDF + Logistic Regression) and Google Gemini API for large language model (LLM) analysis. Includes preprocessing, model training, evaluation, a hybrid detector, and a Streamlit UI.

## Features
- Classical ML pipeline: TF-IDF + Logistic Regression
- Gemini API integration for text analysis and structured JSON outputs
- Hybrid decision logic combining ML probabilities and Gemini reasoning
- Dataset preprocessing with NLTK: tokenization, lemmatization, stopword removal
- Model training, evaluation, and inference
- Streamlit UI for single-article and CSV batch analysis
- Simple sample dataset to get started

## Project Structure
```
fake_news_gemini/
  ├─ app.py                      # Streamlit app
  ├─ config.py                   # Env & configuration
  ├─ requirements.txt
  ├─ .env.example
  ├─ data/
  │   └─ sample_news.csv        # Small sample dataset
  └─ src/
      ├─ __init__.py
      ├─ data/
      │   ├─ preprocess.py
      │   └─ nltk_preprocessor.py
      ├─ inference/
      │   └─ predict.py
      ├─ models/
      │   ├─ train.py
      │   └─ evaluate.py
      └─ services/
          ├─ gemini_client.py
          └─ hybrid_detector.py
```

## Setup
1) Python 3.10–3.12 recommended (to avoid building heavy wheels).
2) Create a virtual environment and install requirements:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
3) Configure environment variables:
- Copy `.env.example` to `.env` and fill in `GEMINI_API_KEY` if you plan to use the Gemini features.

### NLTK First-Run Downloads
On the first run, the pipeline attempts to download required NLTK resources if missing: `punkt`, `stopwords`, `wordnet`, `omw-1.4`. If your environment blocks downloads, pre-download them:
```python
import nltk
for p in ["punkt","stopwords","wordnet","omw-1.4"]:
    nltk.download(p)
```

## Datasets
- Expected CSV format: columns `text` and `label` where `label` ∈ {`real`, `fake`}.
- A tiny `data/sample_news.csv` is included for demonstration only. For real experiments, use a larger dataset (e.g., Kaggle fake news datasets).

## Preprocessing (src/data)
- `preprocess.py`: normalization (lowercase, remove URLs/HTML/non-alphanumeric, collapse whitespace).
- `nltk_preprocessor.py`: tokenization → stopword removal → lemmatization; integrated in the sklearn Pipeline.

## Model Training (src/models/train.py)
- Pipeline: `NLTKPreprocessor()` → `TfidfVectorizer(stop_words='english', ngram_range=(1,2))` → `LogisticRegression(max_iter=1000)`.
- Saves a single `scikit-learn` Pipeline with both vectorizer and classifier using `joblib`.
- CLI usage example (run from project root):
```bash
python -m src.models.train --data_path data/sample_news.csv --model_out models/fake_news_lr.joblib
```

## Evaluation (src/models/evaluate.py)
- Loads the trained pipeline and evaluates on a hold-out split.
- Reports accuracy, precision, recall, F1, and confusion matrix.
- CLI usage example (run from project root):
```bash
python -m src.models.evaluate --data_path data/sample_news.csv --model_path models/fake_news_lr.joblib
```

## Inference (src/inference/predict.py)
- Utility functions to load the saved pipeline and predict labels and probabilities for raw text.

## Gemini API Integration (src/services/gemini_client.py)
- Wrapper around `google-generativeai` to analyze text with Gemini.
- Returns structured JSON: `verdict` (`real`/`fake`/`uncertain`), `confidence` (0-1), and `reasoning`.
- Works when `GEMINI_API_KEY` is set. If absent, it degrades gracefully.

## Hybrid Detector (src/services/hybrid_detector.py)
- Combines ML probability with Gemini verdict.
- Rule-of-thumb logic: trust high-confidence Gemini; otherwise, rely on ML with configurable thresholds.

## Streamlit UI (app.py)
- Single-article analysis with ML-only or ML+Gemini modes.
- CSV batch analysis to process many rows.
- Displays probabilities, LLM verdict, and combined (hybrid) decision.

## How to Run
- Train a model:
```bash
python -m src.models.train --data_path data/sample_news.csv --model_out models/fake_news_lr.joblib
```
- Evaluate:
```bash
python -m src.models.evaluate --data_path data/sample_news.csv --model_path models/fake_news_lr.joblib
```
- Launch UI:
```bash
streamlit run app.py
```

## Module-by-Module Explanation
- `config.py`: Loads environment variables (`GEMINI_API_KEY`, model name) and shared constants.
- `src/data/preprocess.py`: Text cleaning helpers and dataset loader.
- `src/data/nltk_preprocessor.py`: NLTK-based transformer for tokenization, stopwords, lemmatization.
- `src/models/train.py`: End-to-end training script for TF-IDF + Logistic Regression pipeline, saving the trained model.
- `src/models/evaluate.py`: Evaluation script for reporting classification metrics.
- `src/inference/predict.py`: Loads the saved pipeline and exposes a single-text prediction function.
- `src/services/gemini_client.py`: Light wrapper to call Gemini API and parse structured outputs.
- `src/services/hybrid_detector.py`: Combines ML and Gemini signals into one decision with an explanation.
- `app.py`: Streamlit UI integrating ML, Gemini, and Hybrid modes for interactive use.

## Future Scope
- Incorporate retrieval (search) for evidence-based fact checking.
- Use more advanced models (e.g., linear SVM, calibrated classifiers, or transformers).
- Active learning loop: let users provide feedback to continuously improve the model.
- Explainability: show top TF-IDF features contributing to predictions.
- Multi-lingual support and language detection.
- Model monitoring and drift detection if deployed as a service.

## Ethics and Limitations
- LLMs can hallucinate; Hybrid mode tries to mitigate by combining signals.
- Do not blindly trust automated verdicts for high-stakes decisions.
- Datasets may contain biases—ensure diverse, representative data and continuous evaluation.