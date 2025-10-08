# Fake News Detection using Gemini API — Project Report

## 1. Abstract
This project detects whether a news article is real or fake using a classical machine learning approach (TF-IDF + Logistic Regression) and an LLM-based analysis with the Gemini API. A hybrid strategy combines both signals to improve reliability. We provide end-to-end preprocessing (including NLTK), training, evaluation, inference utilities, and an interactive Streamlit UI.

## 2. Introduction
- Motivation: Rapid spread of misinformation and the need for automated assistance.
- Goal: Build a robust, extensible pipeline that blends interpretable ML with LLM reasoning.
- Contributions: Clean architecture, reproducible training/evaluation, Gemini wrapper, NLTK preprocessing, and a hybrid combiner.

## 3. Dataset
- Expected schema: CSV with `text` and `label` where `label` ∈ {`real`,`fake`}.
- Included sample: `data/sample_news.csv` (tiny, for demonstration only).
- Recommended real datasets: Kaggle fake news datasets, LIAR dataset, etc.

## 4. Preprocessing
- Normalization: lowercase, URL/HTML removal, non-alphanumeric filtering, whitespace normalization (`src/data/preprocess.py`).
- NLTK Processing: tokenization, stopword removal, lemmatization via `NLTKPreprocessor` (`src/data/nltk_preprocessor.py`).
- First-run resource downloads: `punkt`, `stopwords`, `wordnet`, `omw-1.4` (attempted automatically with graceful fallback).

## 5. Model: TF-IDF + Logistic Regression
- Vectorizer: `TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=1)`.
- Classifier: `LogisticRegression(max_iter=1000)`.
- Pipeline: `NLTKPreprocessor()` → `TF-IDF` → `LogReg`.
- Persisted as a single `sklearn` Pipeline via `joblib`.
- Training script: `src/models/train.py`.

## 6. Evaluation
- Hold-out validation using `train_test_split` with a fixed seed.
- Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix, full classification report.
- Script: `src/models/evaluate.py`.

## 7. Inference
- `src/inference/predict.py` exposes `load_model` and `predict_text` returning label and probabilities.
- Used by the Streamlit app and the hybrid detector.

## 8. Gemini API Integration
- Wrapper: `src/services/gemini_client.py` using `google-generativeai`.
- Prompting: Requests a compact JSON with `verdict` (`real`/`fake`/`uncertain`), `confidence` in [0,1], and `reasoning`.
- Graceful degradation when API key is missing or errors occur.

## 9. Hybrid Detection Logic
- Module: `src/services/hybrid_detector.py`.
- Strategy: Trust Gemini if confidence ≥ 0.75; otherwise fall back to ML if the probability margin ≥ 0.15. Else return `uncertain`.
- Returns combined output with rationale for transparency.

## 10. Streamlit UI
- File: `app.py`.
- Features: Single-article analysis, CSV batch processing, toggles for ML, Gemini, and Hybrid modes.
- Displays probabilities, LLM verdict, and combined decision.

## 11. System Setup and Usage
- Python: Recommended 3.10–3.12.
- Install: `pip install -r requirements.txt` (set up a virtual environment).
- Env: Copy `.env.example` → `.env` and set `GEMINI_API_KEY` to enable Gemini.
- Train: `python -m src.models.train --data_path data/your.csv --model_out models/fake_news_lr.joblib`.
- Evaluate: `python -m src.models.evaluate --data_path data/your.csv --model_path models/fake_news_lr.joblib`.
- Run UI: `streamlit run app.py`.

## 12. Results (Example)
- On real datasets, report macro metrics and confusion matrix.
- Discuss tradeoffs, threshold tuning, and class imbalance handling.

## 13. Future Work
- Retrieval-augmented generation (RAG) for evidence-based fact checking.
- Advanced ML: linear SVM, calibrated probabilities, transformer fine-tuning.
- Explainability: feature importance, LIME/SHAP for text.
- Multilingual support and domain adaptation.
- Feedback loop and continuous learning.

## 14. Ethical Considerations
- LLM hallucinations; combine with ML for safety.
- Avoid high-stakes automation without human oversight.
- Dataset bias and fairness monitoring.

## 15. Module Map and Responsibilities
- `config.py`: Environment configuration.
- `src/data/preprocess.py`: Text normalization and dataset loader.
- `src/data/nltk_preprocessor.py`: NLTK tokenization, stopword removal, lemmatization transformer.
- `src/models/train.py`: Train and persist TF-IDF + LR pipeline including NLTK step.
- `src/models/evaluate.py`: Validation metrics and reporting.
- `src/inference/predict.py`: Load model and predict on raw text.
- `src/services/gemini_client.py`: Gemini API client wrapper.
- `src/services/hybrid_detector.py`: Combine ML and Gemini outputs.
- `app.py`: Streamlit-based user interface.

## 16. References
- scikit-learn documentation (`https://scikit-learn.org/`)
- Streamlit documentation (`https://docs.streamlit.io/`)
- Google Generative AI Python SDK (`https://ai.google.dev/gemini-api/docs`)
- NLTK documentation (`https://www.nltk.org/`)