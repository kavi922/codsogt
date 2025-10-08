import json
from typing import Optional

import pandas as pd
import streamlit as st

from config import CONFIG
from src.inference.predict import load_model, predict_text
from src.services.gemini_client import GeminiClient
from src.services.hybrid_detector import HybridDetector
from src.services.spacy_ner import SpacyNER
from src.data.keywords import FINANCIAL_KEYWORDS, SCAM_INDICATOR_TERMS


st.set_page_config(page_title="Fake News Detection (ML + Gemini)", layout="wide")
st.title("Fake News & Bank Statement Detection using Gemini API")


def _get_model(model_path: str):
    return load_model(model_path)


@st.cache_resource(show_spinner=False)
def get_gemini_client() -> Optional[GeminiClient]:
    try:
        client = GeminiClient(api_key=CONFIG.gemini_api_key, model_name=CONFIG.gemini_model_name)
        if not client.available:
            return None
        return client
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_hybrid_detector():
    return HybridDetector(gemini_weight=0.5, ml_weight=0.5)


@st.cache_resource(show_spinner=False)
def get_ner():
    return SpacyNER()


def highlight_terms(text: str, terms: list[str]) -> str:
    out = text
    for term in terms:
        if not term:
            continue
        out = out.replace(term, f"**{term}**")
        out = out.replace(term.capitalize(), f"**{term.capitalize()}**")
        out = out.replace(term.upper(), f"**{term.upper()}**")
    return out


with st.sidebar:
    st.header("Settings")
    default_model_path = "models/fake_news_lr.joblib"
    model_path = st.text_input("Model path", value=default_model_path)
    use_gemini = st.checkbox("Use Gemini", value=False)
    use_hybrid = st.checkbox("Use Hybrid (ML + Gemini)", value=False)

    st.caption("Tip: Train the ML model and set GEMINI_API_KEY in .env for Gemini features")

single_tab, batch_tab, admin_tab = st.tabs(["Single Input", "Batch CSV", "Admin / Retrain"])

with single_tab:
    st.subheader("Analyze News or Bank Statement")
    article = st.text_area("Paste content", height=220)
    if st.button("Analyze", type="primary"):
        if not article.strip():
            st.warning("Please paste some text.")
        else:
            try:
                model = _get_model(model_path)
            except Exception as e:
                st.error(f"Failed to load model from {model_path}: {e}")
                model = None

            ml_result = None
            if model is not None:
                ml_result = predict_text(model, article)

            gemini_result = None
            if use_gemini:
                client = get_gemini_client()
                if client is None:
                    st.warning("Gemini client unavailable. Set GEMINI_API_KEY and restart.")
                else:
                    gemini_result = client.analyze_text(article)

            combined = None
            if use_hybrid:
                detector = get_hybrid_detector()
                combined = detector.combine(ml_result, gemini_result)

            col1, col2 = st.columns(2)
            with col1:
                if ml_result is not None:
                    st.markdown("**ML Prediction**")
                    st.json(ml_result)
            with col2:
                if gemini_result is not None:
                    st.markdown("**Gemini Analysis**")
                    st.json(gemini_result)

            if combined is not None:
                st.markdown("**Hybrid Decision**")
                st.json(combined)
                # Simple confidence bar
                scores = combined.get("scores", {})
                fake_score = float(scores.get("fake", 0.0))
                real_score = float(scores.get("real", 0.0))
                st.progress(min(1.0, max(fake_score, real_score)))

            # Keyword highlighting and NER
            st.subheader("Content Highlights")
            ner = get_ner()
            ents = ner.extract(article)
            st.markdown("**Named Entities (financial)**")
            st.write(ents)
            st.markdown("**Keyword Highlighter**")
            st.write("Financial keywords highlighted below:")
            st.markdown(highlight_terms(article, FINANCIAL_KEYWORDS + SCAM_INDICATOR_TERMS))

            # Feedback collection
            st.subheader("Feedback")
            user_label = st.selectbox("Mark this prediction as", ["", "real", "fake", "bank_fake"], index=0)
            if st.button("Submit Feedback"):
                if user_label:
                    try:
                        fb = pd.DataFrame([
                            {"text": article, "label": user_label}
                        ])
                        fb.to_csv("data/feedback.csv", mode="a", index=False, header=False)
                        st.success("Feedback recorded.")
                    except Exception as e:
                        st.error(f"Failed to record feedback: {e}")
                else:
                    st.info("Please choose a label.")

with batch_tab:
    st.subheader("Batch Analyze CSV")
    uploaded = st.file_uploader("Upload CSV with a 'text' column (optional 'label')", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        if "text" not in df.columns:
            st.error("CSV must include a 'text' column.")
        else:
            try:
                model = _get_model(model_path)
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                model = None

            results = []
            if model is not None:
                for t in df["text"].astype(str).tolist():
                    results.append(predict_text(model, t))
                df_out = df.copy()
                df_out["ml_label"] = [r["label"] for r in results]
                df_out["prob_fake"] = [r["proba"].get("fake", 0.0) for r in results]
                df_out["prob_real"] = [r["proba"].get("real", 0.0) for r in results]
                st.dataframe(df_out.head(50))

                # Simple chart
                st.subheader("Prediction Summary")
                st.bar_chart(df_out["ml_label"].value_counts())

                if "label" in df_out.columns:
                    try:
                        from sklearn.metrics import classification_report

                        report = classification_report(df_out["label"], df_out["ml_label"], digits=4)
                        st.markdown("**ML Classification Report**")
                        st.code(report)
                    except Exception as e:
                        st.warning(f"Could not compute metrics: {e}")
            else:
                st.warning("Model not loaded.")

with admin_tab:
    st.subheader("Retraining Mode (Admin)")
    st.write("Append verified data for retraining. Place CSVs under data/ and run the training scripts.")
    st.code("python -m src.models.train_multi --data_path data/merged.csv --out_dir models")