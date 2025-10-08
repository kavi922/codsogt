import json
from typing import Optional

import pandas as pd
import streamlit as st

from config import CONFIG
from src.inference.predict import load_model, predict_text
from src.services.gemini_client import GeminiClient
from src.services.hybrid_detector import HybridDetector


st.set_page_config(page_title="Fake News Detection (ML + Gemini)", layout="wide")
st.title("Fake News Detection using Gemini API")


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
    return HybridDetector()


with st.sidebar:
    st.header("Settings")
    default_model_path = "models/fake_news_lr.joblib"
    model_path = st.text_input("Model path", value=default_model_path)
    use_gemini = st.checkbox("Use Gemini", value=False)
    use_hybrid = st.checkbox("Use Hybrid (ML + Gemini)", value=False)

    st.caption("Tip: Turn on Gemini or Hybrid after training the ML model and setting GEMINI_API_KEY in .env")

single_tab, batch_tab = st.tabs(["Single Article", "Batch CSV"])

with single_tab:
    st.subheader("Analyze Single Article")
    article = st.text_area("Paste article text", height=220)
    if st.button("Analyze", type="primary"):
        if not article.strip():
            st.warning("Please paste some article text.")
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

            if use_hybrid:
                detector = get_hybrid_detector()
                combined = detector.combine(ml_result, gemini_result)
                st.markdown("**Hybrid Decision**")
                st.json(combined)

            if ml_result is not None:
                st.markdown("**ML Prediction**")
                st.json(ml_result)

            if gemini_result is not None:
                st.markdown("**Gemini Analysis**")
                st.json(gemini_result)

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
                df_out["prob_fake"] = [r["proba"]["fake"] for r in results]
                df_out["prob_real"] = [r["proba"]["real"] for r in results]
                st.dataframe(df_out.head(50))

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