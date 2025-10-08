from typing import Dict, Any
import joblib


def load_model(model_path: str):
    return joblib.load(model_path)


def predict_text(model, text: str) -> Dict[str, Any]:
    proba = None
    if hasattr(model, "predict_proba"):
        proba_arr = model.predict_proba([text])[0]
        classes = list(model.classes_)
        proba = {cls: float(prob) for cls, prob in zip(classes, proba_arr)}
    label = model.predict([text])[0]
    return {
        "label": str(label),
        "proba": proba or {},
    }