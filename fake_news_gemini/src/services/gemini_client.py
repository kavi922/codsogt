from __future__ import annotations
from typing import Any, Dict, Optional

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None  # type: ignore


class GeminiClient:
    def __init__(self, api_key: Optional[str], model_name: str = "gemini-1.5-flash") -> None:
        self.available = bool(api_key) and genai is not None
        self.model_name = model_name
        if self.available:
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(model_name)
        else:
            self._model = None

    def analyze_text(self, text: str) -> Dict[str, Any]:
        if not self.available or self._model is None:
            return {
                "verdict": "uncertain",
                "confidence": 0.0,
                "reasoning": "Gemini unavailable or API key not set.",
            }
        prompt = (
            "You are a fact-checking assistant. Given a news article, judge if it is likely real or fake. "
            "Return a compact JSON with keys: verdict (one of 'real','fake','uncertain'), confidence (0-1), reasoning (short).\n\n"
            f"ARTICLE:\n{text}\n\n"
            "Return ONLY JSON, no other text."
        )
        try:
            response = self._model.generate_content(prompt)
            content = response.text or "{}"
        except Exception as e:  # pragma: no cover
            return {
                "verdict": "uncertain",
                "confidence": 0.0,
                "reasoning": f"Gemini call failed: {e}",
            }

        try:
            import json

            parsed = json.loads(content)
            verdict = str(parsed.get("verdict", "uncertain")).lower()
            confidence = float(parsed.get("confidence", 0.0))
            reasoning = str(parsed.get("reasoning", ""))
            if verdict not in {"real", "fake", "uncertain"}:
                verdict = "uncertain"
            confidence = max(0.0, min(1.0, confidence))
            return {"verdict": verdict, "confidence": confidence, "reasoning": reasoning}
        except Exception:
            return {
                "verdict": "uncertain",
                "confidence": 0.0,
                "reasoning": "Failed to parse Gemini output.",
            }