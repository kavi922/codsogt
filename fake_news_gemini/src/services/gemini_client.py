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
                "domain": "general",
            }
        prompt = (
            "You are a fact-checking and financial fraud detection assistant. Given a news article or banking/financial statement,"
            " analyze whether it is likely real or fake. Consider speculative or phishing-style language."
            " Return ONLY a compact JSON with keys: \n"
            " - verdict: one of ['real','fake','uncertain']\n"
            " - confidence: number in [0,1]\n"
            " - reasoning: short explanation\n"
            " - domain: 'general' or 'banking' (based on content)\n"
            " - red_flags: list of suspicious phrases if any\n\n"
            f"CONTENT:\n{text}\n\n"
            "Return ONLY JSON, no extra text."
        )
        try:
            response = self._model.generate_content(prompt)
            content = response.text or "{}"
        except Exception as e:  # pragma: no cover
            return {
                "verdict": "uncertain",
                "confidence": 0.0,
                "reasoning": f"Gemini call failed: {e}",
                "domain": "general",
            }

        try:
            import json

            parsed = json.loads(content)
            verdict = str(parsed.get("verdict", "uncertain")).lower()
            confidence = float(parsed.get("confidence", 0.0))
            reasoning = str(parsed.get("reasoning", ""))
            domain = str(parsed.get("domain", "general"))
            red_flags = parsed.get("red_flags", [])
            if verdict not in {"real", "fake", "uncertain"}:
                verdict = "uncertain"
            confidence = max(0.0, min(1.0, confidence))
            if not isinstance(red_flags, list):
                red_flags = []
            return {
                "verdict": verdict,
                "confidence": confidence,
                "reasoning": reasoning,
                "domain": domain,
                "red_flags": red_flags,
            }
        except Exception:
            return {
                "verdict": "uncertain",
                "confidence": 0.0,
                "reasoning": "Failed to parse Gemini output.",
                "domain": "general",
                "red_flags": [],
            }