from typing import Any, Dict, Optional


class HybridDetector:
    def __init__(self, gemini_high_conf: float = 0.75, ml_margin: float = 0.15) -> None:
        self.gemini_high_conf = gemini_high_conf
        self.ml_margin = ml_margin

    def combine(
        self,
        ml_result: Optional[Dict[str, Any]],
        gemini_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        # Default values
        final_label = "uncertain"
        rationale = []

        # If Gemini is confident, trust it
        if gemini_result is not None:
            verdict = str(gemini_result.get("verdict", "uncertain"))
            conf = float(gemini_result.get("confidence", 0.0))
            if conf >= self.gemini_high_conf and verdict in {"real", "fake"}:
                final_label = verdict
                rationale.append(f"Gemini high confidence ({conf:.2f}) verdict: {verdict}.")

        # Otherwise, use ML if available
        if final_label == "uncertain" and ml_result is not None:
            proba = ml_result.get("proba", {})
            p_fake = float(proba.get("fake", 0.0))
            p_real = float(proba.get("real", 0.0))
            if abs(p_fake - p_real) >= self.ml_margin:
                final_label = "fake" if p_fake > p_real else "real"
                rationale.append(
                    f"ML margin {abs(p_fake - p_real):.2f} exceeds threshold {self.ml_margin:.2f}."
                )

        if final_label == "uncertain":
            rationale.append("Insufficient confidence; returning uncertain.")

        return {
            "label": final_label,
            "ml": ml_result or {},
            "gemini": gemini_result or {},
            "rationale": " ".join(rationale),
        }