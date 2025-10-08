from typing import Any, Dict, Optional


class HybridDetector:
    def __init__(self, gemini_weight: float = 0.5, ml_weight: float = 0.5, gemini_high_conf: float = 0.75) -> None:
        self.gemini_weight = gemini_weight
        self.ml_weight = ml_weight
        self.gemini_high_conf = gemini_high_conf

    def combine(
        self,
        ml_result: Optional[Dict[str, Any]],
        gemini_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        # Scores for fake vs real
        ml_fake = 0.0
        ml_real = 0.0
        if ml_result is not None and "proba" in ml_result:
            proba = ml_result.get("proba", {})
            ml_fake = float(proba.get("fake", 0.0))
            ml_real = float(proba.get("real", 0.0))
            # optional: if class bank_fake exists, treat as fake
            bank_fake = float(proba.get("bank_fake", 0.0))
            ml_fake = max(ml_fake, bank_fake)

        gem_fake = 0.0
        gem_real = 0.0
        domain = "general"
        reasoning = ""
        if gemini_result is not None:
            verdict = str(gemini_result.get("verdict", "uncertain"))
            conf = float(gemini_result.get("confidence", 0.0))
            domain = str(gemini_result.get("domain", "general"))
            reasoning = str(gemini_result.get("reasoning", ""))
            if verdict == "fake":
                gem_fake = conf
            elif verdict == "real":
                gem_real = conf

        # High-confidence Gemini overrides
        if gemini_result is not None and max(gem_fake, gem_real) >= self.gemini_high_conf:
            final_label = "fake" if gem_fake >= gem_real else "real"
            return {
                "label": final_label,
                "scores": {
                    "ml_fake": ml_fake,
                    "ml_real": ml_real,
                    "gem_fake": gem_fake,
                    "gem_real": gem_real,
                },
                "domain": domain,
                "rationale": f"Gemini high confidence override ({max(gem_fake, gem_real):.2f}). {reasoning}",
                "ml": ml_result or {},
                "gemini": gemini_result or {},
            }

        # Weighted voting
        fake_score = self.ml_weight * ml_fake + self.gemini_weight * gem_fake
        real_score = self.ml_weight * ml_real + self.gemini_weight * gem_real
        final_label = "fake" if fake_score >= real_score else "real"
        return {
            "label": final_label,
            "scores": {
                "fake": fake_score,
                "real": real_score,
                "ml_fake": ml_fake,
                "ml_real": ml_real,
                "gem_fake": gem_fake,
                "gem_real": gem_real,
            },
            "domain": domain,
            "rationale": "Weighted combination of ML and Gemini confidences.",
            "ml": ml_result or {},
            "gemini": gemini_result or {},
        }