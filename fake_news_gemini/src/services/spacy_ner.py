from typing import Dict, List, Optional

try:
    import spacy
except Exception:  # pragma: no cover
    spacy = None  # type: ignore


FIN_ENTITY_LABELS = {"MONEY", "ORG", "DATE", "CARDINAL", "PERCENT"}


class SpacyNER:
    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.model_name = model_name
        self._nlp = None
        if spacy is not None:
            try:
                self._nlp = spacy.load(model_name)
            except Exception:
                self._nlp = None

    def extract(self, text: str) -> List[Dict[str, str]]:
        if self._nlp is None:
            return []
        doc = self._nlp(text)
        ents: List[Dict[str, str]] = []
        for ent in doc.ents:
            if ent.label_ in FIN_ENTITY_LABELS:
                ents.append({"text": ent.text, "label": ent.label_})
        return ents