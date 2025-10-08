from typing import Iterable, List
import re

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except Exception:  # pragma: no cover
    nltk = None
    stopwords = None
    WordNetLemmatizer = None  # type: ignore
    word_tokenize = None  # type: ignore

from src.data.preprocess import clean_text


class NLTKPreprocessor:
    def __init__(self, enable_download: bool = True) -> None:
        self.enable_download = enable_download
        self._lemmatizer = None
        self._stop_words = set()
        self._initialized = False

    def _ensure_nltk(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        if nltk is None:
            return
        # Try to load required resources; optionally download if missing
        needed = ["punkt", "stopwords", "wordnet", "omw-1.4"]
        for res in needed:
            try:
                nltk.data.find(res)
            except LookupError:
                if self.enable_download:
                    try:
                        nltk.download(res, quiet=True)
                    except Exception:
                        pass
        # Initialize tools if available
        try:
            self._lemmatizer = WordNetLemmatizer() if WordNetLemmatizer else None
        except Exception:
            self._lemmatizer = None
        try:
            self._stop_words = set(stopwords.words("english")) if stopwords else set()
        except Exception:
            self._stop_words = set()

    def fit(self, X: Iterable[str], y=None):  # sklearn API
        self._ensure_nltk()
        return self

    def _process_one(self, text: str) -> str:
        # Basic normalization then NLTK pipeline
        text = clean_text(text)
        if nltk is None or word_tokenize is None:
            return text
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()
        processed: List[str] = []
        for token in tokens:
            token = token.strip()
            if not token.isalpha():
                continue
            lower = token.lower()
            if self._stop_words and lower in self._stop_words:
                continue
            if self._lemmatizer is not None:
                try:
                    lower = self._lemmatizer.lemmatize(lower)
                except Exception:
                    pass
            processed.append(lower)
        return " ".join(processed)

    def transform(self, X: Iterable[str]) -> List[str]:  # sklearn API
        self._ensure_nltk()
        return [self._process_one(x if isinstance(x, str) else "") for x in X]