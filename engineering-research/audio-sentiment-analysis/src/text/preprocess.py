"""English / Bengali text preprocessing for classical and transformer pipelines."""
from __future__ import annotations

import re
import string
from typing import Iterable, List

# Minimal emoji → word map (extend as needed)
EMOJI_MAP = {
    "😊": " smile ",
    "😂": " laugh ",
    "😢": " cry ",
    "😡": " angry ",
    "😍": " love ",
    "😱": " fear ",
    "😮": " surprise ",
    "😐": " neutral ",
}

_BN_CHARS = re.compile(r"[^\u0980-\u09FF\s]")
_BN_DIGITS = re.compile(r"[০-৯0-9]+")
_EN_CLEAN = re.compile(r"[^a-z\s]")
_WS = re.compile(r"\s+")


def _ensure_nltk() -> None:
    import nltk

    for pkg in ("stopwords", "wordnet", "omw-1.4", "punkt", "punkt_tab"):
        try:
            nltk.data.find(f"corpora/{pkg}" if pkg in ("stopwords", "wordnet", "omw-1.4") else f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


def preprocess_english_classical(text: str) -> str:
    """Thesis classical English pipeline (lemmatize + stopwords)."""
    _ensure_nltk()
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    t = str(text).lower()
    for e, w in EMOJI_MAP.items():
        t = t.replace(e, w)
    t = _EN_CLEAN.sub(" ", t)
    tokens = t.split()
    stops = set(stopwords.words("english")) | {"rt", "amp", "https", "http", "www"}
    lem = WordNetLemmatizer()
    tokens = [lem.lemmatize(tok) for tok in tokens if tok not in stops and len(tok) > 1]
    return " ".join(tokens)


def preprocess_english_bert(text: str) -> str:
    """Light cleanup for BERT — keep morphology/context."""
    t = str(text).lower()
    for e, w in EMOJI_MAP.items():
        t = t.replace(e, w)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = t.translate(str.maketrans("", "", string.punctuation))
    return _WS.sub(" ", t).strip()


def preprocess_bengali_classical(text: str) -> str:
    """Thesis Bengali classical pipeline (no lemmatizer)."""
    t = str(text).lower()
    t = _BN_CHARS.sub(" ", t)
    t = _BN_DIGITS.sub(" ", t)
    t = re.sub(r"[\u0964\u0965" + re.escape(string.punctuation) + "]+", " ", t)
    # Small built-in Bengali stopword seed; replace with full list when available
    bn_stops = {
        "এবং", "ও", "বা", "কিন্তু", "যে", "এই", "সে", "তা", "কি", "না",
        "হয়", "হয়ে", "থেকে", "জন্য", "একটি", "আমি", "তুমি", "আমরা",
    }
    tokens = [tok for tok in t.split() if tok not in bn_stops and len(tok) > 1]
    return " ".join(tokens)


def preprocess_bengali_mbert(text: str) -> str:
    t = str(text).lower()
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"http\S+", " ", t)
    return _WS.sub(" ", t).strip()


def batch_apply(texts: Iterable[str], fn) -> List[str]:
    return [fn(t) for t in texts]
