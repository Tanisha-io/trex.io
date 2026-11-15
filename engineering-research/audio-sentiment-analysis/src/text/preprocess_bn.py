"""Bengali single-label preprocessing from Chapter 4 (custom CSV)."""

from __future__ import annotations

import re
import unicodedata

# Common Bengali function words; no robust lemmatizer is used (thesis).
BENGALI_STOPWORDS = {
    "এবং",
    "এবংও",
    "বা",
    "কিন্তু",
    "যদি",
    "তবে",
    "এই",
    "সেই",
    "ওই",
    "এটা",
    "সেটা",
    "ওটা",
    "আমি",
    "আমার",
    "আমরা",
    "আমাদের",
    "তুমি",
    "তোমার",
    "তোমরা",
    "সে",
    "তার",
    "তারা",
    "তাদের",
    "এরা",
    "ওরা",
    "কি",
    "কী",
    "কে",
    "কেন",
    "কখন",
    "কোথায়",
    "কোন",
    "কোনো",
    "না",
    "নি",
    "নেই",
    "হয়",
    "হয়েছে",
    "হবে",
    "ছিল",
    "ছিলেন",
    "আছে",
    "থাকে",
    "থেকে",
    "জন্য",
    "সাথে",
    "মধ্যে",
    "উপর",
    "নিচে",
    "পরে",
    "আগে",
    "এক",
    "একটি",
    "এবং",
    "যে",
    "যা",
    "যারা",
    "তাই",
    "তেমনি",
    "অথবা",
    "ইত্যাদি",
    "করে",
    "করা",
    "করতে",
    "করেছে",
    "গিয়ে",
    "গেছে",
    "যাচ্ছে",
    "দিয়ে",
    "দেওয়া",
    "হলো",
    "হল",
    "বলে",
    "বলা",
    "মত",
    "মতো",
    "খুব",
    "অনেক",
    "সব",
    "সকল",
    "প্রতি",
    "এখন",
    "তখন",
    "আজ",
    "কাল",
    "যখন",
    "যেহেতু",
    "অর্থাৎ",
    "শুধু",
    "মাত্র",
    "আবার",
    "এমন",
    "তেমন",
    "কিছু",
    "কেউ",
    "সবাই",
}

BENGALI_TOKEN_RE = re.compile(r"[\u0980-\u09FF]+")
NON_BENGALI_RE = re.compile(r"[A-Za-z]+")
DIGIT_PUNCT_RE = re.compile(
    r"[0-9\u09E6-\u09EF\u0660-\u0669"
    r"।,;:!?\.\-\(\)\[\]\{\}\"\"''`~@#$%^&*_+=/\\|<>]+"
)


def preprocess_bengali(text: str, stopword_set: set[str] | None = None) -> str:
    """Lowercase, drop Latin, strip punctuation/digits, Bengali stopwords. No lemmatization."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    text = NON_BENGALI_RE.sub(" ", text)
    text = DIGIT_PUNCT_RE.sub(" ", text)
    tokens = BENGALI_TOKEN_RE.findall(text)
    if stopword_set is None:
        stopword_set = BENGALI_STOPWORDS
    kept = [tok for tok in tokens if tok not in stopword_set]
    return " ".join(kept)
