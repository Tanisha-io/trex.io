"""English single-label preprocessing from Chapter 4 (text_emotions.csv)."""

from __future__ import annotations

import re
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import regexp_tokenize

from src.text.emoji_dict import EMOJI_TO_TEXT

CUSTOM_STOPWORDS = {
    "im",
    "ive",
    "dont",
    "cant",
    "wont",
    "didnt",
    "doesnt",
    "isnt",
    "arent",
    "wasnt",
    "werent",
    "hasnt",
    "havent",
    "hadnt",
    "wouldnt",
    "couldnt",
    "shouldnt",
    "thats",
    "theres",
    "heres",
    "youre",
    "theyre",
    "hes",
    "shes",
    "whats",
    "whos",
    "lets",
    "gonna",
    "wanna",
    "gotta",
    "u",
    "ur",
    "r",
    "n",
    "na",
}

TOKEN_PATTERN = r"[a-z]+"


def ensure_nltk() -> None:
    for pkg in ("stopwords", "wordnet", "omw-1.4"):
        nltk.download(pkg, quiet=True)


def replace_emojis(text: str) -> str:
    for emoji, replacement in EMOJI_TO_TEXT.items():
        text = text.replace(emoji, replacement)
    return text


def preprocess_english(text: str, stopword_set: set[str] | None = None) -> str:
    """Lowercase, emoji substitution, strip punctuation/digits, stopwords, lemmatize."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    text = replace_emojis(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = regexp_tokenize(text, TOKEN_PATTERN)
    if stopword_set is None:
        stopword_set = set(stopwords.words("english")) | CUSTOM_STOPWORDS
    lemmatizer = WordNetLemmatizer()
    kept = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stopword_set and len(tok) > 1]
    return " ".join(kept)
