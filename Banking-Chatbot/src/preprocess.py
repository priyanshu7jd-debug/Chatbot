"""
Text preprocessing utilities for the banking support chatbot.

Cleaning is intentionally simple (lowercase -> strip punctuation -> remove
stopwords) so it is easy to follow, but it is written once here and imported
everywhere else, so training and inference are guaranteed to preprocess text
the exact same way.
"""

import re

import nltk
from nltk.corpus import stopwords

_STOPWORDS = None


def _get_stopwords():
    """Lazily download (once) and cache the English stopword list."""
    global _STOPWORDS
    if _STOPWORDS is None:
        nltk.download("stopwords", quiet=True)
        _STOPWORDS = set(stopwords.words("english"))
    return _STOPWORDS


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation, and remove stopwords from a string.

    Args:
        text: Raw customer question, e.g. "What is my current account balance?"

    Returns:
        A cleaned string, e.g. "current account balance"
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    stopword_set = _get_stopwords()
    words = [w for w in words if w not in stopword_set]
    return " ".join(words)
