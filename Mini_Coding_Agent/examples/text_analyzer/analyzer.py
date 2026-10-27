"""Word-frequency analysis."""
from collections import Counter
from normalizer import tokenize


def top_words(text, limit=5):
    counts = Counter(tokenize(text))
    return sorted(counts.items())[:limit]
