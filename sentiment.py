"""
sentiment.py

Lightweight, dependency-free sentiment scoring utilities.
This is a simplified educational example; in production you would
typically connect this to an LLM or a dedicated NLP model/API.
"""

from typing import Iterable

POSITIVE_WORDS = {
    "bullish", "moon", "surge", "rally", "breakout", "gain",
    "buy", "growth", "upgrade", "positive", "adoption",
}

NEGATIVE_WORDS = {
    "bearish", "crash", "dump", "sell-off", "decline", "loss",
    "sell", "downgrade", "negative", "hack", "ban",
}


def score_headline(text: str) -> float:
    """
    Returns a naive sentiment score between -1 (very negative)
    and 1 (very positive) based on keyword matching.
    """
    words = set(text.lower().split())
    pos_hits = len(words & POSITIVE_WORDS)
    neg_hits = len(words & NEGATIVE_WORDS)

    total = pos_hits + neg_hits
    if total == 0:
        return 0.0

    return (pos_hits - neg_hits) / total


def average_sentiment(headlines: Iterable[str]) -> float:
    """Average sentiment score across a collection of headlines."""
    scores = [score_headline(h) for h in headlines]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


if __name__ == "__main__":
    sample_headlines = [
        "Bitcoin rallies as institutional adoption grows",
        "Exchange hacked, millions in crypto stolen",
        "Analysts remain neutral on short-term price action",
    ]
    print("Average sentiment:", average_sentiment(sample_headlines))
