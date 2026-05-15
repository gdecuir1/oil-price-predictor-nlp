"""
Oil-market keyword features from article text.
==============================================

Extracts a fixed-size vector of domain keyword hit counts from each article.
These features are concatenated to FinBERT day embeddings so the LSTM can use
explicit bullish/bearish/supply/demand signals, not only dense vectors.

All keyword lists live here (single source of truth).  Counts are normalised by
article length so long articles are not automatically scored higher.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import torch

# Ordered groups — each becomes one feature dimension after normalisation.
KEYWORD_GROUPS: Dict[str, List[str]] = {
    "bullish": [
        "surge", "rally", "soar", "climb", "jump", "gain", "rise", "boost",
        "recover", "rebound", "uptick", "bullish", "demand growth", "supply cut",
        "production cut", "opec cut", "shortage", "tighten", "drawdown",
        "inventory decline", "higher prices",
    ],
    "bearish": [
        "plunge", "crash", "fall", "drop", "decline", "slump", "tumble",
        "slide", "sink", "bearish", "oversupply", "glut", "demand destruction",
        "production increase", "output hike", "recession", "surplus",
        "inventory build", "lower prices", "weakening",
    ],
    "supply_up": [
        "production increase", "output hike", "ramp up", "new wells", "drilling",
        "shale boom", "opec increase", "supply rise", "surplus", "export increase",
    ],
    "supply_down": [
        "production cut", "supply disruption", "outage", "sanctions", "opec cut",
        "output reduction", "maintenance", "shut-in", "export ban", "embargo",
    ],
    "demand_up": [
        "demand growth", "economic recovery", "travel rebound", "consumption rise",
        "refinery run", "seasonal demand", "china demand", "import increase",
    ],
    "demand_down": [
        "demand destruction", "recession", "lockdown", "slowdown",
        "consumption decline", "refinery shutdown", "weak demand",
    ],
    "geopolitical": [
        "war", "conflict", "invasion", "sanctions", "embargo", "military",
        "attack", "drone strike", "pipeline attack", "nuclear", "coup",
    ],
    "volatility": [
        "volatile", "volatility", "swing", "whipsaw", "turbulent", "uncertainty",
        "sharp move", "sudden", "dramatic",
    ],
}

GROUP_NAMES: Tuple[str, ...] = tuple(KEYWORD_GROUPS.keys())
KEYWORD_DIM: int = len(GROUP_NAMES)


def extract_keyword_vector(text: str) -> torch.Tensor:
    """Count normalised keyword-group hits in ``text``.

    Args:
        text: Article body (any case).

    Returns:
        Float tensor of shape ``(KEYWORD_DIM,)`` — one score per group in
        ``GROUP_NAMES`` order, each in ``[0, 1]`` via ``hits / (hits + 10)``.
    """
    text_lower = text.lower()
    # Word count proxy for length normalisation.
    n_words = max(len(re.findall(r"\b\w+\b", text_lower)), 1)
    scores: List[float] = []
    for name in GROUP_NAMES:
        hits = sum(1 for kw in KEYWORD_GROUPS[name] if kw in text_lower)
        # Per-1000-words rate, squashed to (0,1).
        rate = hits / (n_words / 1000.0 + 1.0)
        scores.append(min(1.0, rate / (rate + 2.0)))
    return torch.tensor(scores, dtype=torch.float32)


def aggregate_keyword_vectors(vectors: List[torch.Tensor]) -> torch.Tensor:
    """Mean-aggregate per-article keyword vectors for one day.

    Args:
        vectors: Non-empty list of ``(KEYWORD_DIM,)`` tensors.

    Returns:
        Day-level keyword vector; zeros if ``vectors`` is empty.
    """
    if not vectors:
        return torch.zeros(KEYWORD_DIM, dtype=torch.float32)
    return torch.stack(vectors, dim=0).mean(dim=0)
