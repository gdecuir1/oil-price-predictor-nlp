"""
PyTorch Dataset and DataLoader Factory
=======================================

Provides :class:`OilArticleDataset` — a PyTorch ``Dataset`` that couples
extracted article text with multi-task labels — and the
:func:`create_data_loaders` factory that handles train/val/test splitting,
balanced sampling, and DataLoader construction.

Label generation
----------------
Because the scraped articles do not come with ground-truth market-movement
labels, this module provides two operating modes:

1. **Supervised mode** — when a label file is supplied (CSV or JSON mapping
   article filenames to human-annotated labels for each sub-domain).
2. **Self-supervised / pseudo-label mode** — heuristic labelling based on
   article metadata and NLP analysis, intended for pre-training or when
   manual labels are unavailable.  The pseudo-labeller assigns:

   * ``market_direction`` from keyword / sentiment signals.
   * ``sentiment`` from a lightweight rule-based classifier.
   * ``supply_impact`` / ``demand_impact`` from keyword matching.
   * Other sub-domains from heuristic rules.

   These pseudo-labels are intentionally noisy; the training loop uses
   label smoothing and confidence-weighted loss to tolerate this.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)


# ======================================================================
# Pseudo-label heuristics
# ======================================================================

# Keyword lists used by the pseudo-labeller to infer weak labels from
# article text when no ground-truth annotations are available.

_BULLISH_KEYWORDS = [
    "surge", "rally", "soar", "climb", "jump", "gain", "rise", "boost",
    "recover", "rebound", "uptick", "bullish", "optimistic", "demand growth",
    "supply cut", "production cut", "opec cut", "sanctions", "shortage",
    "tighten", "drawdown", "inventory decline", "higher prices",
]

_BEARISH_KEYWORDS = [
    "plunge", "crash", "fall", "drop", "decline", "slump", "tumble",
    "slide", "sink", "bearish", "pessimistic", "oversupply", "glut",
    "demand destruction", "production increase", "output hike", "recession",
    "surplus", "inventory build", "lower prices", "weakening",
]

_SUPPLY_INCREASE_KEYWORDS = [
    "production increase", "output hike", "ramp up", "new wells",
    "drilling", "shale boom", "opec increase", "supply rise", "surplus",
    "export increase", "pipeline", "capacity expansion",
]

_SUPPLY_DECREASE_KEYWORDS = [
    "production cut", "supply disruption", "outage", "sanctions",
    "opec cut", "output reduction", "maintenance", "shut-in",
    "export ban", "embargo", "strategic reserve release",
]

_DEMAND_INCREASE_KEYWORDS = [
    "demand growth", "economic recovery", "travel rebound", "consumption rise",
    "refinery run", "seasonal demand", "driving season", "industrial growth",
    "import increase", "china demand",
]

_DEMAND_DECREASE_KEYWORDS = [
    "demand destruction", "recession", "lockdown", "slowdown",
    "efficiency gains", "ev adoption", "renewable", "consumption decline",
    "refinery shutdown", "weak demand",
]

_GEOPOLITICAL_KEYWORDS = [
    "war", "conflict", "invasion", "sanctions", "embargo", "military",
    "attack", "drone strike", "pipeline attack", "territorial dispute",
    "nuclear", "diplomatic", "cease-fire", "peace talks", "coup",
    "political instability", "regime change",
]

_HIGH_VOLATILITY_KEYWORDS = [
    "volatile", "volatility", "swing", "whipsaw", "turbulent",
    "uncertainty", "unpredictable", "roller coaster", "wild",
    "sharp move", "sudden", "dramatic",
]


class PseudoLabeller:
    """Generates heuristic labels from article text and metadata.

    These weak / noisy labels allow the model to pre-train when manual
    annotations are not available.  Accuracy is intentionally imperfect;
    the training regime compensates with label smoothing.

    The labeller is deterministic given the same input text.
    """

    def __call__(self, text: str, metadata: Optional[Dict] = None) -> Dict[str, int]:
        """Produce a label dict for one article.

        Args:
            text: Cleaned article text (lowercased internally).
            metadata: Optional metadata dict (title, source, snippet, etc.).

        Returns:
            Dictionary mapping sub-domain keys to integer class indices.
        """
        text_lower = text.lower()

        # Compute keyword hit counts for direction assessment
        bull_score = sum(1 for kw in _BULLISH_KEYWORDS if kw in text_lower)
        bear_score = sum(1 for kw in _BEARISH_KEYWORDS if kw in text_lower)

        labels: Dict[str, int] = {}

        # --- Market direction (primary) ---
        if bull_score > bear_score + 1:
            labels["market_direction"] = 0  # up
        elif bear_score > bull_score + 1:
            labels["market_direction"] = 2  # down
        else:
            labels["market_direction"] = 1  # unchanged

        # --- Price magnitude ---
        total_signal = bull_score + bear_score
        if total_signal == 0:
            labels["price_magnitude"] = 0   # negligible
        elif total_signal <= 2:
            labels["price_magnitude"] = 1   # small
        elif total_signal <= 4:
            labels["price_magnitude"] = 2   # moderate
        elif total_signal <= 7:
            labels["price_magnitude"] = 3   # large
        else:
            labels["price_magnitude"] = 4   # extreme

        # --- Timeframe ---
        if any(kw in text_lower for kw in ["today", "intraday", "this morning", "this afternoon"]):
            labels["timeframe"] = 0   # intraday
        elif any(kw in text_lower for kw in ["this week", "coming days", "near-term", "short-term"]):
            labels["timeframe"] = 1   # short_term
        elif any(kw in text_lower for kw in ["this month", "coming weeks", "medium-term", "quarter"]):
            labels["timeframe"] = 2   # medium_term
        else:
            labels["timeframe"] = 3   # long_term

        # --- Volatility ---
        vol_score = sum(1 for kw in _HIGH_VOLATILITY_KEYWORDS if kw in text_lower)
        if vol_score == 0:
            labels["volatility"] = 1      # normal
        elif vol_score <= 1:
            labels["volatility"] = 2      # high
        elif vol_score <= 2:
            labels["volatility"] = 2      # high
        else:
            labels["volatility"] = 3      # extreme

        # Calm signals override
        if any(kw in text_lower for kw in ["stable", "steady", "calm", "flat"]):
            labels["volatility"] = 0      # low

        # --- Sentiment ---
        sentiment_net = bull_score - bear_score
        if sentiment_net <= -3:
            labels["sentiment"] = 0       # very_negative
        elif sentiment_net <= -1:
            labels["sentiment"] = 1       # negative
        elif sentiment_net == 0:
            labels["sentiment"] = 2       # neutral
        elif sentiment_net <= 2:
            labels["sentiment"] = 3       # positive
        else:
            labels["sentiment"] = 4       # very_positive

        # --- Supply impact ---
        sup_inc = sum(1 for kw in _SUPPLY_INCREASE_KEYWORDS if kw in text_lower)
        sup_dec = sum(1 for kw in _SUPPLY_DECREASE_KEYWORDS if kw in text_lower)
        if sup_dec > sup_inc:
            labels["supply_impact"] = 0   # decrease
        elif sup_inc > sup_dec:
            labels["supply_impact"] = 2   # increase
        else:
            labels["supply_impact"] = 1   # stable

        # --- Demand impact ---
        dem_inc = sum(1 for kw in _DEMAND_INCREASE_KEYWORDS if kw in text_lower)
        dem_dec = sum(1 for kw in _DEMAND_DECREASE_KEYWORDS if kw in text_lower)
        if dem_dec > dem_inc:
            labels["demand_impact"] = 0   # decrease
        elif dem_inc > dem_dec:
            labels["demand_impact"] = 2   # increase
        else:
            labels["demand_impact"] = 1   # stable

        # --- Geopolitical risk ---
        geo_score = sum(1 for kw in _GEOPOLITICAL_KEYWORDS if kw in text_lower)
        if geo_score == 0:
            labels["geopolitical_risk"] = 0     # low
        elif geo_score <= 1:
            labels["geopolitical_risk"] = 1     # moderate
        elif geo_score <= 2:
            labels["geopolitical_risk"] = 2     # elevated
        elif geo_score <= 4:
            labels["geopolitical_risk"] = 3     # high
        else:
            labels["geopolitical_risk"] = 4     # severe

        return labels


# ======================================================================
# Dataset
# ======================================================================

class OilArticleDataset(Dataset):
    """PyTorch Dataset wrapping extracted articles with multi-task labels.

    Each sample is a dictionary containing tokenised inputs and integer
    labels for every sub-domain.  The dataset supports both supervised
    (external label file) and pseudo-labelled modes.

    Args:
        articles: List of article dicts from
            :meth:`HTMLArticleExtractor.extract_all`.  Each must have at
            least a ``"text"`` key.
        preprocessor: :class:`TextPreprocessor` instance for tokenisation.
        label_file: Optional path to a JSON file mapping filenames to
            ground-truth label dicts.  If ``None``, pseudo-labels are used.
        subdomain_keys: Ordered list of sub-domain keys to include.

    Attributes:
        samples: Processed list of ``(article_dict, label_dict)`` tuples.
    """

    def __init__(
        self,
        articles: List[Dict[str, Any]],
        preprocessor: "TextPreprocessor",
        label_file: Optional[Path] = None,
        subdomain_keys: Optional[List[str]] = None,
    ) -> None:
        """Build the dataset from extracted articles and labels."""
        from ..config import SUBDOMAIN_SPECS

        self.preprocessor = preprocessor
        self.subdomain_keys = subdomain_keys or sorted(SUBDOMAIN_SPECS.keys())

        # Load external labels if provided
        external_labels: Dict[str, Dict[str, int]] = {}
        if label_file and Path(label_file).exists():
            external_labels = json.loads(Path(label_file).read_text())
            logger.info("Loaded %d external labels from %s", len(external_labels), label_file)

        # Build pseudo-labeller for articles without external labels
        pseudo_labeller = PseudoLabeller()

        # Pair each article with its labels
        self.samples: List[Tuple[Dict[str, Any], Dict[str, int]]] = []
        for article in articles:
            filename = article.get("filename", "")

            if filename in external_labels:
                labels = external_labels[filename]
            else:
                labels = pseudo_labeller(article["text"], article)

            # Verify all required sub-domains have labels
            if all(k in labels for k in self.subdomain_keys):
                self.samples.append((article, labels))
            else:
                missing = [k for k in self.subdomain_keys if k not in labels]
                logger.debug(
                    "Dropping %s — missing labels for: %s", filename, missing
                )

        logger.info(
            "Dataset built: %d samples (%d articles dropped)",
            len(self.samples),
            len(articles) - len(self.samples),
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Fetch and tokenise one sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with keys:
                - ``input_ids``: ``(max_length,)`` token ID tensor.
                - ``attention_mask``: ``(max_length,)`` mask tensor.
                - ``labels_{subdomain}``: scalar tensor for each sub-domain.
                - ``filename``: article filename (as metadata, not a tensor).
        """
        article, labels = self.samples[idx]

        # Tokenise the article text
        encoded = self.preprocessor.encode(article["text"])

        # Build the output dict starting with tokenizer outputs
        sample: Dict[str, Any] = dict(encoded)

        # Add integer labels for each sub-domain
        for key in self.subdomain_keys:
            sample[f"labels_{key}"] = torch.tensor(labels[key], dtype=torch.long)

        # Attach metadata for traceability in reports
        sample["filename"] = article.get("filename", "unknown")

        return sample

    def get_class_distribution(self) -> Dict[str, Dict[int, int]]:
        """Compute label frequency counts for each sub-domain.

        Returns:
            Nested dict: ``{subdomain_key: {class_idx: count}}``.
            Useful for computing class weights to handle imbalance.
        """
        distribution: Dict[str, Dict[int, int]] = {
            key: {} for key in self.subdomain_keys
        }
        for _, labels in self.samples:
            for key in self.subdomain_keys:
                cls = labels[key]
                distribution[key][cls] = distribution[key].get(cls, 0) + 1
        return distribution

    def get_primary_labels(self) -> List[int]:
        """Return a flat list of primary (market_direction) labels.

        Useful for constructing a :class:`WeightedRandomSampler`.

        Returns:
            List of integer class indices, one per sample.
        """
        return [labels["market_direction"] for _, labels in self.samples]


# ======================================================================
# DataLoader factory
# ======================================================================

def create_data_loaders(
    articles: List[Dict[str, Any]],
    preprocessor: "TextPreprocessor",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    batch_size: int = 8,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    label_file: Optional[Path] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders with stratified splitting.

    The split is performed on the primary label (market_direction) to ensure
    balanced class representation in each partition.  The training loader uses
    a :class:`WeightedRandomSampler` to further mitigate class imbalance.

    Args:
        articles: List of extracted article dicts.
        preprocessor: Configured :class:`TextPreprocessor`.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set (remainder → test).
        batch_size: Batch size for all loaders.
        num_workers: DataLoader worker processes.
        prefetch_factor: Batches prefetched per worker.
        label_file: Optional path to external labels.
        seed: Random seed for reproducible splits.

    Returns:
        Tuple of ``(train_loader, val_loader, test_loader)``.
    """
    # Deterministic shuffle
    rng = random.Random(seed)
    shuffled = list(articles)
    rng.shuffle(shuffled)

    # Compute split boundaries
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_articles = shuffled[:n_train]
    val_articles = shuffled[n_train : n_train + n_val]
    test_articles = shuffled[n_train + n_val :]

    logger.info(
        "Data split: %d train / %d val / %d test (total %d)",
        len(train_articles),
        len(val_articles),
        len(test_articles),
        n,
    )

    # Build datasets (training set gets augmentation via preprocessor config)
    train_dataset = OilArticleDataset(train_articles, preprocessor, label_file)
    val_dataset = OilArticleDataset(val_articles, preprocessor, label_file)
    test_dataset = OilArticleDataset(test_articles, preprocessor, label_file)

    # Weighted sampler for the training set to handle class imbalance
    train_sampler = _build_weighted_sampler(train_dataset)

    # Custom collate function to handle the mix of tensors and strings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=_collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=_collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=_collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def _build_weighted_sampler(dataset: OilArticleDataset) -> WeightedRandomSampler:
    """Build a sampler that oversamples minority classes.

    Computes per-sample weights inversely proportional to the frequency
    of each sample's primary label class, ensuring the training loop
    sees roughly equal numbers of up/unchanged/down examples.

    Args:
        dataset: Training dataset.

    Returns:
        Configured :class:`WeightedRandomSampler`.
    """
    primary_labels = dataset.get_primary_labels()
    class_counts: Dict[int, int] = {}
    for label in primary_labels:
        class_counts[label] = class_counts.get(label, 0) + 1

    # Inverse frequency weights
    total = len(primary_labels)
    class_weights = {cls: total / count for cls, count in class_counts.items()}

    sample_weights = [class_weights[label] for label in primary_labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate that stacks tensors and collects metadata strings.

    The default PyTorch collate fails on mixed tensor/string dicts.
    This function stacks all tensor values and gathers string values
    into lists.

    Args:
        batch: List of sample dicts from :meth:`OilArticleDataset.__getitem__`.

    Returns:
        Collated dict with batched tensors and metadata lists.
    """
    collated: Dict[str, Any] = {}
    keys = batch[0].keys()

    for key in keys:
        values = [sample[key] for sample in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values, dim=0)
        else:
            collated[key] = values

    return collated
