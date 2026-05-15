"""
Sliding-window dataset builder: news embeddings → (X, y).
=========================================================

Reads **read-only** HTML from ``raw_articles/<MM_DD_YYYY>/article_*.html``,
extracts text via :class:`~ml_model.data.html_extractor.HTMLArticleExtractor`,
embeds each article with **frozen FinBERT** (no gradients), aggregates to
per-day vectors, and stacks ``window_days`` days into each sample.

Worked example (``window_days=5``, ``gap_days=0``)
--------------------------------------------------
Suppose ``prediction_date`` = **2026-05-06** and a price label exists that day.

::

    window_dates = [2026-05-01, 2026-05-02, 2026-05-03, 2026-05-04, 2026-05-05]
    prediction_date = 2026-05-06
    label = price_df.loc[2026-05-06, "label"]   # direction of that day's log return

For each day in ``window_dates``:

1. Load all ``article_*.html`` under ``raw_articles/05_01_2026/``, etc.
2. Embed each article → (768,) mean-pooled hidden states.
3. ``day_embedding = mean(article embeddings)`` or zeros if no articles.
4. ``sample_x = stack(day_embeddings)`` → shape ``(5, 768)``.
5. ``sample_y = label`` (scalar 0/1/2).

With ``gap_days=1``, the window ends at **2026-05-04** (two calendar days before
prediction), simulating a one-day publication lag.

Caching
-------
Day-level embeddings are stored at ``config.embed_cache_path`` as
``{ "YYYY-MM-DD": Tensor(768) }``.  Reruns skip FinBERT for cached dates.

Public API
----------
:func:`build_windows` — main entry; returns ``(X, y)`` tensors plus metadata via
:func:`build_windows_with_metadata` for evaluation backtests.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from ..pipeline_config import PipelineConfig
from .html_extractor import HTMLArticleExtractor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frozen FinBERT — loaded once; never trained.
# ---------------------------------------------------------------------------
_TOKENIZER = None
_MODEL = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_finbert(model_name: str):
    """Lazy-load tokenizer and model in eval mode on the chosen device.

    Args:
        model_name: HuggingFace model id (e.g. ``ProsusAI/finbert``).

    Returns:
        Tuple ``(tokenizer, model)``.
    """
    global _TOKENIZER, _MODEL
    if _TOKENIZER is None or _MODEL is None:
        logger.info("Loading frozen embedding model: %s", model_name)
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _MODEL = AutoModel.from_pretrained(model_name)
        _MODEL.eval()
        _MODEL.to(_DEVICE)
        for param in _MODEL.parameters():
            param.requires_grad = False
    return _TOKENIZER, _MODEL


def _date_to_folder(d: datetime) -> str:
    """Map a datetime to ``raw_articles`` subfolder name ``MM_DD_YYYY``.

    Args:
        d: Calendar date.

    Returns:
        Folder name string, e.g. ``"05_06_2026"``.
    """
    return d.strftime("%m_%d_%Y")


def _parse_folder_date(folder_name: str) -> Optional[datetime]:
    """Parse ``MM_DD_YYYY`` folder name to datetime at midnight.

    Args:
        folder_name: Basename of a date subfolder.

    Returns:
        Datetime or ``None`` if the pattern does not match.
    """
    try:
        return datetime.strptime(folder_name, "%m_%d_%Y")
    except ValueError:
        return None


def _window_dates(prediction_date: pd.Timestamp, config: PipelineConfig) -> List[pd.Timestamp]:
    """Compute ordered list of news days for one sample.

    Args:
        prediction_date: Day whose price label is predicted.
        config: Window and gap parameters.

    Returns:
        List of length ``window_days`` ending at ``prediction_date - gap_days - 1``.
    """
    pred = pd.Timestamp(prediction_date).normalize()
    # Last news day is gap_days before prediction (exclusive of prediction day).
    last_news = pred - timedelta(days=config.gap_days + 1)
    first_news = pred - timedelta(days=config.window_days + config.gap_days)
    dates = []
    current = first_news
    while current <= last_news:
        dates.append(pd.Timestamp(current).normalize())
        current += timedelta(days=1)
    # Enforce exact window length (calendar days may vary if logic off by one).
    if len(dates) > config.window_days:
        dates = dates[-config.window_days :]
    elif len(dates) < config.window_days:
        # Pad from the front with earlier days if needed.
        while len(dates) < config.window_days:
            dates.insert(0, dates[0] - timedelta(days=1))
    return dates


def _load_day_article_paths(raw_root: Path, day: pd.Timestamp) -> List[Path]:
    """List ``article_*.html`` paths for one calendar day folder.

    Args:
        raw_root: ``raw_articles`` directory.
        day: Target calendar day.

    Returns:
        Sorted list of HTML paths (may be empty).
    """
    folder = raw_root / _date_to_folder(day.to_pydatetime())
    if not folder.is_dir():
        return []
    return sorted(folder.glob("article_*.html"))


def _embed_text(text: str, config: PipelineConfig) -> Tensor:
    """Embed one article string → (embed_dim,) mean-pooled FinBERT vector.

    Args:
        text: Plain article body.
        config: Token length limit and model name.

    Returns:
        1-D float tensor of shape ``(embed_dim,)``.

    Note:
        Runs under ``torch.no_grad()``; FinBERT weights are frozen.
    """
    tokenizer, model = _get_finbert(config.embedding_model)
    encoded = tokenizer(
        text,
        max_length=config.max_tokens_per_article,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(_DEVICE)
    attention_mask = encoded["attention_mask"].to(_DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state: (1, seq_len, hidden)
        hidden = outputs.last_hidden_state.squeeze(0)
        mask = attention_mask.squeeze(0).bool()
        # Mean pool only over non-padding tokens.
        if mask.any():
            pooled = hidden[mask].mean(dim=0)
        else:
            pooled = hidden.mean(dim=0)
    return pooled.cpu()


def _day_embedding(
    day: pd.Timestamp,
    raw_root: Path,
    extractor: HTMLArticleExtractor,
    config: PipelineConfig,
    cache: Dict[str, Tensor],
) -> Tuple[Tensor, List[str], bool]:
    """Return cached or freshly computed day-level embedding.

    Args:
        day: Calendar day.
        raw_root: Read-only HTML root.
        extractor: Shared HTML text extractor.
        config: Pipeline limits.
        cache: Mutable day-string → vector cache (updated in place).

    Returns:
        Tuple of ``(embedding_768, list_of_filenames, was_zero_padded)``.
        ``was_zero_padded`` is True when no articles were available.
    """
    key = day.strftime("%Y-%m-%d")
    if key in cache:
        return cache[key], [], False

    paths = _load_day_article_paths(raw_root, day)
    filenames: List[str] = []
    vectors: List[Tensor] = []

    # Respect max articles per day — take first N after sort for reproducibility.
    for path in paths[: config.max_articles_per_day]:
        result = extractor.extract(path)
        if result and result.get("text"):
            vec = _embed_text(result["text"], config)
            vectors.append(vec)
            filenames.append(path.name)

    if vectors:
        day_vec = torch.stack(vectors, dim=0).mean(dim=0)
        zero_padded = False
    else:
        # No news that day: use zero vector so LSTM still receives fixed shape.
        day_vec = torch.zeros(config.embed_dim)
        zero_padded = True

    cache[key] = day_vec
    return day_vec, filenames, zero_padded


def _load_embed_cache(path: Path) -> Dict[str, Tensor]:
    """Load embedding cache dict from disk if present.

    Args:
        path: ``.pt`` file path.

    Returns:
        ``date_str → Tensor`` mapping (empty dict if missing).
    """
    if path.exists():
        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            logger.info("Loaded embedding cache: %d days from %s", len(data), path)
            return data
    return {}


def _save_embed_cache(path: Path, cache: Dict[str, Tensor]) -> None:
    """Persist embedding cache to disk.

    Args:
        path: Destination ``.pt`` path.
        cache: Day-string to vector mapping.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    logger.info("Saved embedding cache: %d days → %s", len(cache), path)


def build_windows(
    config: PipelineConfig,
    price_df: pd.DataFrame,
) -> Tuple[Tensor, Tensor]:
    """Build all sliding-window samples aligned to price labels.

    Iterates over every index date in ``price_df`` where a full news window
    can be formed and articles/embeddings assembled.  Samples are sorted
    chronologically by ``prediction_date``.

    Args:
        config: Full pipeline configuration.
        price_df: Output of :func:`~ml_model.data.price_fetcher.get_price_labels`.

    Returns:
        * **X** — ``FloatTensor`` of shape ``(N, window_days, embed_dim)``
        * **y** — ``LongTensor`` of shape ``(N,)`` with values in ``{0,1,2}``

    Raises:
        ValueError: If no samples could be constructed.
    """
    X, y, _meta = build_windows_with_metadata(config, price_df)
    return X, y


def build_windows_with_metadata(
    config: PipelineConfig,
    price_df: pd.DataFrame,
) -> Tuple[Tensor, Tensor, List[Dict[str, Any]]]:
    """Like :func:`build_windows` but also returns per-sample metadata.

    Args:
        config: Pipeline configuration.
        price_df: Labelled price DataFrame indexed by date.

    Returns:
        Tuple ``(X, y, metadata)`` where each metadata dict contains
        ``prediction_date``, ``window_dates``, ``article_filenames`` per day,
        and ``zero_padded_days`` count for that sample.

    Raises:
        ValueError: If zero samples are built.
    """
    raw_root = config.raw_articles_path
    extractor = HTMLArticleExtractor(
        primary_backend="trafilatura",
        min_chars=100,
        include_metadata_header=True,
        parsed_articles_dir=config.resolve_path(config.parsed_articles_dir),
    )

    cache = _load_embed_cache(config.embed_cache_file)
    samples_x: List[Tensor] = []
    samples_y: List[int] = []
    metadata: List[Dict[str, Any]] = []
    zero_day_count = 0

    price_index = pd.to_datetime(price_df.index).normalize()

    for pred_date in sorted(price_index):
        if pred_date not in price_df.index:
            continue
        wdates = _window_dates(pred_date, config)
        day_tensors: List[Tensor] = []
        day_files: Dict[str, List[str]] = {}
        sample_zero_days = 0

        for wd in wdates:
            vec, files, zp = _day_embedding(wd, raw_root, extractor, config, cache)
            day_tensors.append(vec)
            day_files[wd.strftime("%Y-%m-%d")] = files
            if zp:
                sample_zero_days += 1
                zero_day_count += 1

        if len(day_tensors) != config.window_days:
            continue

        label = int(price_df.loc[pred_date, "label"])
        samples_x.append(torch.stack(day_tensors, dim=0))
        samples_y.append(label)
        metadata.append(
            {
                "prediction_date": pred_date.strftime("%Y-%m-%d"),
                "window_dates": [d.strftime("%Y-%m-%d") for d in wdates],
                "article_filenames": day_files,
                "zero_padded_days": sample_zero_days,
            }
        )

    _save_embed_cache(config.embed_cache_file, cache)

    if not samples_x:
        raise ValueError(
            "No window samples built — check raw_articles date folders and price range."
        )

    X = torch.stack(samples_x, dim=0).float()
    y = torch.tensor(samples_y, dtype=torch.long)

    # Class distribution logging.
    unique, counts = torch.unique(y, return_counts=True)
    dist = {int(u.item()): int(c.item()) for u, c in zip(unique, counts)}
    logger.info("Built %d window samples; class distribution: %s", len(y), dist)
    logger.info("Total zero-padded (no article) days across samples: %d", zero_day_count)

    return X, y, metadata


def build_single_window(
    config: PipelineConfig,
    end_date: str,
    price_df: Optional[pd.DataFrame] = None,
) -> Tuple[Tensor, Optional[int], Dict[str, Any]]:
    """Build one window for inference ending at ``end_date`` (prediction day).

    Args:
        config: Pipeline configuration.
        end_date: ``YYYY-MM-DD`` prediction date.
        price_df: Optional labelled prices for ground-truth lookup.

    Returns:
        * **X** — ``(1, window_days, embed_dim)``
        * **y** — integer label if available in ``price_df``, else ``None``
        * **metadata** — window dates and article filenames
    """
    pred = pd.Timestamp(end_date).normalize()
    wdates = _window_dates(pred, config)
    raw_root = config.raw_articles_path
    extractor = HTMLArticleExtractor(
        min_chars=100,
        parsed_articles_dir=config.resolve_path(config.parsed_articles_dir),
    )
    cache = _load_embed_cache(config.embed_cache_file)
    day_tensors: List[Tensor] = []
    day_files: Dict[str, List[str]] = {}

    for wd in wdates:
        vec, files, _ = _day_embedding(wd, raw_root, extractor, config, cache)
        day_tensors.append(vec)
        day_files[wd.strftime("%Y-%m-%d")] = files

    _save_embed_cache(config.embed_cache_file, cache)
    X = torch.stack(day_tensors, dim=0).unsqueeze(0).float()

    label: Optional[int] = None
    if price_df is not None:
        pred_norm = pd.Timestamp(pred).normalize()
        if pred_norm in price_df.index:
            label = int(price_df.loc[pred_norm, "label"])

    meta = {
        "prediction_date": pred.strftime("%Y-%m-%d"),
        "window_dates": [d.strftime("%Y-%m-%d") for d in wdates],
        "article_filenames": day_files,
    }
    return X, label, meta
