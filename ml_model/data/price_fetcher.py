"""
Daily oil-price labels for supervised LSTM training.
====================================================

This module is the **sole source of ground-truth labels** for the LSTM pipeline.
It downloads (or loads from cache) daily closing prices for a configurable
ticker, computes **log returns**, and maps each day to a 3-class direction:

* **0 — Down**  : log return < ``-flat_band_pct / 100``
* **1 — Flat**  : log return within the symmetric band (inclusive)
* **2 — Up**    : log return > ``+flat_band_pct / 100``

Log return definition
---------------------
For trading day *t* with close price ``C_t``:

    r_t = ln(C_t / C_{t-1})

Log returns are additive across time, symmetric around zero for small moves,
and standard in quantitative finance.  They measure *relative* price change,
not dollar change.

Why a flat band instead of a binary threshold?
----------------------------------------------
A binary rule (any positive return = Up) labels tiny noise (e.g. +0.01%) as
"Up", inflating class imbalance and teaching the model to chase noise.  The
**flat_band_pct** parameter defines a dead zone: moves smaller than the band
are **Flat** (class 1).  Widening the band increases Flat samples and makes
Up/Down more "decisive" moves only.

Public API
----------
:func:`get_price_labels` — single entry point used by training and evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from ..pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)

# Human-readable names aligned with class integers 0, 1, 2.
_LABEL_STR = {0: "Down", 1: "Flat", 2: "Up"}


def get_price_labels(config: PipelineConfig) -> pd.DataFrame:
    """Fetch daily prices and assign 3-class direction labels.

    Attempts a live download via yfinance for ``config.price_ticker`` between
    ``config.price_start`` and ``config.price_end``.  On success, writes
    ``config.price_cache_path`` as CSV for offline reuse.  On network failure,
    loads the cached CSV if it exists; otherwise raises.

    Args:
        config: Pipeline configuration with ticker, date range, and flat band.

    Returns:
        DataFrame indexed by calendar date (``datetime64[ns]``) with columns:

        * **date** — same as index, normalized to midnight.
        * **close** — adjusted close (or close if adjusted unavailable).
        * **log_return** — ``ln(close_t / close_{t-1})``; NaN on first row.
        * **label** — integer class 0/1/2.
        * **label_str** — ``"Down"`` / ``"Flat"`` / ``"Up"``.

    Raises:
        RuntimeError: If download fails and no cache file is present.
        ValueError: If the downloaded series is empty.

    Note:
        Class on day *t* describes the move **into** that day's close from the
        prior session — the label used when *t* is the ``prediction_date`` in
        :mod:`window_builder`.
    """
    cache_path = config.price_cache_file
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    df = _download_or_load(config, cache_path)
    df = _compute_log_returns(df)
    df = _assign_labels(df, config.flat_band_pct)

    # Log class distribution for quick sanity check.
    counts = df["label"].value_counts().sort_index()
    logger.info(
        "Price labels (%s): %s",
        config.price_ticker,
        {int(k): int(v) for k, v in counts.items()},
    )
    return df


def _download_or_load(config: PipelineConfig, cache_path: Path) -> pd.DataFrame:
    """Download OHLCV from yfinance or read CSV cache.

    Args:
        config: Pipeline config with ticker and date strings.
        cache_path: Destination / fallback CSV path.

    Returns:
        DataFrame with DatetimeIndex and at least a ``close`` column.

    Raises:
        RuntimeError: When download fails and cache is missing.
        ValueError: When result is empty.
    """
    try:
        logger.info(
            "Downloading %s from %s to %s via yfinance",
            config.price_ticker,
            config.price_start,
            config.price_end,
        )
        ticker = yf.Ticker(config.price_ticker)
        raw = ticker.history(
            start=config.price_start,
            end=config.price_end,
            auto_adjust=True,
        )
        if raw is None or raw.empty:
            raise ValueError("yfinance returned no rows")

        # Prefer adjusted close when column exists.
        if "Close" in raw.columns:
            close = raw["Close"].astype(float)
        else:
            close = raw.iloc[:, 0].astype(float)

        out = pd.DataFrame({"close": close})
        out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
        out = out.sort_index()
        out.to_csv(cache_path, index_label="date")
        logger.info("Saved %d price rows to %s", len(out), cache_path)
        return out

    except Exception as exc:
        logger.warning("yfinance download failed (%s); trying cache", exc)
        if cache_path.exists():
            cached = pd.read_csv(cache_path, parse_dates=["date"], index_col="date")
            cached.index = pd.to_datetime(cached.index).normalize()
            logger.info("Loaded %d rows from cache %s", len(cached), cache_path)
            return cached[["close"]] if "close" in cached.columns else cached
        raise RuntimeError(
            f"Could not download {config.price_ticker} and no cache at {cache_path}"
        ) from exc


def _compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``log_return`` column as ln(close_t / close_{t-1}).

    Args:
        df: DataFrame with ``close`` column.

    Returns:
        Copy with ``log_return`` appended; first row is NaN.
    """
    out = df.copy()
    # Log return: natural log of price ratio vs previous trading day.
    out["log_return"] = np.log(out["close"] / out["close"].shift(1))
    out["date"] = out.index
    return out


def _assign_labels(df: pd.DataFrame, flat_band_pct: float) -> pd.DataFrame:
    """Map log returns to integer labels using symmetric percent band.

    Args:
        df: DataFrame with ``log_return`` column.
        flat_band_pct: Half-width of flat zone in **percent** (e.g. 0.5 → ±0.5%).

    Returns:
        DataFrame with ``label`` and ``label_str`` columns; rows with NaN
        log return are dropped.
    """
    out = df.copy()
    # Convert percent band to decimal threshold for log-return comparison.
    band = flat_band_pct / 100.0

    labels = np.full(len(out), 1, dtype=np.int64)  # default Flat
    lr = out["log_return"].values
    labels[lr > band] = 2   # Up
    labels[lr < -band] = 0  # Down
    # NaN log_return (first day) → keep as Flat then drop below.

    out["label"] = labels
    out["label_str"] = out["label"].map(_LABEL_STR)

    # Drop first row (no prior close) and any remaining NaNs.
    out = out.dropna(subset=["log_return"])
    return out
