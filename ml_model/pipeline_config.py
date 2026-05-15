"""
Pipeline configuration for the LSTM oil-direction predictor.
=============================================================

This module is the **single source of truth** for every hyperparameter,
path, and behavioral knob in the LSTM pipeline.  No other file in the
LSTM stack should hard-code numbers; import :class:`PipelineConfig` instead.

How this fits the pipeline
--------------------------
1. :mod:`ml_model.data.price_fetcher` reads ``price_ticker``, date range, and
   ``flat_band_pct`` to build 3-class labels from market closes.
2. :mod:`ml_model.data.window_builder` reads window geometry, embedding limits,
   and cache paths to turn ``raw_articles/`` HTML into tensors ``(N, T, 768)``.
3. :mod:`ml_model.model_lstm`, :mod:`ml_model.train_lstm`, :mod:`ml_model.evaluate_lstm`,
   and :mod:`ml_model.predict_lstm`` consume architecture and training fields.

Temporal semantics (window + gap + horizon)
-------------------------------------------
* **window_days** — How many consecutive *calendar* days of news embeddings
  are stacked into one input sample.  Larger values give more history but
  shrink the number of valid samples and increase compute.
* **gap_days** — Empty days between the last news day in the window and the
  **prediction_date**.  ``gap_days=0`` means the window ends the day before
  prediction; ``gap_days=1`` skips one day (simulating delayed article availability).
* **horizon_days** — Reserved for multi-day-ahead targets; with the current
  label definition (same-day log return on ``prediction_date``), keep at ``1``
  unless you extend ``price_fetcher`` to label ``d + horizon``.

Labeling and market data
------------------------
* **flat_band_pct** — Half-width of the "unchanged" zone in percent.  Log returns
  inside ``[-band, +band]`` become class 1 (Flat).  Wider band → more Flat labels.
* **price_ticker** — yfinance symbol (default ``USO``, US Oil Fund ETF proxy).

Embeddings (frozen FinBERT)
---------------------------
* **embedding_model**, **embed_dim**, **max_tokens_per_article**,
  **max_articles_per_day** — Control text → vector conversion cost and quality.
  Lower caps speed up first-run embedding; cached days are free on rerun.

Model and training
------------------
* **lstm_hidden**, **lstm_layers**, **dropout**, **bidirectional** — LSTM capacity.
  Smaller hidden size and ``lstm_layers=1`` are the first levers if CPU runtime
  exceeds ~30 minutes.
* **train_val_test_split** — Chronological fractions; never shuffle time series
  before splitting.
* **batch_size**, **epochs**, **lr**, **weight_decay**, **patience** — Standard
  training schedule with early stopping on validation loss.
* **use_class_weights** — Balances loss when Up/Flat/Down counts differ.

Paths (relative to project root unless absolute)
------------------------------------------------
Checkpoints, reports, embedding cache, and price CSV cache live under
``ml_model/outputs/``.  Directories are created automatically by training scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

# Project root is parent of the ml_model package.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PipelineConfig:
    """All configurable parameters for the LSTM news → direction pipeline.

    Instantiate once, optionally override fields from CLI, then pass the
    instance through ``get_price_labels``, ``build_windows``, and training.
    """

    # --- Temporal window ---
    window_days: int = 5
    gap_days: int = 0
    horizon_days: int = 1

    # --- Price labels ---
    flat_band_pct: float = 0.5
    price_ticker: str = "USO"
    price_start: str = "2025-04-28"
    price_end: str = "2026-05-13"

    # --- Frozen text encoder ---
    embedding_model: str = "ProsusAI/finbert"
    embed_dim: int = 768
    max_articles_per_day: int = 10
    max_tokens_per_article: int = 256

    # --- LSTM classifier ---
    lstm_hidden: int = 128
    lstm_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True

    # --- Data split & training ---
    train_val_test_split: Tuple[float, float, float] = (0.80, 0.10, 0.10)
    batch_size: int = 16
    epochs: int = 30
    lr: float = 2e-4
    weight_decay: float = 1e-4
    patience: int = 6
    use_class_weights: bool = True

    # --- Paths (strings for easy JSON serialisation) ---
    checkpoint_dir: str = "ml_model/outputs/checkpoints"
    report_dir: str = "ml_model/outputs/reports"
    embed_cache_path: str = "ml_model/outputs/embed_cache.pt"
    price_cache_path: str = "ml_model/outputs/price_cache.csv"

    # --- Read-only data roots (never written) ---
    raw_articles_dir: str = "raw_articles"
    parsed_articles_dir: str = "parsed_articles"

    def resolve_path(self, path_str: str) -> Path:
        """Return an absolute :class:`~pathlib.Path` for a config path string.

        Args:
            path_str: Relative path from project root or already absolute.

        Returns:
            Resolved absolute path.
        """
        p = Path(path_str)
        if p.is_absolute():
            return p
        return _PROJECT_ROOT / p

    @property
    def checkpoint_path(self) -> Path:
        """Absolute checkpoint directory."""
        return self.resolve_path(self.checkpoint_dir)

    @property
    def report_path(self) -> Path:
        """Absolute report output directory."""
        return self.resolve_path(self.report_dir)

    @property
    def embed_cache_file(self) -> Path:
        """Absolute embedding cache file."""
        return self.resolve_path(self.embed_cache_path)

    @property
    def price_cache_file(self) -> Path:
        """Absolute price CSV cache file."""
        return self.resolve_path(self.price_cache_path)

    @property
    def raw_articles_path(self) -> Path:
        """Absolute path to read-only HTML corpus."""
        return self.resolve_path(self.raw_articles_dir)

    @property
    def lstm_output_dim(self) -> int:
        """LSTM sequence vector size after bidirectional concat."""
        mult = 2 if self.bidirectional else 1
        return self.lstm_hidden * mult
