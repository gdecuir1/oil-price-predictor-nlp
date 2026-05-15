"""
Pipeline configuration for the LSTM oil-direction predictor.
=============================================================

v3 defaults (recommended): compact LSTM, FinBERT+keyword inputs, binary
Up/Down labels (Flat days dropped), narrower flat band when using ternary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PipelineConfig:
    """All configurable parameters for the LSTM news → direction pipeline."""

    window_days: int = 5
    gap_days: int = 0
    horizon_days: int = 1

    # label_mode: "binary" keeps only Up/Down (drops Flat days from samples).
    # label_mode: "ternary" uses Down/Flat/Up with flat_band_pct.
    label_mode: str = "binary"
    flat_band_pct: float = 0.35
    price_ticker: str = "USO"
    price_start: str = "2025-04-28"
    price_end: str = "2026-05-13"

    embedding_model: str = "ProsusAI/finbert"
    finbert_dim: int = 768
    use_keywords: bool = True
    max_articles_per_day: int = 20
    max_tokens_per_article: int = 256
    article_pooling: str = "cls"

    # Compact LSTM (fits ~200 training windows).
    proj_dim: int = 256
    lstm_hidden: int = 128
    lstm_layers: int = 2
    mlp_hidden: int = 48
    dropout: float = 0.35
    bidirectional: bool = True
    use_residual: bool = True

    train_val_test_split: Tuple[float, float, float] = (0.80, 0.10, 0.10)
    batch_size: int = 16
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-3
    patience: int = 12
    # Class balance: "none" | "sqrt" (mild) | "full" (inverse freq).
    class_weight_mode: str = "sqrt"
    use_class_weights: bool = True
    use_weighted_sampler: bool = True
    label_smoothing: float = 0.0
    max_grad_norm: float = 1.0
    normalize_features: bool = True
    normalize_finbert_only: bool = True
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    min_train_epochs: int = 25
    # val_min_recall | val_macro_f1 | val_balanced_accuracy | val_loss
    early_stopping_metric: str = "val_min_recall"
    # Stop if either class has 0 val recall for this many epochs in a row.
    zero_up_recall_patience: int = 8
    zero_down_recall_patience: int = 8
    # Only save / early-stop on epochs that predict both classes on val.
    reject_collapsed_val_predictions: bool = True

    # Post-training: tune P(Up) cutoff on validation (binary only).
    tune_up_threshold: bool = True
    threshold_tuning_metric: str = "balanced_accuracy"  # or macro_f1
    # Threshold must achieve at least this recall on each class on val (if possible).
    threshold_min_class_recall: float = 0.15
    up_probability_threshold: Optional[float] = None  # set after val tuning; stored in checkpoint

    checkpoint_dir: str = "ml_model/outputs/checkpoints"
    report_dir: str = "ml_model/outputs/reports"
    embed_cache_path: str = "ml_model/outputs/embed_cache_v3.pt"
    price_cache_path: str = "ml_model/outputs/price_cache.csv"

    raw_articles_dir: str = "raw_articles"
    parsed_articles_dir: str = "parsed_articles"

    @property
    def keyword_dim(self) -> int:
        """Keyword feature size (0 when disabled)."""
        if not self.use_keywords:
            return 0
        from .data.keyword_extractor import KEYWORD_DIM
        return KEYWORD_DIM

    @property
    def input_dim(self) -> int:
        """Per-day vector width: FinBERT + optional keywords."""
        return self.finbert_dim + self.keyword_dim

    # Back-compat alias used in older code paths.
    @property
    def embed_dim(self) -> int:
        return self.input_dim

    @property
    def num_classes(self) -> int:
        return 2 if self.label_mode == "binary" else 3

    def resolve_path(self, path_str: str) -> Path:
        p = Path(path_str)
        return p if p.is_absolute() else _PROJECT_ROOT / p

    @property
    def checkpoint_path(self) -> Path:
        return self.resolve_path(self.checkpoint_dir)

    @property
    def report_path(self) -> Path:
        return self.resolve_path(self.report_dir)

    @property
    def embed_cache_file(self) -> Path:
        return self.resolve_path(self.embed_cache_path)

    @property
    def price_cache_file(self) -> Path:
        return self.resolve_path(self.price_cache_path)

    @property
    def raw_articles_path(self) -> Path:
        return self.resolve_path(self.raw_articles_dir)

    @property
    def lstm_output_dim(self) -> int:
        mult = 2 if self.bidirectional else 1
        return self.lstm_hidden * mult

    @property
    def class_names(self) -> Tuple[str, ...]:
        if self.label_mode == "binary":
            return ("Down", "Up")
        return ("Down", "Flat", "Up")
