"""
Centralized Configuration for the Oil Market Prediction Transformer
====================================================================

All hyperparameters, file paths, label definitions, and sub-domain
specifications live here so that every other module imports from a single
source of truth.  Values can be overridden at runtime via CLI flags or
environment variables where noted.

Design rationale
----------------
* **Dataclass-based** for type safety and IDE autocompletion.
* **Frozen after construction** to prevent accidental mutation mid-training.
* **Hierarchical** — separate dataclasses for data, model architecture,
  training schedule, and inference, composed into one top-level ``Config``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project root is one level above the ml_model package
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ML_MODEL_ROOT = Path(__file__).resolve().parent


# ===================================================================
# Sub-domain label definitions
# ===================================================================
# Each sub-domain is a secondary classification task that the model
# predicts alongside the primary market-direction label.  The keys
# used here are referenced throughout the codebase (loss weighting,
# metrics, report generation).
# ===================================================================

@dataclass(frozen=True)
class SubdomainSpec:
    """Specification for a single prediction sub-domain.

    Attributes:
        name: Human-readable name shown in reports.
        labels: Ordered list of class labels (index 0 = class 0, etc.).
        description: Explanatory text included in output reports.
        loss_weight: Relative weight of this sub-domain's loss contribution.
    """

    name: str
    labels: Tuple[str, ...]
    description: str
    loss_weight: float = 1.0


# Registry of all sub-domain prediction tasks.
# The primary task ("market_direction") is always present; the rest are
# auxiliary tasks that regularise the shared encoder and produce the
# comprehensive output report the user requested.
SUBDOMAIN_SPECS: Dict[str, SubdomainSpec] = {
    # ---- Primary output ----
    "market_direction": SubdomainSpec(
        name="Market Direction",
        labels=("up", "unchanged", "down"),
        description=(
            "Primary prediction: whether the US oil stock market will go up, "
            "remain effectively unchanged, or go down."
        ),
        loss_weight=3.0,  # emphasised — this is the main objective
    ),
    # ---- Auxiliary sub-domains ----
    "price_magnitude": SubdomainSpec(
        name="Price Movement Magnitude",
        labels=("negligible", "small", "moderate", "large", "extreme"),
        description=(
            "Expected magnitude of the price movement, from negligible "
            "(< 0.25 %) to extreme (> 5 %)."
        ),
        loss_weight=1.5,
    ),
    "timeframe": SubdomainSpec(
        name="Movement Timeframe",
        labels=("intraday", "short_term", "medium_term", "long_term"),
        description=(
            "Anticipated time horizon over which the predicted movement "
            "materialises: intraday (same day), short-term (1-5 days), "
            "medium-term (1-4 weeks), or long-term (1+ months)."
        ),
        loss_weight=1.0,
    ),
    "volatility": SubdomainSpec(
        name="Market Volatility",
        labels=("low", "normal", "high", "extreme"),
        description=(
            "Expected volatility in the oil market, reflecting how turbulent "
            "price action is likely to be regardless of direction."
        ),
        loss_weight=1.0,
    ),
    "sentiment": SubdomainSpec(
        name="Article Sentiment",
        labels=("very_negative", "negative", "neutral", "positive", "very_positive"),
        description=(
            "Overall sentiment of the input articles toward the oil market, "
            "ranging from very negative to very positive."
        ),
        loss_weight=1.2,
    ),
    "supply_impact": SubdomainSpec(
        name="Supply-Side Impact",
        labels=("decrease", "stable", "increase"),
        description=(
            "Whether the articles indicate oil supply will decrease, remain "
            "stable, or increase."
        ),
        loss_weight=1.0,
    ),
    "demand_impact": SubdomainSpec(
        name="Demand-Side Impact",
        labels=("decrease", "stable", "increase"),
        description=(
            "Whether the articles indicate oil demand will decrease, remain "
            "stable, or increase."
        ),
        loss_weight=1.0,
    ),
    "geopolitical_risk": SubdomainSpec(
        name="Geopolitical Risk Level",
        labels=("low", "moderate", "elevated", "high", "severe"),
        description=(
            "Assessed geopolitical risk that could affect oil markets, from "
            "low (stable environment) to severe (active conflict / sanctions)."
        ),
        loss_weight=1.0,
    ),
}


# ===================================================================
# Data configuration
# ===================================================================

@dataclass(frozen=True)
class DataConfig:
    """Parameters governing data loading, extraction, and splitting.

    Attributes:
        raw_articles_dir: Path to scraped full-article HTML files.
        parsed_articles_dir: Path to JSON metadata parsed from search results.
        max_sequence_length: Maximum number of tokens per article after
            truncation.  512 is the default BERT limit; increase if using
            a model with longer context.
        train_ratio: Proportion of data used for training.
        val_ratio: Proportion held out for validation (remainder is test).
        min_article_chars: Articles shorter than this (after HTML stripping)
            are discarded as low-quality / empty shells.
        tokenizer_name: HuggingFace tokenizer identifier.  Defaults to
            ``ProsusAI/finbert`` for domain-appropriate sub-word vocabulary.
        extraction_backend: Library used to pull clean text from raw HTML.
            Supported: ``trafilatura``, ``bs4``, ``readability``.
        num_workers: DataLoader worker processes.
        prefetch_factor: Batches prefetched per worker.
    """

    raw_articles_dir: Path = _PROJECT_ROOT / "raw_articles"
    parsed_articles_dir: Path = _PROJECT_ROOT / "parsed_articles"
    max_sequence_length: int = 512
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    min_article_chars: int = 100
    tokenizer_name: str = "ProsusAI/finbert"
    extraction_backend: str = "trafilatura"
    num_workers: int = 4
    prefetch_factor: int = 2


# ===================================================================
# Model architecture configuration
# ===================================================================

@dataclass(frozen=True)
class ModelConfig:
    """Transformer architecture hyper-parameters.

    The model supports two modes controlled by ``use_pretrained``:

    1. **Fine-tune mode** (default) — loads a pre-trained encoder
       (e.g. FinBERT) and attaches fresh multi-task classification heads.
    2. **From-scratch mode** — builds a custom transformer encoder
       from the parameters below with random initialisation.

    Attributes:
        use_pretrained: If True, load ``pretrained_model_name`` as the
            encoder backbone.  If False, build from scratch.
        pretrained_model_name: HuggingFace model identifier for the
            pre-trained encoder.  Only used when ``use_pretrained=True``.
        vocab_size: Vocabulary size for scratch-mode embeddings.
        d_model: Dimensionality of token embeddings / hidden states.
        n_heads: Number of attention heads.  Must divide ``d_model``.
        n_encoder_layers: Number of stacked transformer encoder blocks.
        d_feedforward: Inner dimension of the position-wise FFN.
        dropout: Dropout probability applied throughout the model.
        attention_dropout: Separate dropout for attention weights.
        activation: Activation function in the FFN (``gelu`` or ``relu``).
        layer_norm_eps: Epsilon for layer normalisation stability.
        max_position_embeddings: Maximum sequence length for learned
            positional embeddings (scratch mode).
        classifier_hidden_dim: Hidden dimension inside each classification
            head's MLP.
        classifier_dropout: Dropout in classification heads (can be higher
            than the encoder dropout to reduce overfitting on small data).
        pool_strategy: How to aggregate token-level representations into
            a single sequence vector.  Options: ``cls``, ``mean``, ``max``,
            ``attention_pool``.
        freeze_encoder_epochs: Number of initial epochs during which the
            pre-trained encoder weights are frozen (warm-up for heads only).
    """

    use_pretrained: bool = True
    pretrained_model_name: str = "ProsusAI/finbert"
    vocab_size: int = 30_522
    d_model: int = 768
    n_heads: int = 12
    n_encoder_layers: int = 12
    d_feedforward: int = 3072
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation: str = "gelu"
    layer_norm_eps: float = 1e-12
    max_position_embeddings: int = 512
    classifier_hidden_dim: int = 256
    classifier_dropout: float = 0.3
    pool_strategy: str = "cls"
    freeze_encoder_epochs: int = 2


# ===================================================================
# Training configuration
# ===================================================================

@dataclass(frozen=True)
class TrainingConfig:
    """Training schedule, optimiser settings, and regularisation.

    Attributes:
        epochs: Total training epochs.
        batch_size: Samples per gradient step.
        accumulation_steps: Gradient accumulation steps for effective
            batch size = ``batch_size * accumulation_steps``.
        learning_rate: Peak learning rate for the encoder.
        head_learning_rate: Peak learning rate for classification heads
            (typically higher than encoder LR when fine-tuning).
        weight_decay: L2 regularisation coefficient.
        warmup_ratio: Fraction of total steps used for linear LR warm-up.
        lr_scheduler: Scheduler type — ``cosine``, ``linear``, ``plateau``.
        max_grad_norm: Gradient clipping threshold.
        early_stopping_patience: Epochs without validation improvement
            before stopping.
        early_stopping_metric: Metric to monitor for early stopping.
        label_smoothing: Cross-entropy label smoothing factor.
        mixup_alpha: Alpha for Beta distribution in Mixup augmentation.
            Set to 0.0 to disable.
        checkpoint_dir: Where to save model checkpoints.
        save_top_k: Number of best checkpoints to keep.
        log_every_n_steps: Logging frequency (in training steps).
        eval_every_n_steps: Validation frequency.  ``None`` = once per epoch.
        seed: Global random seed for reproducibility.
        use_amp: Enable automatic mixed-precision (FP16) training.
    """

    epochs: int = 50
    batch_size: int = 8
    accumulation_steps: int = 4
    learning_rate: float = 2e-5
    head_learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler: str = "cosine"
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 7
    early_stopping_metric: str = "val_primary_f1"
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    checkpoint_dir: Path = _ML_MODEL_ROOT / "outputs" / "checkpoints"
    save_top_k: int = 3
    log_every_n_steps: int = 10
    eval_every_n_steps: Optional[int] = None
    seed: int = 42
    use_amp: bool = True


# ===================================================================
# Inference configuration
# ===================================================================

@dataclass(frozen=True)
class InferenceConfig:
    """Settings for the prediction and report-generation pipeline.

    Attributes:
        checkpoint_path: Path to the model checkpoint to load.
            ``None`` means the best checkpoint from the most recent run.
        report_output_dir: Directory for generated prediction reports.
        batch_size: Inference batch size (can be larger than training).
        generate_html_report: Produce a styled HTML report.
        generate_json_report: Produce a machine-readable JSON report.
        confidence_threshold: Minimum softmax probability to accept the
            primary prediction.  Below this, the report flags low confidence.
        ensemble_passes: Number of stochastic forward passes for MC-Dropout
            uncertainty estimation.  Set to 1 to disable.
        top_k_articles: Number of most-influential articles to highlight
            in the report (via attention-weight analysis).
    """

    checkpoint_path: Optional[Path] = None
    report_output_dir: Path = _ML_MODEL_ROOT / "outputs" / "reports"
    batch_size: int = 16
    generate_html_report: bool = True
    generate_json_report: bool = True
    confidence_threshold: float = 0.40
    ensemble_passes: int = 5
    top_k_articles: int = 10


# ===================================================================
# Top-level composed configuration
# ===================================================================

@dataclass(frozen=True)
class Config:
    """Root configuration object that composes all sub-configs.

    Typical usage::

        cfg = Config()                 # all defaults
        cfg = Config(                  # override selectively
            training=TrainingConfig(epochs=100, batch_size=16),
        )

    Attributes:
        data: Data loading and preprocessing settings.
        model: Transformer architecture parameters.
        training: Training schedule and regularisation.
        inference: Prediction and reporting settings.
        subdomains: Registry of sub-domain prediction tasks.
        project_root: Absolute path to the project root.
        ml_model_root: Absolute path to the ml_model package.
    """

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    subdomains: Dict[str, SubdomainSpec] = field(
        default_factory=lambda: dict(SUBDOMAIN_SPECS)
    )
    project_root: Path = _PROJECT_ROOT
    ml_model_root: Path = _ML_MODEL_ROOT

    def num_classes_for(self, subdomain_key: str) -> int:
        """Return the number of output classes for a given sub-domain.

        Args:
            subdomain_key: Key into ``self.subdomains``.

        Returns:
            Integer count of class labels.

        Raises:
            KeyError: If ``subdomain_key`` is not registered.
        """
        return len(self.subdomains[subdomain_key].labels)

    @property
    def all_subdomain_keys(self) -> List[str]:
        """Sorted list of all registered sub-domain keys."""
        return sorted(self.subdomains.keys())

    @property
    def total_output_classes(self) -> Dict[str, int]:
        """Mapping of sub-domain key → number of classes."""
        return {k: len(v.labels) for k, v in self.subdomains.items()}
