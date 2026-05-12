#!/usr/bin/env python3
"""
Training Entry Point
=====================

Main script to train the Oil Market Prediction Transformer.  Orchestrates
the full pipeline:

    1. Parse command-line arguments and load configuration.
    2. Set random seeds for reproducibility.
    3. Extract text from raw HTML articles.
    4. Tokenise and build PyTorch DataLoaders.
    5. Construct the transformer model.
    6. Train with validation, checkpointing, and early stopping.
    7. Evaluate on the held-out test set.
    8. Save the training summary.

Usage::

    # Train with all defaults (FinBERT fine-tuning)
    python -m ml_model.run_training

    # Train from scratch with custom parameters
    python -m ml_model.run_training \\
        --no-pretrained \\
        --epochs 100 \\
        --batch-size 16 \\
        --learning-rate 1e-4

    # Use custom article directory
    python -m ml_model.run_training \\
        --articles-dir /path/to/raw_articles
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on the Python path so imports work
# when running as `python -m ml_model.run_training` from the project root
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ml_model.config import Config, DataConfig, ModelConfig, TrainingConfig
from ml_model.data.html_extractor import HTMLArticleExtractor
from ml_model.data.preprocessor import TextPreprocessor
from ml_model.data.dataset import create_data_loaders
from ml_model.model import OilMarketTransformer
from ml_model.training.trainer import Trainer
from ml_model.utils.logger import setup_logger
from ml_model.utils.helpers import set_seed, count_parameters, format_parameters, Timer

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training configuration.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train the Oil Market Prediction Transformer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--articles-dir", type=Path, default=None,
        help="Path to raw_articles HTML directory. Defaults to config.",
    )
    data_group.add_argument(
        "--parsed-dir", type=Path, default=None,
        help="Path to parsed_articles JSON directory.",
    )
    data_group.add_argument(
        "--label-file", type=Path, default=None,
        help="Path to external label file (JSON). Omit for pseudo-labels.",
    )
    data_group.add_argument(
        "--max-seq-length", type=int, default=512,
        help="Maximum token sequence length.",
    )

    # Model arguments
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--no-pretrained", action="store_true",
        help="Train from scratch instead of fine-tuning a pre-trained model.",
    )
    model_group.add_argument(
        "--pretrained-model", type=str, default="ProsusAI/finbert",
        help="HuggingFace model name for the pre-trained encoder.",
    )
    model_group.add_argument(
        "--pool-strategy", type=str, default="cls",
        choices=["cls", "mean", "attention_pool"],
        help="Sequence pooling strategy.",
    )
    model_group.add_argument(
        "--classifier-dropout", type=float, default=0.3,
        help="Dropout in classification heads.",
    )

    # Training arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--epochs", type=int, default=50,
        help="Maximum training epochs.",
    )
    train_group.add_argument(
        "--batch-size", type=int, default=8,
        help="Training batch size.",
    )
    train_group.add_argument(
        "--learning-rate", type=float, default=2e-5,
        help="Encoder learning rate.",
    )
    train_group.add_argument(
        "--head-lr", type=float, default=1e-3,
        help="Classification head learning rate.",
    )
    train_group.add_argument(
        "--weight-decay", type=float, default=0.01,
        help="L2 regularisation weight.",
    )
    train_group.add_argument(
        "--scheduler", type=str, default="cosine",
        choices=["cosine", "linear", "plateau"],
        help="Learning rate scheduler.",
    )
    train_group.add_argument(
        "--patience", type=int, default=7,
        help="Early stopping patience (epochs).",
    )
    train_group.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.",
    )
    train_group.add_argument(
        "--no-amp", action="store_true",
        help="Disable automatic mixed-precision training.",
    )
    train_group.add_argument(
        "--freeze-epochs", type=int, default=2,
        help="Epochs to freeze encoder during warm-up.",
    )

    return parser.parse_args()


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    # --- Logging ---
    setup_logger(level=logging.INFO)
    logger.info("=" * 70)
    logger.info("OIL MARKET PREDICTION TRANSFORMER — TRAINING")
    logger.info("=" * 70)

    # --- Configuration ---
    data_config = DataConfig(
        raw_articles_dir=args.articles_dir or DataConfig().raw_articles_dir,
        parsed_articles_dir=args.parsed_dir or DataConfig().parsed_articles_dir,
        max_sequence_length=args.max_seq_length,
    )
    model_config = ModelConfig(
        use_pretrained=not args.no_pretrained,
        pretrained_model_name=args.pretrained_model,
        pool_strategy=args.pool_strategy,
        classifier_dropout=args.classifier_dropout,
        freeze_encoder_epochs=args.freeze_epochs,
    )
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        head_learning_rate=args.head_lr,
        weight_decay=args.weight_decay,
        lr_scheduler=args.scheduler,
        early_stopping_patience=args.patience,
        seed=args.seed,
        use_amp=not args.no_amp,
    )

    config = Config(
        data=data_config,
        model=model_config,
        training=training_config,
    )

    # --- Reproducibility ---
    set_seed(config.training.seed)
    logger.info("Random seed set to %d", config.training.seed)

    # --- Data extraction ---
    with Timer("HTML article extraction", logger):
        extractor = HTMLArticleExtractor(
            primary_backend=config.data.extraction_backend,
            min_chars=config.data.min_article_chars,
            include_metadata_header=True,
            parsed_articles_dir=config.data.parsed_articles_dir,
        )
        articles = extractor.extract_all(config.data.raw_articles_dir)

    if not articles:
        logger.error("No articles extracted! Check raw_articles directory.")
        sys.exit(1)

    logger.info("Extracted %d articles for training", len(articles))

    # --- Preprocessing and data loaders ---
    with Timer("data preprocessing", logger):
        preprocessor = TextPreprocessor(
            tokenizer_name=config.data.tokenizer_name,
            max_length=config.data.max_sequence_length,
            augment=True,  # enable augmentation for training
        )
        train_loader, val_loader, test_loader = create_data_loaders(
            articles=articles,
            preprocessor=preprocessor,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            batch_size=config.training.batch_size,
            num_workers=config.data.num_workers,
            label_file=args.label_file,
            seed=config.training.seed,
        )

    logger.info(
        "Data loaders ready: %d train batches, %d val batches, %d test batches",
        len(train_loader), len(val_loader), len(test_loader),
    )

    # --- Model construction ---
    with Timer("model construction", logger):
        model = OilMarketTransformer(config)

    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    logger.info(
        "Model: %s total params, %s trainable",
        format_parameters(total_params),
        format_parameters(trainable_params),
    )

    # --- Training ---
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    with Timer("full training", logger):
        summary = trainer.train()

    # --- Final summary ---
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("  Best primary F1:  %.4f", summary["best_primary_f1"])
    logger.info("  Total epochs:     %d", summary["total_epochs"])
    logger.info("  History saved:    %s", summary["history_path"])

    if summary.get("test_results"):
        from ml_model.training.metrics import MetricsCalculator
        calc = MetricsCalculator(
            subdomain_keys=config.all_subdomain_keys,
            label_names={k: list(v.labels) for k, v in config.subdomains.items()},
        )
        logger.info("\n" + calc.format_summary(summary["test_results"]))

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
