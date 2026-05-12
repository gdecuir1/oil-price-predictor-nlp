#!/usr/bin/env python3
"""
Inference Entry Point
======================

Main script to run predictions on raw HTML articles using a trained
Oil Market Prediction Transformer and generate comprehensive reports.

Usage::

    # Predict using the best checkpoint and generate reports
    python -m ml_model.run_inference

    # Specify a custom checkpoint and article directory
    python -m ml_model.run_inference \\
        --checkpoint outputs/checkpoints/best_model.pt \\
        --articles-dir /path/to/raw_articles

    # Increase MC-Dropout passes for better uncertainty estimates
    python -m ml_model.run_inference --ensemble-passes 10

    # Generate only JSON report (skip HTML)
    python -m ml_model.run_inference --no-html
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on the Python path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ml_model.config import Config, DataConfig, InferenceConfig
from ml_model.inference.predictor import OilMarketPredictor
from ml_model.inference.report_generator import ReportGenerator
from ml_model.utils.logger import setup_logger
from ml_model.utils.helpers import set_seed, Timer

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for inference.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run oil market predictions and generate reports",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Path to model checkpoint (.pt). Defaults to best_model.pt.",
    )
    parser.add_argument(
        "--articles-dir", type=Path, default=None,
        help="Path to raw_articles HTML directory.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for report output files.",
    )
    parser.add_argument(
        "--report-name", type=str, default=None,
        help="Base name for report files (without extension).",
    )
    parser.add_argument(
        "--ensemble-passes", type=int, default=5,
        help="Number of MC-Dropout forward passes for uncertainty.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--top-k-articles", type=int, default=10,
        help="Number of top articles to include in the report.",
    )
    parser.add_argument(
        "--no-html", action="store_true",
        help="Skip HTML report generation.",
    )
    parser.add_argument(
        "--no-json", action="store_true",
        help="Skip JSON report generation.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )

    return parser.parse_args()


def main() -> None:
    """Main inference entry point."""
    args = parse_args()

    # --- Logging ---
    setup_logger(level=logging.INFO)
    logger.info("=" * 70)
    logger.info("OIL MARKET PREDICTION TRANSFORMER — INFERENCE")
    logger.info("=" * 70)

    # --- Configuration ---
    set_seed(args.seed)

    # Resolve checkpoint path
    default_checkpoint = (
        Path(__file__).resolve().parent / "outputs" / "checkpoints" / "best_model.pt"
    )
    checkpoint_path = args.checkpoint or default_checkpoint

    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        logger.error("Train the model first with: python -m ml_model.run_training")
        sys.exit(1)

    # Build config with inference overrides
    inference_config = InferenceConfig(
        checkpoint_path=checkpoint_path,
        report_output_dir=args.output_dir or InferenceConfig().report_output_dir,
        batch_size=args.batch_size,
        generate_html_report=not args.no_html,
        generate_json_report=not args.no_json,
        ensemble_passes=args.ensemble_passes,
        top_k_articles=args.top_k_articles,
    )

    articles_dir = args.articles_dir or DataConfig().raw_articles_dir

    config = Config(inference=inference_config)

    # --- Load model ---
    with Timer("model loading", logger):
        predictor = OilMarketPredictor.from_checkpoint(
            checkpoint_path=checkpoint_path,
            config=config,
        )

    # --- Run predictions ---
    logger.info("Processing articles from: %s", articles_dir)

    with Timer("prediction pipeline", logger):
        result = predictor.predict_from_directory(articles_dir)

    # --- Display summary ---
    logger.info("")
    logger.info("=" * 50)
    logger.info("  PREDICTION SUMMARY")
    logger.info("=" * 50)
    logger.info("  Primary prediction:  %s", result.primary_prediction.upper())
    logger.info("  Confidence:          %.1f%%", result.primary_confidence * 100)
    logger.info("  Uncertainty (std):   %.4f", result.ensemble_std)
    logger.info("  Articles processed:  %d / %d",
                result.n_articles_processed, result.n_articles_total)
    logger.info("")

    logger.info("  Probability breakdown:")
    for label, prob in sorted(
        result.primary_probabilities.items(),
        key=lambda x: -x[1],
    ):
        bar = "█" * int(prob * 40)
        logger.info("    %-12s %5.1f%%  %s", label.upper(), prob * 100, bar)

    logger.info("")
    logger.info("  Sub-domain predictions:")
    for key, agg in sorted(result.subdomain_aggregations.items()):
        logger.info(
            "    %-22s → %-15s (conf: %.1f%%)",
            key, agg["prediction"].upper(), agg["confidence"] * 100,
        )
    logger.info("=" * 50)

    # --- Generate reports ---
    with Timer("report generation", logger):
        report_gen = ReportGenerator(
            config=config,
            output_dir=inference_config.report_output_dir,
        )
        output_paths = report_gen.generate(result, report_name=args.report_name)

    logger.info("")
    logger.info("Reports generated:")
    for fmt, path in output_paths.items():
        logger.info("  %s: %s", fmt.upper(), path)

    logger.info("")
    logger.info("Done.")


if __name__ == "__main__":
    main()
