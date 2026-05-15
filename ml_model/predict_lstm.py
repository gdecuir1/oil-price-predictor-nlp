#!/usr/bin/env python3
"""
Single-date inference for the BiLSTM oil-direction model.
=======================================================

Entry point::

    python -m ml_model.predict_lstm --end-date YYYY-MM-DD [--window INT] [--gap INT] [--checkpoint PATH]

Loads a trained checkpoint, builds one news window ending at ``end-date``
(where ``end-date`` is the **prediction date**), and prints the predicted
class, class probabilities, per-day attention weights, and article filenames
that contributed to each day in the window.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch.nn

import pandas as pd
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml_model.data.price_fetcher import get_price_labels
from ml_model.data.window_builder import build_single_window
from ml_model.evaluate_lstm import _config_from_payload
from ml_model.model_lstm import OilLSTMPredictorLegacy, load_model_from_checkpoint
from ml_model.pipeline_config import PipelineConfig
from ml_model.train_lstm import apply_feature_norm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)



def parse_args() -> argparse.Namespace:
    """Parse prediction CLI arguments.

    Returns:
        Parsed namespace with ``end_date``, optional window/gap/checkpoint.
    """
    parser = argparse.ArgumentParser(description="Predict oil direction for one date")
    parser.add_argument("--end-date", type=str, required=True, help="Prediction date YYYY-MM-DD")
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--gap", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def load_checkpoint(path: Path) -> Tuple[Any, PipelineConfig, Dict[str, Any]]:
    """Load model, config, and full pickle payload."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    config = _config_from_payload(payload)
    model = load_model_from_checkpoint(payload, config)
    return model, config, payload


def find_latest_checkpoint(ckpt_dir: Path) -> Path:
    """Return newest ``model_*.pkl`` by modification time.

    Args:
        ckpt_dir: Directory to search.

    Returns:
        Path to checkpoint.

    Raises:
        FileNotFoundError: If none found.
    """
    files = sorted(ckpt_dir.glob("model_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    return files[0]


@torch.no_grad()
def run_prediction(
    model: torch.nn.Module,
    X: torch.Tensor,
    config: PipelineConfig,
    device: torch.device,
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """Forward pass for a single window batch.

    Args:
        model: Trained LSTM.
        X: Shape ``(1, window_days, embed_dim)``.
        config: Pipeline config.
        device: CPU/CUDA.

    Returns:
        ``(predicted_class, probabilities, attention_weights)``.
    """
    model.to(device)
    X = X.to(device)
    logits, attn = model(X)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
    pred = int(logits.argmax(dim=1).item())
    return pred, probs, attn.squeeze(0).cpu()


def main() -> None:
    """Load checkpoint, build window, print prediction and interpretability."""
    args = parse_args()
    ckpt = args.checkpoint or find_latest_checkpoint(PipelineConfig().checkpoint_path)
    logger.info("Checkpoint: %s", ckpt)

    model, config, payload = load_checkpoint(ckpt)
    if args.window is not None:
        config.window_days = args.window
    if args.gap is not None:
        config.gap_days = args.gap

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    price_df = get_price_labels(config)
    pred_ts = pd.Timestamp(args.end_date).normalize()
    last_price_date = pd.Timestamp(price_df.index.max()).normalize()

    X, true_label, meta = build_single_window(config, args.end_date, price_df)
    X = apply_feature_norm(X, payload.get("feature_mean"), payload.get("feature_std"))
    if X.size(-1) > 768 and isinstance(model, OilLSTMPredictorLegacy):
        X = X[..., :768]

    pred_cls, probs, attn = run_prediction(model, X, config, device)
    names = list(config.class_names)

    print("\n" + "=" * 60)
    print(f"Prediction date: {meta['prediction_date']}")
    print(f"Predicted direction: {names[pred_cls]} (class {pred_cls})")
    prob_parts = [f"{names[i]}: {probs[i]:.4f}" for i in range(len(names))]
    print("Probabilities — " + "  ".join(prob_parts))

    if pred_ts > last_price_date:
        print("\nNote: end-date is beyond downloaded price history — no ground-truth label.")
    elif true_label is not None:
        print(f"Actual direction:    {names[true_label]} (class {true_label})")
        print(f"Match: {'YES' if pred_cls == true_label else 'NO'}")
    else:
        print("\nNo price label available for this date (non-trading day or missing data).")

    print("\nAttention weights (which window days mattered most):")
    for day_str, w in zip(meta["window_dates"], attn.tolist()):
        print(f"  {day_str}: {w:.4f}")

    print("\nArticles used per day:")
    for day_str, files in meta["article_filenames"].items():
        if files:
            print(f"  {day_str}:")
            for fn in files:
                print(f"    - {fn}")
        else:
            print(f"  {day_str}: (no articles — zero vector used)")
    print("=" * 60)


if __name__ == "__main__":
    main()
