#!/usr/bin/env python3
"""
Train the BiLSTM oil-direction model on FinBERT day embeddings.
================================================================

Entry point::

    python -m ml_model.train_lstm [--window INT] [--gap INT] [--epochs INT] [--ticker STR]

Pipeline steps:

1. Parse CLI overrides into :class:`~ml_model.pipeline_config.PipelineConfig`.
2. Download / cache price labels via :func:`~ml_model.data.price_fetcher.get_price_labels`.
3. Build ``(X, y)`` windows via :func:`~ml_model.data.window_builder.build_windows`.
4. **Chronological** train/val/test split (no shuffle before split).
5. Train :class:`~ml_model.model_lstm.OilLSTMPredictor` with optional class weights.
6. Evaluate on test set; save ``.pkl`` checkpoint and JSON config snapshot.

Target wall time: under 30 minutes on CPU (frozen FinBERT + embedding cache).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset

# Project root on path for ``python -m ml_model.train_lstm``.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml_model.data.price_fetcher import get_price_labels
from ml_model.data.window_builder import build_windows
from ml_model.model_lstm import OilLSTMPredictor
from ml_model.pipeline_config import PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments that override :class:`PipelineConfig`.

    Returns:
        Parsed namespace with optional ``window``, ``gap``, ``epochs``, ``ticker``.
    """
    parser = argparse.ArgumentParser(
        description="Train BiLSTM oil direction model on news embeddings",
    )
    parser.add_argument("--window", type=int, default=None, help="window_days")
    parser.add_argument("--gap", type=int, default=None, help="gap_days")
    parser.add_argument("--epochs", type=int, default=None, help="Max training epochs")
    parser.add_argument("--ticker", type=str, default=None, help="yfinance ticker")
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Construct config, applying CLI overrides.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Config ready for the pipeline.
    """
    config = PipelineConfig()
    if args.window is not None:
        config.window_days = args.window
    if args.gap is not None:
        config.gap_days = args.gap
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.ticker is not None:
        config.price_ticker = args.ticker
    return config


def chronological_split(
    X: torch.Tensor,
    y: torch.Tensor,
    split: Tuple[float, float, float],
) -> Tuple[torch.Tensor, ...]:
    """Split tensors by time order using cumulative index fractions.

    Samples are assumed already ordered oldest → newest.  **No shuffle**
    is applied, preventing future information from leaking into training.

    Args:
        X: Features ``(N, T, D)``.
        y: Labels ``(N,)``.
        split: ``(train_frac, val_frac, test_frac)`` summing to ~1.

    Returns:
        ``(X_train, y_train, X_val, y_val, X_test, y_test)``.
    """
    n = len(y)
    # 80th / 90th percentile indices for default (0.8, 0.1, 0.1).
    i_train_end = int(n * split[0])
    i_val_end = int(n * (split[0] + split[1]))

    X_train, y_train = X[:i_train_end], y[:i_train_end]
    X_val, y_val = X[i_train_end:i_val_end], y[i_train_end:i_val_end]
    X_test, y_test = X[i_val_end:], y[i_val_end:]

    logger.info(
        "Chronological split: train=%d val=%d test=%d (total %d)",
        len(y_train),
        len(y_val),
        len(y_test),
        n,
    )
    _log_class_dist("train", y_train)
    _log_class_dist("val", y_val)
    _log_class_dist("test", y_test)
    return X_train, y_train, X_val, y_val, X_test, y_test


def _log_class_dist(name: str, y: torch.Tensor) -> None:
    """Log per-split label counts.

    Args:
        name: Split name for logging.
        y: Label tensor.
    """
    unique, counts = torch.unique(y, return_counts=True)
    dist = {int(u.item()): int(c.item()) for u, c in zip(unique, counts)}
    logger.info("  %s class distribution: %s", name, dist)


def compute_class_weights(y_train: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency weights: weight_i = N / (3 * count_i).

    Args:
        y_train: Training labels only.

    Returns:
        Float tensor of shape ``(3,)`` for ``CrossEntropyLoss(weight=...)``.
    """
    n = len(y_train)
    weights = torch.ones(3, dtype=torch.float32)
    for cls in range(3):
        count = (y_train == cls).sum().item()
        if count > 0:
            weights[cls] = n / (3.0 * count)
    logger.info("Class weights (train): %s", weights.tolist())
    return weights


def make_loader(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Wrap tensors in a :class:`DataLoader`.

    Args:
        X: Feature tensor.
        y: Label tensor.
        batch_size: Batch size from config.
        shuffle: True only for training (after chronological split).

    Returns:
        DataLoader yielding ``(X_batch, y_batch)``.
    """
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(
    model: OilLSTMPredictor,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch; return mean loss.

    Args:
        model: LSTM predictor.
        loader: Training loader (shuffled batches OK).
        criterion: Loss function.
        optimizer: AdamW optimiser.
        device: CPU or CUDA.

    Returns:
        Average loss over batches.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits, _ = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_epoch(
    model: OilLSTMPredictor,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Compute val/test loss, accuracy, and macro F1.

    Args:
        model: LSTM predictor.
        loader: Validation or test loader.
        criterion: Loss function.
        device: Compute device.

    Returns:
        ``(mean_loss, accuracy, macro_f1)``.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds: List[int] = []
    all_true: List[int] = []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits, _ = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item()
        n_batches += 1
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_true.extend(yb.cpu().tolist())

    mean_loss = total_loss / max(n_batches, 1)
    acc = accuracy_score(all_true, all_preds) if all_true else 0.0
    f1 = f1_score(all_true, all_preds, average="macro", zero_division=0) if all_true else 0.0
    return mean_loss, acc, f1


def run_training_loop(
    model: OilLSTMPredictor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: PipelineConfig,
    class_weights: Optional[torch.Tensor],
    device: torch.device,
) -> Tuple[OilLSTMPredictor, List[Dict[str, Any]]]:
    """Full training with early stopping and best-weight restore.

    Args:
        model: Fresh or continued model.
        train_loader: Training data.
        val_loader: Validation data.
        config: Training hyperparameters.
        class_weights: Optional CE weights.
        device: Compute device.

    Returns:
        Tuple of ``(model_with_best_weights, train_history)``.
    """
    weight = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience_counter = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
        }
        history.append(record)
        logger.info(
            "Epoch %d/%d — train_loss=%.4f val_loss=%.4f val_acc=%.4f val_f1=%.4f",
            epoch,
            config.epochs,
            train_loss,
            val_loss,
            val_acc,
            val_f1,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def evaluate_test_set(
    model: OilLSTMPredictor,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Final test metrics and confusion matrix.

    Args:
        model: Trained model.
        test_loader: Test loader.
        device: Compute device.

    Returns:
        Dict with ``accuracy``, ``confusion_matrix``, ``classification_report``,
        and per-class metrics.
    """
    model.eval()
    all_preds: List[int] = []
    all_true: List[int] = []
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits, _ = model(xb)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_true.extend(yb.tolist())

    cm = confusion_matrix(all_true, all_preds, labels=[0, 1, 2])
    report = classification_report(
        all_true, all_preds, labels=[0, 1, 2],
        target_names=["Down", "Flat", "Up"],
        zero_division=0,
        output_dict=True,
    )
    acc = accuracy_score(all_true, all_preds)

    print("\n=== Test confusion matrix (rows=true, cols=pred) ===")
    print("       Down  Flat   Up")
    for i, row_name in enumerate(["Down", "Flat", "Up"]):
        print(f"{row_name:5}  {cm[i]}")

    print("\n=== Per-class precision / recall / F1 ===")
    print(classification_report(
        all_true, all_preds, labels=[0, 1, 2],
        target_names=["Down", "Flat", "Up"],
        zero_division=0,
    ))

    return {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "y_true": all_true,
        "y_pred": all_preds,
    }


def save_checkpoint(
    model: OilLSTMPredictor,
    config: PipelineConfig,
    history: List[Dict[str, Any]],
    test_metrics: Dict[str, Any],
) -> Path:
    """Save pickle checkpoint and JSON config per spec.

    Args:
        model: Trained model.
        config: Pipeline config.
        history: Per-epoch metrics list.
        test_metrics: Test evaluation dict.

    Returns:
        Path to the ``.pkl`` file.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = config.checkpoint_path
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_name = f"model_{ts}_{config.window_days}w_{config.gap_days}g.pkl"
    ckpt_path = ckpt_dir / ckpt_name

    payload = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "train_history": history,
        "embed_cache_path": config.embed_cache_path,
        "test_metrics": test_metrics,
    }
    with open(ckpt_path, "wb") as f:
        pickle.dump(payload, f)
    logger.info("Saved checkpoint: %s", ckpt_path)

    cfg_path = ckpt_dir / f"config_{ts}.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)
    logger.info("Saved config snapshot: %s", cfg_path)
    return ckpt_path


def main() -> None:
    """Orchestrate training end-to-end."""
    t0 = time.time()
    args = parse_args()
    config = build_config_from_args(args)

    os.makedirs(config.checkpoint_path, exist_ok=True)
    os.makedirs(config.report_path, exist_ok=True)
    config.embed_cache_file.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    price_df = get_price_labels(config)
    X, y = build_windows(config, price_df)

    X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(
        X, y, config.train_val_test_split
    )

    class_weights = None
    if config.use_class_weights:
        class_weights = compute_class_weights(y_train)

    train_loader = make_loader(X_train, y_train, config.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, config.batch_size, shuffle=False)
    test_loader = make_loader(X_test, y_test, config.batch_size, shuffle=False)

    model = OilLSTMPredictor(config).to(device)
    logger.info("Trainable parameters: %d", model.count_parameters())

    model, history = run_training_loop(
        model, train_loader, val_loader, config, class_weights, device
    )

    test_metrics = evaluate_test_set(model, test_loader, device)
    save_checkpoint(model, config, history, test_metrics)

    elapsed = time.time() - t0
    logger.info("Total elapsed time: %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)
    print(f"\nTotal elapsed: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
