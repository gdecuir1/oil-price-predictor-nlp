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
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# Project root on path for ``python -m ml_model.train_lstm``.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml_model.data.price_fetcher import get_price_labels
from ml_model.data.window_builder import build_windows
from ml_model.model_lstm import OilLSTMPredictor
from torch import nn as _nn
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


def compute_class_weights(y_train: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights: weight_i = N / (C * count_i).

    Args:
        y_train: Training labels only.
        num_classes: Number of classes (2 or 3).

    Returns:
        Float tensor of shape ``(num_classes,)`` for ``CrossEntropyLoss``.
    """
    n = len(y_train)
    weights = torch.ones(num_classes, dtype=torch.float32)
    for cls in range(num_classes):
        count = (y_train == cls).sum().item()
        if count > 0:
            weights[cls] = n / (float(num_classes) * count)
    logger.info("Class weights (train): %s", weights.tolist())
    return weights


def apply_feature_norm(
    X: torch.Tensor,
    feature_mean: Optional[torch.Tensor],
    feature_std: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply training-set z-score stored in a checkpoint.

    Args:
        X: Features ``(N, T, D)``.
        feature_mean: ``(1, 1, D)`` or None to skip.
        feature_std: ``(1, 1, D)`` or None to skip.

    Returns:
        Normalised tensor (or ``X`` unchanged).
    """
    if feature_mean is None or feature_std is None:
        return X
    return (X - feature_mean) / feature_std.clamp(min=1e-6)


def normalize_features(
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    X_test: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Z-score normalise using training-set mean/std over (time, feature).

    Args:
        X_train, X_val, X_test: Feature tensors ``(N, T, D)``.

    Returns:
        Normalised tensors plus ``feature_mean`` and ``feature_std`` (1, 1, D).
    """
    mean = X_train.mean(dim=(0, 1), keepdim=True)
    std = X_train.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
    return (
        (X_train - mean) / std,
        (X_val - mean) / std,
        (X_test - mean) / std,
        mean.cpu(),
        std.cpu(),
    )


def make_loader(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    weighted_sampler: bool = False,
) -> DataLoader:
    """Wrap tensors in a :class:`DataLoader`.

    Args:
        X: Feature tensor.
        y: Label tensor.
        batch_size: Batch size from config.
        shuffle: Used when ``weighted_sampler`` is False.
        weighted_sampler: Oversample minority classes in training.

    Returns:
        DataLoader yielding ``(X_batch, y_batch)``.
    """
    ds = TensorDataset(X, y)
    if weighted_sampler:
        counts: Dict[int, int] = {}
        for label in y.tolist():
            counts[label] = counts.get(label, 0) + 1
        weights = [1.0 / counts[int(lbl)] for lbl in y.tolist()]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(
    model: _nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 0.0,
) -> float:
    """Run one training epoch; return mean loss.

    Args:
        model: LSTM predictor.
        loader: Training loader (shuffled batches OK).
        criterion: Loss function.
        optimizer: AdamW optimiser.
        device: CPU or CUDA.
        max_grad_norm: Clip global gradient norm; 0 disables clipping.

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
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _per_class_recall(
    y_true: List[int],
    y_pred: List[int],
    num_classes: int,
) -> Dict[int, float]:
    """Recall per class id; 0.0 if class absent from y_true."""
    if not y_true:
        return {c: 0.0 for c in range(num_classes)}
    _, recall, _, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    return {c: float(recall[c]) if support[c] > 0 else 0.0 for c in range(num_classes)}


@torch.no_grad()
def evaluate_epoch(
    model: _nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, float, float, float, Dict[int, float]]:
    """Compute val/test loss, accuracy, macro F1, balanced accuracy, per-class recall.

    Returns:
        ``(mean_loss, accuracy, macro_f1, balanced_accuracy, recall_by_class)``.
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
    if not all_true:
        return mean_loss, 0.0, 0.0, 0.0, {c: 0.0 for c in range(num_classes)}

    acc = accuracy_score(all_true, all_preds)
    f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(all_true, all_preds)
    recall = _per_class_recall(all_true, all_preds, num_classes)
    return mean_loss, acc, f1, bal_acc, recall


def run_training_loop(
    model: _nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: PipelineConfig,
    class_weights: Optional[torch.Tensor],
    device: torch.device,
) -> Tuple[_nn.Module, List[Dict[str, Any]]]:
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
    criterion = nn.CrossEntropyLoss(
        weight=weight,
        label_smoothing=config.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    up_class_idx = list(config.class_names).index("Up")
    maximize_metric = config.early_stopping_metric in (
        "val_macro_f1",
        "val_balanced_accuracy",
    )
    best_score = float("-inf") if maximize_metric else float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience_counter = 0
    zero_up_recall_streak = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_grad_norm=config.max_grad_norm,
        )
        val_loss, val_acc, val_f1, val_bal_acc, val_recall = evaluate_epoch(
            model, val_loader, criterion, device, config.num_classes
        )
        scheduler.step(val_loss)

        recall_parts = ", ".join(
            f"{config.class_names[c]}={val_recall[c]:.3f}" for c in range(config.num_classes)
        )
        up_recall = val_recall.get(up_class_idx, 0.0)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "val_balanced_accuracy": val_bal_acc,
            "val_recall": {config.class_names[k]: v for k, v in val_recall.items()},
            "val_up_recall": up_recall,
        }
        history.append(record)
        logger.info(
            "Epoch %d/%d — train_loss=%.4f val_loss=%.4f val_acc=%.4f "
            "val_f1=%.4f val_bal_acc=%.4f | val recall: %s",
            epoch,
            config.epochs,
            train_loss,
            val_loss,
            val_acc,
            val_f1,
            val_bal_acc,
            recall_parts,
        )

        if config.early_stopping_metric == "val_macro_f1":
            score = val_f1
        elif config.early_stopping_metric == "val_balanced_accuracy":
            score = val_bal_acc
        else:
            score = val_loss

        improved = score > best_score if maximize_metric else score < best_score
        if improved:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if up_recall <= 0.0:
            zero_up_recall_streak += 1
        else:
            zero_up_recall_streak = 0

        if zero_up_recall_streak >= config.zero_up_recall_patience:
            logger.warning(
                "Early stopping at epoch %d: Up recall 0 for %d consecutive epochs",
                epoch,
                zero_up_recall_streak,
            )
            break

        if patience_counter >= config.patience:
            logger.info(
                "Early stopping at epoch %d (metric=%s)",
                epoch,
                config.early_stopping_metric,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def evaluate_test_set(
    model: _nn.Module,
    config: PipelineConfig,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Final test metrics and confusion matrix.

    Args:
        model: Trained model.
        config: Pipeline config (num_classes, class names).
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

    labels = list(range(config.num_classes))
    names = list(config.class_names)
    cm = confusion_matrix(all_true, all_preds, labels=labels)
    report = classification_report(
        all_true, all_preds, labels=labels,
        target_names=names,
        zero_division=0,
        output_dict=True,
    )
    acc = accuracy_score(all_true, all_preds)
    macro_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)

    print("\n=== Test confusion matrix (rows=true, cols=pred) ===")
    header = "".join(f"{n:>8}" for n in names)
    print(f"       {header}")
    for i, row_name in enumerate(names):
        print(f"{row_name:8}  {cm[i]}")

    print("\n=== Per-class precision / recall / F1 ===")
    print(classification_report(
        all_true, all_preds, labels=labels,
        target_names=names,
        zero_division=0,
    ))
    if config.label_mode == "ternary":
        _print_down_up_only_metrics(all_true, all_preds)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "y_true": all_true,
        "y_pred": all_preds,
    }


def _print_down_up_only_metrics(y_true: List[int], y_pred: List[int]) -> None:
    """Print accuracy on non-Flat test rows (ternary labels 0 and 2 only)."""
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t != 1]
    if not pairs:
        return
    correct = sum(1 for t, p in pairs if t == p)
    print(f"\n=== Down vs Up only (excluding Flat) ===")
    print(f"  Accuracy: {correct / len(pairs):.4f} ({correct}/{len(pairs)})")
    for cls, name in [(0, "Down"), (2, "Up")]:
        sub = [(t, p) for t, p in pairs if t == cls]
        if sub:
            acc = sum(1 for t, p in sub if t == p) / len(sub)
            print(f"  {name} recall: {acc:.4f} ({sum(1 for t,p in sub if t==p)}/{len(sub)})")


def save_checkpoint(
    model: OilLSTMPredictor,
    config: PipelineConfig,
    history: List[Dict[str, Any]],
    test_metrics: Dict[str, Any],
    feature_mean: Optional[torch.Tensor] = None,
    feature_std: Optional[torch.Tensor] = None,
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
        "feature_mean": feature_mean,
        "feature_std": feature_std,
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

    feature_mean: Optional[torch.Tensor] = None
    feature_std: Optional[torch.Tensor] = None
    if config.normalize_features:
        X_train, X_val, X_test, feature_mean, feature_std = normalize_features(
            X_train, X_val, X_test
        )
        logger.info("Applied train-set feature normalisation (per-dimension z-score)")

    class_weights = None
    if config.use_class_weights:
        class_weights = compute_class_weights(y_train, config.num_classes)

    train_loader = make_loader(
        X_train,
        y_train,
        config.batch_size,
        shuffle=not config.use_weighted_sampler,
        weighted_sampler=config.use_weighted_sampler,
    )
    val_loader = make_loader(X_val, y_val, config.batch_size, shuffle=False)
    test_loader = make_loader(X_test, y_test, config.batch_size, shuffle=False)

    model = OilLSTMPredictor(config).to(device)
    logger.info("Trainable parameters: %d", model.count_parameters())

    model, history = run_training_loop(
        model, train_loader, val_loader, config, class_weights, device
    )

    test_metrics = evaluate_test_set(model, config, test_loader, device)
    save_checkpoint(
        model, config, history, test_metrics, feature_mean, feature_std
    )

    elapsed = time.time() - t0
    logger.info("Total elapsed time: %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)
    print(f"\nTotal elapsed: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
