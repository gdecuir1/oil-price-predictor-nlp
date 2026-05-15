"""
Post-training decision threshold for binary Up/Down models.
==========================================================

When argmax collapses to one class but P(Up) is spread out, sweep a validation
cutoff for class Up.  Thresholds that predict only one class are rejected when
``threshold_min_class_recall`` is set.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from .pipeline_config import PipelineConfig


def up_class_index(config: PipelineConfig) -> int:
    """Label id for Up (binary: 1; ternary: 2)."""
    return list(config.class_names).index("Up")


def predict_classes(
    logits: torch.Tensor,
    config: PipelineConfig,
    up_threshold: Optional[float] = None,
) -> torch.Tensor:
    """Argmax predictions, or thresholded Up if binary and ``up_threshold`` is set."""
    if config.label_mode == "binary" and up_threshold is not None:
        probs = torch.softmax(logits, dim=-1)
        p_up = probs[..., up_class_index(config)]
        return (p_up >= up_threshold).long()
    return logits.argmax(dim=-1)


def worst_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Minimum per-class recall (0 if a class is never predicted correctly)."""
    _, recall, _, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    active = [float(recall[i]) for i in (0, 1) if support[i] > 0]
    return min(active) if active else 0.0


def predictions_collapsed(y_pred: np.ndarray) -> bool:
    """True if every prediction is the same class."""
    return len(np.unique(y_pred)) < 2


@torch.no_grad()
def collect_val_probabilities(
    model: torch.nn.Module,
    loader: DataLoader,
    config: PipelineConfig,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gather true labels, P(Up), and argmax preds on a loader."""
    model.eval()
    y_true: List[int] = []
    p_up_list: List[float] = []
    argmax_list: List[int] = []
    up_idx = up_class_index(config)

    for xb, yb in loader:
        xb = xb.to(device)
        logits, _ = model(xb)
        probs = torch.softmax(logits, dim=-1)
        p_up_list.extend(probs[:, up_idx].cpu().numpy().tolist())
        argmax_list.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        y_true.extend(yb.numpy().tolist())

    return (
        np.asarray(y_true, dtype=np.int64),
        np.asarray(p_up_list, dtype=np.float64),
        np.asarray(argmax_list, dtype=np.int64),
    )


def _score_preds(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return float(balanced_accuracy_score(y_true, y_pred))


def tune_binary_up_threshold(
    y_true: np.ndarray,
    p_up: np.ndarray,
    argmax_preds: Optional[np.ndarray] = None,
    metric: str = "balanced_accuracy",
    min_recall_floor: float = 0.15,
    n_grid: int = 91,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Sweep P(Up) cutoffs; prefer thresholds that predict both classes on val.

    Returns:
        ``(best_threshold or None, metrics_dict)``.  ``None`` means use argmax.
    """
    if len(y_true) == 0:
        return None, {}

    if argmax_preds is None:
        argmax_preds = (p_up >= 0.5).astype(np.int64)

    grid = np.unique(np.concatenate([p_up, np.linspace(0.05, 0.95, n_grid)]))

    best_t: Optional[float] = None
    best_score = -1.0
    best_min_recall = -1.0

    for t in grid:
        preds = (p_up >= t).astype(np.int64)
        if predictions_collapsed(preds):
            continue
        mrec = worst_class_recall(y_true, preds)
        if mrec < min_recall_floor:
            continue
        score = _score_preds(y_true, preds, metric)
        if score > best_score or (score == best_score and mrec > best_min_recall):
            best_score = score
            best_min_recall = mrec
            best_t = float(t)

    # Relaxed pass: both classes predicted, any positive min recall
    if best_t is None:
        for t in grid:
            preds = (p_up >= t).astype(np.int64)
            if predictions_collapsed(preds):
                continue
            mrec = worst_class_recall(y_true, preds)
            if mrec <= 0:
                continue
            score = _score_preds(y_true, preds, metric)
            if score > best_score or (score == best_score and mrec > best_min_recall):
                best_score = score
                best_min_recall = mrec
                best_t = float(t)

    use_argmax = best_t is None
    if use_argmax:
        best_t = None
        preds_final = argmax_preds
    else:
        preds_final = (p_up >= best_t).astype(np.int64)

    _, recall, _, support = precision_recall_fscore_support(
        y_true, preds_final, labels=[0, 1], zero_division=0,
    )
    _, argmax_recall, _, argmax_support = precision_recall_fscore_support(
        y_true, argmax_preds, labels=[0, 1], zero_division=0,
    )

    return best_t, {
        "threshold_metric": metric,
        "threshold_min_class_recall": min_recall_floor,
        "used_argmax_fallback": use_argmax,
        "val_threshold": best_t,
        "val_score_at_threshold": best_score if not use_argmax else _score_preds(y_true, argmax_preds, metric),
        "val_min_recall_at_threshold": worst_class_recall(y_true, preds_final),
        "val_accuracy_at_threshold": float(accuracy_score(y_true, preds_final)),
        "val_balanced_accuracy_at_threshold": float(balanced_accuracy_score(y_true, preds_final)),
        "val_macro_f1_at_threshold": float(
            f1_score(y_true, preds_final, average="macro", zero_division=0)
        ),
        "val_recall_down_at_threshold": float(recall[0]) if support[0] > 0 else 0.0,
        "val_recall_up_at_threshold": float(recall[1]) if support[1] > 0 else 0.0,
        "val_accuracy_argmax": float(accuracy_score(y_true, argmax_preds)),
        "val_balanced_accuracy_argmax": float(balanced_accuracy_score(y_true, argmax_preds)),
        "val_macro_f1_argmax": float(
            f1_score(y_true, argmax_preds, average="macro", zero_division=0)
        ),
        "val_recall_down_argmax": float(argmax_recall[0]) if argmax_support[0] > 0 else 0.0,
        "val_recall_up_argmax": float(argmax_recall[1]) if argmax_support[1] > 0 else 0.0,
        "mean_p_up": float(p_up.mean()),
        "std_p_up": float(p_up.std()),
    }


@torch.no_grad()
def tune_threshold_on_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    config: PipelineConfig,
    device: torch.device,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Tune Up probability cutoff on validation (binary only)."""
    if config.label_mode != "binary" or not config.tune_up_threshold:
        return None, {}

    y_true, p_up, argmax_preds = collect_val_probabilities(model, loader, config, device)
    return tune_binary_up_threshold(
        y_true,
        p_up,
        argmax_preds=argmax_preds,
        metric=config.threshold_tuning_metric,
        min_recall_floor=config.threshold_min_class_recall,
    )
