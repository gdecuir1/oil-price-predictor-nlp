"""
Evaluation Metrics Calculator
==============================

Computes classification metrics for all sub-domain tasks: accuracy,
precision, recall, F1 (macro and weighted), and confusion matrices.

Designed to accumulate predictions across batches and compute final
metrics at epoch end, avoiding the memory cost of storing all predictions
simultaneously.

The calculator also produces formatted metric summaries suitable for
logging and inclusion in the comprehensive output report.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Accumulates predictions and computes multi-task classification metrics.

    Usage::

        calc = MetricsCalculator(subdomain_keys=["market_direction", ...])
        for batch in loader:
            calc.update(logits_dict, targets_dict)
        results = calc.compute()
        calc.reset()

    Args:
        subdomain_keys: List of sub-domain keys to track.
        label_names: Optional mapping of ``{subdomain_key: [label_strs]}``.
    """

    def __init__(
        self,
        subdomain_keys: List[str],
        label_names: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Initialise empty accumulators for each sub-domain."""
        self.subdomain_keys = subdomain_keys
        self.label_names = label_names or {}

        # Accumulators: lists of (predictions, targets) tensors per sub-domain
        self._predictions: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self._targets: Dict[str, List[torch.Tensor]] = defaultdict(list)

    def reset(self) -> None:
        """Clear all accumulated predictions and targets.

        Call this at the start of each epoch to begin fresh accumulation.
        """
        self._predictions = defaultdict(list)
        self._targets = defaultdict(list)

    def update(
        self,
        logits: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> None:
        """Accumulate one batch of predictions.

        Args:
            logits: ``{subdomain_key: (batch, n_classes)}`` model logits.
            targets: ``{subdomain_key: (batch,)}`` integer class labels.
        """
        for key in self.subdomain_keys:
            if key in logits and key in targets:
                preds = logits[key].argmax(dim=-1).detach().cpu()
                tgts = targets[key].detach().cpu()
                self._predictions[key].append(preds)
                self._targets[key].append(tgts)

    def compute(self) -> Dict[str, Dict[str, Any]]:
        """Compute final metrics from all accumulated batches.

        Returns:
            Nested dict: ``{subdomain_key: {metric_name: value}}``.

            Each sub-domain dict contains:
                - ``accuracy``: overall accuracy (float).
                - ``precision_macro``: macro-averaged precision.
                - ``recall_macro``: macro-averaged recall.
                - ``f1_macro``: macro-averaged F1 score.
                - ``f1_weighted``: weighted-average F1 score.
                - ``per_class_precision``: dict of per-class precisions.
                - ``per_class_recall``: dict of per-class recalls.
                - ``per_class_f1``: dict of per-class F1 scores.
                - ``confusion_matrix``: 2-D numpy array.
                - ``support``: per-class sample counts.
        """
        results: Dict[str, Dict[str, Any]] = {}

        for key in self.subdomain_keys:
            if key not in self._predictions or not self._predictions[key]:
                continue

            all_preds = torch.cat(self._predictions[key]).numpy()
            all_targets = torch.cat(self._targets[key]).numpy()

            n_classes = max(all_preds.max(), all_targets.max()) + 1

            # Build confusion matrix manually (avoid sklearn dependency)
            cm = self._confusion_matrix(all_targets, all_preds, n_classes)

            # Derive per-class metrics from the confusion matrix
            per_class = self._per_class_metrics(cm)

            # Aggregate metrics
            accuracy = float(np.sum(all_preds == all_targets)) / max(len(all_targets), 1)

            precisions = [m["precision"] for m in per_class]
            recalls = [m["recall"] for m in per_class]
            f1s = [m["f1"] for m in per_class]
            supports = [m["support"] for m in per_class]

            total_support = sum(supports)
            macro_precision = np.mean(precisions)
            macro_recall = np.mean(recalls)
            macro_f1 = np.mean(f1s)

            # Weighted average (by class support)
            if total_support > 0:
                weighted_f1 = sum(
                    f * s for f, s in zip(f1s, supports)
                ) / total_support
            else:
                weighted_f1 = 0.0

            # Build per-class metric dicts with label names if available
            class_labels = self.label_names.get(key, [str(i) for i in range(n_classes)])
            per_class_precision = {
                class_labels[i]: per_class[i]["precision"]
                for i in range(min(len(class_labels), len(per_class)))
            }
            per_class_recall = {
                class_labels[i]: per_class[i]["recall"]
                for i in range(min(len(class_labels), len(per_class)))
            }
            per_class_f1 = {
                class_labels[i]: per_class[i]["f1"]
                for i in range(min(len(class_labels), len(per_class)))
            }

            results[key] = {
                "accuracy": accuracy,
                "precision_macro": float(macro_precision),
                "recall_macro": float(macro_recall),
                "f1_macro": float(macro_f1),
                "f1_weighted": float(weighted_f1),
                "per_class_precision": per_class_precision,
                "per_class_recall": per_class_recall,
                "per_class_f1": per_class_f1,
                "confusion_matrix": cm,
                "support": {
                    class_labels[i]: supports[i]
                    for i in range(min(len(class_labels), len(supports)))
                },
            }

        return results

    def get_primary_f1(self) -> float:
        """Compute F1 for the primary task only (market_direction).

        Convenience method for early stopping decisions.

        Returns:
            Macro F1 for market_direction, or 0.0 if not available.
        """
        results = self.compute()
        if "market_direction" in results:
            return results["market_direction"]["f1_macro"]
        return 0.0

    def format_summary(self, results: Optional[Dict] = None) -> str:
        """Format metrics as a human-readable multi-line string.

        Args:
            results: Precomputed results dict.  If ``None``, calls
                :meth:`compute` internally.

        Returns:
            Formatted string suitable for logging.
        """
        if results is None:
            results = self.compute()

        lines: List[str] = []
        lines.append("=" * 70)
        lines.append("EVALUATION METRICS SUMMARY")
        lines.append("=" * 70)

        for key in self.subdomain_keys:
            if key not in results:
                continue

            m = results[key]
            is_primary = key == "market_direction"
            prefix = "*** " if is_primary else "    "

            lines.append(f"\n{prefix}{key.upper().replace('_', ' ')}"
                         f"{' (PRIMARY)' if is_primary else ''}")
            lines.append(f"{prefix}  Accuracy:         {m['accuracy']:.4f}")
            lines.append(f"{prefix}  Precision (macro): {m['precision_macro']:.4f}")
            lines.append(f"{prefix}  Recall (macro):    {m['recall_macro']:.4f}")
            lines.append(f"{prefix}  F1 (macro):        {m['f1_macro']:.4f}")
            lines.append(f"{prefix}  F1 (weighted):     {m['f1_weighted']:.4f}")

            # Per-class F1 breakdown
            for cls_name, f1_val in m["per_class_f1"].items():
                support = m["support"].get(cls_name, 0)
                lines.append(
                    f"{prefix}    {cls_name:>20s}: F1={f1_val:.4f}  "
                    f"(n={support})"
                )

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal computation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _confusion_matrix(
        y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
    ) -> np.ndarray:
        """Build a confusion matrix without sklearn.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            n_classes: Total number of classes.

        Returns:
            ``(n_classes, n_classes)`` confusion matrix where
            ``cm[i][j]`` = number of samples with true label ``i``
            predicted as ``j``.
        """
        cm = np.zeros((n_classes, n_classes), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            cm[t][p] += 1
        return cm

    @staticmethod
    def _per_class_metrics(
        cm: np.ndarray,
    ) -> List[Dict[str, float]]:
        """Derive precision, recall, F1, and support from a confusion matrix.

        Args:
            cm: ``(n_classes, n_classes)`` confusion matrix.

        Returns:
            List of dicts, one per class, with keys ``precision``,
            ``recall``, ``f1``, ``support``.
        """
        n_classes = cm.shape[0]
        metrics: List[Dict[str, float]] = []

        for i in range(n_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            support = cm[i, :].sum()

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = (
                2 * precision * recall / max(precision + recall, 1e-9)
            )

            metrics.append({
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(support),
            })

        return metrics
