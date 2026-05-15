#!/usr/bin/env python3
"""
Evaluate a trained BiLSTM checkpoint and generate HTML/JSON reports.
====================================================================

Entry point::

    python -m ml_model.evaluate_lstm [--checkpoint PATH] [--window INT] [--gap INT]

Runs classification metrics on the chronological test split, backtests
predictions against actual USO log returns, and writes an HTML report with
attention heatmaps and training curves (loaded from checkpoint history).
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml_model.data.price_fetcher import get_price_labels
from ml_model.data.window_builder import build_windows_with_metadata
from ml_model.model_lstm import OilLSTMPredictorLegacy, load_model_from_checkpoint
from ml_model.pipeline_config import PipelineConfig
from ml_model.threshold_tuning import predict_classes, up_class_index
from ml_model.train_lstm import apply_feature_norm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _config_from_payload(payload: dict) -> PipelineConfig:
    """Rebuild :class:`PipelineConfig` from a checkpoint dict (handles legacy keys)."""
    fields = PipelineConfig.__dataclass_fields__
    raw = payload.get("config", {})
    kwargs = {k: v for k, v in raw.items() if k in fields}
    if "embed_dim" in raw and "finbert_dim" not in kwargs:
        kwargs["finbert_dim"] = 768
    if "finbert_dim" not in kwargs:
        kwargs["finbert_dim"] = 768
    if raw.get("embed_dim", 768) > 768:
        kwargs["use_keywords"] = True
    elif "use_keywords" not in kwargs:
        kwargs["use_keywords"] = False
    if "label_mode" not in kwargs:
        kwargs["label_mode"] = "ternary"
    return PipelineConfig(**kwargs)


def print_down_up_focus(
    y_true: List[int],
    y_pred: List[int],
    config: PipelineConfig,
    n_test: Optional[int] = None,
) -> Dict[str, Any]:
    """Metrics on decisive moves only (Down vs Up), for ternary or binary runs.

    For binary mode this matches full test accuracy.  For ternary, excludes Flat (1).
    """
    names = list(config.class_names)
    if config.label_mode == "binary":
        pairs = list(zip(y_true, y_pred))
        flat_idx = None
    else:
        flat_idx = 1
        pairs = [(t, p) for t, p in zip(y_true, y_pred) if t != flat_idx]

    if not pairs:
        print("\n=== Down vs Up focus: no samples ===")
        return {"accuracy": None, "n": 0}

    correct = sum(1 for t, p in pairs if t == p)
    acc = correct / len(pairs)
    print(f"\n=== Down vs Up focus (label_mode={config.label_mode}) ===")
    if n_test is not None:
        print(f"  (n_test={n_test} — indicative only; prefer balanced acc + per-class recall)")
    print(f"  Accuracy: {acc:.4f} ({correct}/{len(pairs)})")
    if config.label_mode == "ternary":
        for cls, name in [(0, "Down"), (2, "Up")]:
            sub = [(t, p) for t, p in pairs if t == cls]
            if sub:
                r = sum(1 for t, p in sub if t == p) / len(sub)
                print(f"  {name} recall: {r:.4f} ({sum(1 for t,p in sub if t==p)}/{len(sub)})")
    else:
        for i, name in enumerate(names):
            sub = [(t, p) for t, p in pairs if t == i]
            if sub:
                r = sum(1 for t, p in sub if t == p) / len(sub)
                print(f"  {name} recall: {r:.4f}")
    return {"accuracy": acc, "n": len(pairs)}


def parse_args() -> argparse.Namespace:
    """Parse evaluation CLI arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Evaluate BiLSTM oil direction model")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to .pkl checkpoint")
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--gap", type=int, default=None)
    return parser.parse_args()


def load_checkpoint(path: Path) -> Tuple[Any, PipelineConfig, List[Dict], Dict, Any, Any, dict]:
    """Load model (v3, v2, or legacy), config, and normalisation stats.

    Returns:
        ``(model, config, train_history, test_metrics, feature_mean, feature_std, payload)``.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)
    config = _config_from_payload(payload)
    model = load_model_from_checkpoint(payload, config)
    return (
        model,
        config,
        payload.get("train_history", []),
        payload.get("test_metrics", {}),
        payload.get("feature_mean"),
        payload.get("feature_std"),
        payload,
    )


def find_latest_checkpoint(ckpt_dir: Path) -> Path:
    """Return most recently modified ``.pkl`` in directory.

    Args:
        ckpt_dir: Checkpoints folder.

    Returns:
        Path to newest pickle.

    Raises:
        FileNotFoundError: If no checkpoints exist.
    """
    files = sorted(ckpt_dir.glob("model_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    return files[0]


def chronological_test_slice(
    X: torch.Tensor,
    y: torch.Tensor,
    meta: List[Dict[str, Any]],
    split: Tuple[float, float, float],
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """Extract test partition (same indices as training script).

    Args:
        X: All samples.
        y: All labels.
        meta: Per-sample metadata.
        split: Train/val/test fractions.

    Returns:
        Test ``X``, ``y``, and metadata list.
    """
    n = len(y)
    i_val_end = int(n * (split[0] + split[1]))
    return X[i_val_end:], y[i_val_end:], meta[i_val_end:]


@torch.no_grad()
def evaluate_classification(
    model: torch.nn.Module,
    test_loader: DataLoader,
    config: PipelineConfig,
    device: torch.device,
    up_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute confusion matrix, per-class metrics, accuracy, and majority baseline.

    Args:
        model: Trained LSTM.
        test_loader: Test :class:`DataLoader`.
        config: Pipeline config (for label names).
        device: CPU/CUDA.

    Returns:
        Dictionary with ``confusion_matrix``, ``per_class``, ``accuracy``,
        ``macro_f1``, ``majority_baseline_accuracy``, ``y_true``, ``y_pred``,
        ``attention_weights`` (list of arrays per batch).
    """
    model.eval()
    all_true: List[int] = []
    all_pred: List[int] = []
    all_attn: List[np.ndarray] = []

    thresh = up_threshold if up_threshold is not None else config.up_probability_threshold

    for xb, yb in test_loader:
        xb = xb.to(device)
        logits, attn = model(xb)
        preds = predict_classes(logits, config, thresh).cpu().numpy()
        all_pred.extend(preds.tolist())
        all_true.extend(yb.tolist())
        all_attn.append(attn.cpu().numpy())

    label_ids = list(range(config.num_classes))
    names = list(config.class_names)
    cm = confusion_matrix(all_true, all_pred, labels=label_ids)
    prec, rec, f1, support = precision_recall_fscore_support(
        all_true, all_pred, labels=label_ids, zero_division=0
    )
    acc = accuracy_score(all_true, all_pred)
    bal_acc = balanced_accuracy_score(all_true, all_pred)
    macro_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)

    if all_true:
        majority_cls = max(label_ids, key=lambda c: all_true.count(c))
        baseline_acc = sum(1 for t in all_true if t == majority_cls) / len(all_true)
    else:
        baseline_acc = 0.0

    per_class = {}
    for i, name in enumerate(names):
        per_class[name] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    print("\n=== Classification (test set) ===")
    print(f"n_test={len(all_true)} (small sample — metrics are indicative, not definitive)")
    print(
        f"Balanced accuracy: {bal_acc:.4f}  |  Macro F1: {macro_f1:.4f}  |  "
        f"Accuracy: {acc:.4f}  |  Majority baseline: {baseline_acc:.4f}"
    )
    if thresh is not None and config.label_mode == "binary":
        print(f"Decision rule: predict Up if P(Up) >= {thresh:.3f}")
    for i, name in enumerate(names):
        r = float(rec[i]) if support[i] > 0 else 0.0
        print(f"  {name} recall: {r:.4f} (support={int(support[i])})")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    print_down_up_focus(all_true, all_pred, config, n_test=len(all_true))

    return {
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "majority_baseline_accuracy": float(baseline_acc),
        "up_probability_threshold": thresh,
        "y_true": all_true,
        "y_pred": all_pred,
        "attention_weights": all_attn,
    }


@torch.no_grad()
def backtest_vs_actual(
    model: torch.nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    meta_test: List[Dict[str, Any]],
    price_df: pd.DataFrame,
    config: PipelineConfig,
    device: torch.device,
    up_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Compare model predictions to realised price directions on test dates.

  Horizon: news window ends on T−1; label is the USO move on prediction day T
  (see ``meta_test[i]['prediction_date']``).

    For each test sample, records actual close, previous close, log return,
    true and predicted direction, softmax probabilities, and correctness.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: True labels.
        meta_test: Metadata with ``prediction_date``.
        price_df: Labelled price DataFrame from :func:`get_price_labels`.
        config: Pipeline config.
        device: Compute device.

    Returns:
        DataFrame with one row per test prediction date.
    """
    model.eval()
    rows: List[Dict[str, Any]] = []
    thresh = up_threshold if up_threshold is not None else config.up_probability_threshold
    up_idx = up_class_index(config) if config.label_mode == "binary" else 2

    for i in range(len(X_test)):
        xb = X_test[i : i + 1].to(device)
        logits, _ = model(xb)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_cls = int(predict_classes(logits, config, thresh).item())
        pred_cls_argmax = int(logits.argmax(dim=1).item())
        true_cls = int(y_test[i].item())

        pred_date = pd.Timestamp(meta_test[i]["prediction_date"]).normalize()
        window_end = meta_test[i].get("window_end_date") or meta_test[i].get("last_news_date")
        close_t = float(price_df.loc[pred_date, "close"])
        log_ret = float(price_df.loc[pred_date, "log_return"])

        # Previous trading day close (row immediately before in sorted index).
        idx_pos = price_df.index.get_loc(pred_date)
        if isinstance(idx_pos, slice):
            idx_pos = idx_pos.start or 0
        prev_close = float(price_df.iloc[idx_pos - 1]["close"]) if idx_pos > 0 else np.nan

        names = list(config.class_names)
        pred_name = names[pred_cls] if pred_cls < len(names) else f"class_{pred_cls}"
        pred_argmax_name = (
            names[pred_cls_argmax] if pred_cls_argmax < len(names) else f"class_{pred_cls_argmax}"
        )
        row = {
            "prediction_date_T": pred_date.strftime("%Y-%m-%d"),
            "news_through_T_minus_1": str(window_end) if window_end else "",
            "actual_close": close_t,
            "prev_close": prev_close,
            "actual_log_return": log_ret,
            "actual_direction": names[true_cls] if true_cls < len(names) else str(true_cls),
            "predicted_direction": pred_name,
            "predicted_direction_argmax": pred_argmax_name,
            "correct": pred_cls == true_cls,
        }
        if len(probs) >= 1:
            row["predicted_proba_down"] = float(probs[0])
        if config.num_classes == 3 and len(probs) >= 2:
            row["predicted_proba_flat"] = float(probs[1])
        if len(probs) >= 2:
            row["predicted_proba_up"] = float(probs[up_idx])
        rows.append(row)

    df = pd.DataFrame(rows)

    print("\n=== Backtest: per-class recall (Down vs Up rows) ===")
    for cls_id, name in enumerate(config.class_names):
        subset = df[df["actual_direction"] == name]
        if len(subset) == 0:
            print(f"  {name}: no test samples")
            continue
        recall = subset["correct"].mean()
        print(f"  {name} recall: {recall:.4f} ({subset['correct'].sum()}/{len(subset)})")

    if len(df) and "actual_direction" in df.columns:
        names = set(config.class_names)
        pairs = [
            (a, p)
            for a, p in zip(df["actual_direction"], df["predicted_direction"])
            if a in names and p in names
        ]
        if pairs:
            y_t = [config.class_names.index(a) for a, _ in pairs]
            y_p = [config.class_names.index(p) for _, p in pairs]
            print(f"  Balanced accuracy (comparable preds only): {balanced_accuracy_score(y_t, y_p):.4f}")
        flat_n = int((~df["predicted_direction"].isin(names)).sum())
        if flat_n:
            print(f"  Note: {flat_n} predictions outside {list(config.class_names)} (e.g. Flat) — excluded above")
    print(f"  (n={len(df)} test points — treat as indicative)")
    return df


def generate_report(
    classification_results: Dict[str, Any],
    backtest_df: pd.DataFrame,
    attn_weights: List[np.ndarray],
    config: PipelineConfig,
    train_history: List[Dict[str, Any]],
) -> Tuple[Path, Path]:
    """Write HTML and JSON evaluation reports.

    Args:
        classification_results: Output of :func:`evaluate_classification`.
        backtest_df: Output of :func:`backtest_vs_actual`.
        attn_weights: Batched attention arrays from test forward passes.
        config: Pipeline configuration.
        train_history: List of per-epoch dicts from training.

    Returns:
        Tuple ``(html_path, json_path)``.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = config.report_path
    report_dir.mkdir(parents=True, exist_ok=True)
    html_path = report_dir / f"eval_{ts}.html"
    json_path = report_dir / f"eval_{ts}.json"

    # Flatten attention to (n_samples, window_days) for heatmap.
    attn_flat = np.concatenate(attn_weights, axis=0) if attn_weights else np.zeros((0, config.window_days))

    json_payload = {
        "timestamp": ts,
        "config": asdict(config),
        "classification": {
            k: v for k, v in classification_results.items()
            if k not in ("attention_weights", "y_true", "y_pred")
        },
        "backtest": backtest_df.to_dict(orient="records"),
        "train_history": train_history,
        "attention_weights": attn_flat.tolist(),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    html = _build_html(
        config, classification_results, backtest_df, attn_flat, train_history, ts
    )
    html_path.write_text(html, encoding="utf-8")
    logger.info("Wrote report: %s", html_path)
    logger.info("Wrote JSON: %s", json_path)
    return html_path, json_path


def _build_html(
    config: PipelineConfig,
    cls_res: Dict[str, Any],
    backtest_df: pd.DataFrame,
    attn_flat: np.ndarray,
    history: List[Dict[str, Any]],
    ts: str,
) -> str:
    """Assemble HTML string for the evaluation report.

    Args:
        config: Pipeline config.
        cls_res: Classification metrics dict.
        backtest_df: Backtest table.
        attn_flat: ``(n_samples, window_days)`` attention matrix.
        history: Training epoch history.
        ts: Timestamp string.

    Returns:
        Full HTML document as a string.
    """
    cfg_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in asdict(config).items()
    )

    label_names = list(config.class_names)
    n_cls = len(label_names)
    cm = np.array(cls_res["confusion_matrix"])
    cm_html = "<table class='cm'><tr><th></th>" + "".join(
        f"<th>{n}</th>" for n in label_names
    ) + "</tr>"
    for i, name in enumerate(label_names):
        cm_html += f"<tr><th>{name}</th>" + "".join(
            f"<td>{cm[i,j]}</td>" for j in range(n_cls)
        ) + "</tr>"
    cm_html += "</table>"

    metrics_html = "<table><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>"
    for name, m in cls_res["per_class"].items():
        metrics_html += (
            f"<tr><td>{name}</td><td>{m['precision']:.3f}</td>"
            f"<td>{m['recall']:.3f}</td><td>{m['f1']:.3f}</td><td>{m['support']}</td></tr>"
        )
    metrics_html += "</table>"

    bt_html = backtest_df.to_html(index=False, classes="sortable", border=0)

    heatmap_html = _attention_heatmap_html(attn_flat, config.window_days)
    curves_html = _training_curves_svg(history)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>LSTM Oil Direction Evaluation — {ts}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 1200px; }}
    h1, h2 {{ color: #1a365d; }}
    table {{ border-collapse: collapse; margin: 1rem 0; width: 100%; }}
    th, td {{ border: 1px solid #cbd5e0; padding: 0.4rem 0.6rem; text-align: left; }}
    th {{ background: #edf2f7; }}
    .cm td {{ text-align: center; }}
    .heat {{ display: inline-block; width: 28px; height: 18px; margin: 1px; }}
    .heat-row {{ margin-bottom: 2px; }}
    svg {{ max-width: 100%; height: auto; }}
  </style>
</head>
<body>
  <h1>Oil LSTM Evaluation Report</h1>
  <p>Generated: {ts}</p>

  <h2>Configuration</h2>
  <table>{cfg_rows}</table>

  <h2>Classification Metrics</h2>
  <p>Balanced accuracy: <strong>{cls_res.get('balanced_accuracy', 0):.4f}</strong> |
     Macro F1: <strong>{cls_res['macro_f1']:.4f}</strong> |
     Accuracy: <strong>{cls_res['accuracy']:.4f}</strong> |
     Majority baseline: <strong>{cls_res['majority_baseline_accuracy']:.4f}</strong></p>
  <p><em>n_test is small (~23); use per-class recall and Down vs Up focus, not accuracy alone.</em></p>
  <p>Horizon: news through T−1 → USO move on day T (prediction_date_T in backtest table).</p>
  {metrics_html}
  <h3>Confusion Matrix</h3>
  {cm_html}

  <h2>Backtest vs Actual Prices</h2>
  {bt_html}

  <h2>Attention Heatmap (test samples × window days)</h2>
  <p>Darker red = higher attention weight for that day in the window.</p>
  {heatmap_html}

  <h2>Training History</h2>
  {curves_html}
</body>
</html>"""


def _attention_heatmap_html(attn: np.ndarray, window_days: int) -> str:
    """Render a simple CSS heatmap for attention weights.

    Args:
        attn: Shape ``(n_samples, window_days)``.
        window_days: Number of days per row.

    Returns:
        HTML fragment.
    """
    if attn.size == 0:
        return "<p>No attention data.</p>"
    max_samples = min(40, attn.shape[0])
    html_parts = []
    for i in range(max_samples):
        row = attn[i]
        cells = ""
        for t in range(min(window_days, row.shape[0])):
            v = float(row[t])
            # Intensity 0–1 mapped to red channel.
            intensity = int(255 * min(1.0, v * window_days))
            cells += (
                f'<span class="heat" style="background:rgb({intensity},40,40)" '
                f'title="day {t + 1}: {v:.3f}"></span>'
            )
        html_parts.append(f'<div class="heat-row">Sample {i + 1}: {cells}</div>')
    return "\n".join(html_parts)


def _training_curves_svg(history: List[Dict[str, Any]]) -> str:
    """Simple inline SVG for train/val loss and val accuracy.

    Args:
        history: Per-epoch metric dicts.

    Returns:
        HTML with embedded SVG or placeholder.
    """
    if not history:
        return "<p>No training history in checkpoint.</p>"
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_acc = [h.get("val_accuracy", 0) for h in history]

    w, h = 600, 200
    def scale_y(vals, height=h - 20):
        lo, hi = min(vals), max(vals)
        if hi == lo:
            hi = lo + 1e-6
        return [height - 10 - (v - lo) / (hi - lo) * (height - 20) for v in vals]

    def polyline(vals, color):
        ys = scale_y(vals)
        pts = " ".join(f"{20 + i * (w-40)/max(len(vals)-1,1)},{ys[i]}" for i in range(len(vals)))
        return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>'

    return f"""<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">
      <text x="10" y="15" font-size="12">Loss / Accuracy vs epoch</text>
      {polyline(train_loss, "#3182ce")}
      {polyline(val_loss, "#e53e3e")}
      {polyline(val_acc, "#38a169")}
    </svg>
    <p><span style="color:#3182ce">■</span> train_loss
       <span style="color:#e53e3e">■</span> val_loss
       <span style="color:#38a169">■</span> val_accuracy</p>"""


def main() -> None:
    """Run full evaluation pipeline."""
    args = parse_args()
    config = PipelineConfig()
    if args.window is not None:
        config.window_days = args.window
    if args.gap is not None:
        config.gap_days = args.gap

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = find_latest_checkpoint(config.checkpoint_path)
    logger.info("Using checkpoint: %s", ckpt_path)

    model, ckpt_config, history, _, feat_mean, feat_std, payload = load_checkpoint(ckpt_path)
    # Prefer checkpoint config for consistency.
    if args.window is None and args.gap is None:
        config = ckpt_config
    else:
        config.window_days = args.window or ckpt_config.window_days
        config.gap_days = args.gap if args.gap is not None else ckpt_config.gap_days

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    price_df = get_price_labels(config)
    X, y, meta = build_windows_with_metadata(config, price_df)
    X_test, y_test, meta_test = chronological_test_slice(
        X, y, meta, config.train_val_test_split
    )
    X_test = apply_feature_norm(X_test, feat_mean, feat_std)
    if X_test.size(-1) > 768 and isinstance(model, OilLSTMPredictorLegacy):
        X_test = X_test[..., :768]

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=config.batch_size,
        shuffle=False,
    )

    up_thresh = payload.get("up_probability_threshold")
    if up_thresh is None:
        up_thresh = ckpt_config.up_probability_threshold

    cls_res = evaluate_classification(
        model, test_loader, config, device, up_threshold=up_thresh
    )
    backtest_df = backtest_vs_actual(
        model, X_test, y_test, meta_test, price_df, config, device, up_threshold=up_thresh
    )
    generate_report(
        cls_res,
        backtest_df,
        cls_res["attention_weights"],
        config,
        history,
    )


if __name__ == "__main__":
    main()
