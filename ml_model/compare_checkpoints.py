#!/usr/bin/env python3
"""
Compare two or more LSTM checkpoints on the same chronological test split.
============================================================================

    python -m ml_model.compare_checkpoints

    python -m ml_model.compare_checkpoints \\
        --checkpoints ml_model/outputs/checkpoints/model_20260514_234831_5w_0g.pkl \\
        ml_model/outputs/checkpoints/model_20260514_235333_5w_0g.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml_model.data.price_fetcher import get_price_labels
from ml_model.data.window_builder import build_windows_with_metadata
from ml_model.evaluate_lstm import (
    backtest_vs_actual,
    chronological_test_slice,
    evaluate_classification,
    print_down_up_focus,
)
from ml_model.model_lstm import OilLSTMPredictorLegacy, load_model_from_checkpoint
from ml_model.pipeline_config import PipelineConfig
from ml_model.train_lstm import apply_feature_norm


def _config_from_payload(payload: dict) -> PipelineConfig:
    fields = PipelineConfig.__dataclass_fields__
    raw = payload.get("config", {})
    kwargs = {k: v for k, v in raw.items() if k in fields}
    if "embed_dim" in raw and "finbert_dim" not in kwargs:
        kwargs["finbert_dim"] = 768 if raw["embed_dim"] == 768 else raw["embed_dim"]
    if "finbert_dim" not in kwargs:
        kwargs["finbert_dim"] = 768
    if raw.get("embed_dim", 768) > 768:
        kwargs["use_keywords"] = True
    elif "use_keywords" not in kwargs:
        kwargs["use_keywords"] = False
    return PipelineConfig(**kwargs)


def evaluate_one_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    """Load checkpoint and run test-set metrics.

    Args:
        path: ``.pkl`` checkpoint path.
        device: Torch device.

    Returns:
        Summary dict with accuracy, macro_f1, down_up_accuracy, etc.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)
    config = _config_from_payload(payload)
    model = load_model_from_checkpoint(payload, config)
    model.to(device)

    price_df = get_price_labels(config)
    X, y, meta = build_windows_with_metadata(config, price_df)
    X_test, y_test, meta_test = chronological_test_slice(
        X, y, meta, config.train_val_test_split
    )
    X_test = apply_feature_norm(
        X_test, payload.get("feature_mean"), payload.get("feature_std")
    )

    # Legacy checkpoints: use first 768 dims only.
    if X_test.size(-1) > 768 and isinstance(model, OilLSTMPredictorLegacy):
        X_test = X_test[..., :768]

    loader = DataLoader(TensorDataset(X_test, y_test), batch_size=config.batch_size)
    cls_res = evaluate_classification(model, loader, config, device)
    backtest_df = backtest_vs_actual(
        model, X_test, y_test, meta_test, price_df, config, device
    )
    down_up = print_down_up_focus(cls_res["y_true"], cls_res["y_pred"], config)

    return {
        "path": str(path.name),
        "label_mode": config.label_mode,
        "input_dim": config.input_dim,
        "num_classes": config.num_classes,
        "accuracy": cls_res["accuracy"],
        "macro_f1": cls_res["macro_f1"],
        "majority_baseline": cls_res["majority_baseline_accuracy"],
        "down_up_accuracy": down_up.get("accuracy"),
        "n_test": len(y_test),
        "train_test_metrics": payload.get("test_metrics", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LSTM checkpoints")
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        type=Path,
        default=None,
        help="Checkpoint paths (default: two newest in checkpoints/)",
    )
    args = parser.parse_args()

    ckpt_dir = PipelineConfig().checkpoint_path
    paths = args.checkpoints
    if not paths:
        all_pkls = sorted(ckpt_dir.glob("model_*.pkl"), key=lambda p: p.stat().st_mtime)
        legacy = ckpt_dir / "model_20260514_234831_5w_0g.pkl"
        paths = [p for p in all_pkls if p.name.endswith("_5w_0g.pkl")][:2]
        if legacy.exists() and legacy not in paths:
            paths = [legacy] + [p for p in paths if p != legacy][:1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 72)
    print("CHECKPOINT COMPARISON (chronological test split)")
    print("=" * 72)

    rows: List[Dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            print(f"SKIP (missing): {path}")
            continue
        print(f"\n--- {path.name} ---")
        rows.append(evaluate_one_checkpoint(path, device))

    print("\n" + "=" * 72)
    print("SUMMARY TABLE")
    print("=" * 72)
    print(
        f"{'checkpoint':<42} {'mode':<8} {'acc':>6} {'macroF1':>8} "
        f"{'DownUpAcc':>10} {'n_test':>7}"
    )
    for r in rows:
        du = r.get("down_up_accuracy")
        du_s = f"{du:.4f}" if du is not None else "n/a"
        print(
            f"{r['path']:<42} {r['label_mode']:<8} {r['accuracy']:>6.3f} "
            f"{r['macro_f1']:>8.3f} {du_s:>10} {r['n_test']:>7}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
