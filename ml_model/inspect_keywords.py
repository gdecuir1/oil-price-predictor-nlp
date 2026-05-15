#!/usr/bin/env python3
"""
Inspect mean keyword features by label (Up vs Down).
====================================================

    python -m ml_model.inspect_keywords

Uses the same windows as training.  If Up days do not show higher bullish
(or lower bearish) scores on average, keyword groups may need tuning.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml_model.data.keyword_extractor import GROUP_NAMES
from ml_model.data.price_fetcher import get_price_labels
from ml_model.data.window_builder import build_windows_with_metadata
from ml_model.pipeline_config import PipelineConfig


def main() -> None:
    config = PipelineConfig()
    if not config.use_keywords:
        print("use_keywords=False — enable keywords in pipeline_config to inspect.")
        return

    price_df = get_price_labels(config)
    X, y, _meta = build_windows_with_metadata(config, price_df)
    # Last day of window, keyword slice (after FinBERT dims).
    kw = X[:, -1, config.finbert_dim :].numpy()
    labels = y.numpy()
    names = list(config.class_names)

    print(f"\nMean keyword features by label (n={len(labels)}, label_mode={config.label_mode})")
    print(f"{'group':<14} ", end="")
    for name in names:
        print(f"{name:>10}", end="")
    print()
    print("-" * (14 + 10 * len(names)))

    for g, col in enumerate(GROUP_NAMES):
        print(f"{col:<14} ", end="")
        for cls in range(config.num_classes):
            mask = labels == cls
            if mask.sum() == 0:
                print(f"{'n/a':>10}", end="")
            else:
                print(f"{kw[mask, g].mean():>10.4f}", end="")
        print()

    if config.num_classes == 2:
        up_mask, down_mask = labels == 1, labels == 0
        if up_mask.sum() and down_mask.sum():
            diff = kw[up_mask].mean(axis=0) - kw[down_mask].mean(axis=0)
            print("\nUp minus Down (positive => higher on Up days):")
            for g, col in enumerate(GROUP_NAMES):
                print(f"  {col}: {diff[g]:+.4f}")


if __name__ == "__main__":
    main()
