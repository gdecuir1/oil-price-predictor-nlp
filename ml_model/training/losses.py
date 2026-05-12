"""
Multi-Task Loss Functions
=========================

Computes the combined loss for all sub-domain prediction tasks.  Each
sub-domain contributes a weighted cross-entropy term to the total loss.

Key features:
    * **Label smoothing** — softens one-hot targets to prevent the model
      from becoming overconfident, which is critical when training on
      pseudo-labels that contain noise.
    * **Per-task weighting** — the primary task (market direction) is
      weighted more heavily than auxiliary tasks.
    * **Class-weight support** — optional inverse-frequency class weights
      to handle imbalanced label distributions.
    * **Focal loss option** — down-weights easy examples and focuses
      training on hard-to-classify samples, useful for imbalanced data.
    * **Mixup-compatible** — supports soft label targets for when Mixup
      augmentation is applied.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance.

    Focal loss modifies standard cross-entropy by adding a modulating
    factor ``(1 - p_t)^gamma`` that down-weights well-classified examples
    and focuses on hard negatives.

    .. math::
        FL(p_t) = -\\alpha_t (1 - p_t)^\\gamma \\log(p_t)

    Reference: Lin et al., "Focal Loss for Dense Object Detection", 2017.

    Args:
        gamma: Focusing parameter.  ``gamma=0`` recovers standard CE.
            Higher values put more focus on hard examples.
        alpha: Optional per-class weight tensor.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.
        label_smoothing: Smoothing factor applied before focal modulation.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        """Initialise focal loss parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: ``(batch, n_classes)`` raw model output.
            targets: ``(batch,)`` integer class labels.

        Returns:
            Scalar loss tensor.
        """
        n_classes = logits.size(-1)

        # Apply label smoothing to create soft targets
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.full_like(
                    logits, self.label_smoothing / (n_classes - 1)
                )
                smooth_targets.scatter_(
                    1, targets.unsqueeze(1), 1.0 - self.label_smoothing
                )
        else:
            smooth_targets = F.one_hot(targets, n_classes).float()

        # Compute log-probabilities and probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Focal modulating factor: (1 - p_t)^gamma
        focal_weight = (1.0 - probs) ** self.gamma

        # Weighted cross-entropy with focal modulation
        loss = -focal_weight * smooth_targets * log_probs

        # Apply per-class alpha weights if provided
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            loss = loss * alpha.unsqueeze(0)

        # Sum over classes, then reduce over batch
        loss = loss.sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MultiTaskLoss(nn.Module):
    """Aggregated loss across all sub-domain prediction tasks.

    Computes a weighted sum of per-task losses::

        L_total = sum_k  w_k * L_k(logits_k, targets_k)

    where ``w_k`` is the configured loss weight for sub-domain ``k``
    and ``L_k`` is either cross-entropy or focal loss.

    Supports **uncertainty-based automatic loss weighting** (Kendall et al.,
    2018) as an alternative to fixed weights.  When enabled, each task's
    weight is learned as a log-variance parameter, and the loss becomes::

        L_total = sum_k  (1 / 2σ_k²) * L_k + log(σ_k)

    This allows the model to automatically balance tasks based on their
    inherent difficulty and noise level.

    Args:
        subdomain_weights: Dict mapping sub-domain keys to fixed loss
            weights.  Ignored if ``auto_weight=True``.
        label_smoothing: Smoothing factor for cross-entropy.
        use_focal: Use focal loss instead of standard cross-entropy.
        focal_gamma: Focal loss gamma parameter.
        auto_weight: Enable learned uncertainty-based task weighting.
        class_weights: Optional dict of per-task class weight tensors
            for handling class imbalance within each sub-domain.
    """

    def __init__(
        self,
        subdomain_weights: Optional[Dict[str, float]] = None,
        label_smoothing: float = 0.1,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
        auto_weight: bool = False,
        class_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Set up loss functions for each sub-domain."""
        super().__init__()

        from ..config import SUBDOMAIN_SPECS

        self.subdomain_keys = sorted(SUBDOMAIN_SPECS.keys())
        self.auto_weight = auto_weight

        # Fixed task weights from config
        if subdomain_weights is None:
            self.fixed_weights = {
                k: SUBDOMAIN_SPECS[k].loss_weight for k in self.subdomain_keys
            }
        else:
            self.fixed_weights = subdomain_weights

        # Learnable log-variance parameters for auto-weighting
        if auto_weight:
            self.log_vars = nn.ParameterDict({
                key: nn.Parameter(torch.zeros(1))
                for key in self.subdomain_keys
            })

        # Build per-task loss functions
        self.loss_fns: Dict[str, nn.Module] = {}
        for key in self.subdomain_keys:
            cw = class_weights.get(key) if class_weights else None

            if use_focal:
                self.loss_fns[key] = FocalLoss(
                    gamma=focal_gamma,
                    alpha=cw,
                    label_smoothing=label_smoothing,
                )
            else:
                self.loss_fns[key] = nn.CrossEntropyLoss(
                    weight=cw,
                    label_smoothing=label_smoothing,
                    reduction="mean",
                )

        # Register as a ModuleDict so parameters are tracked
        self.loss_modules = nn.ModuleDict(
            {k: v for k, v in self.loss_fns.items() if isinstance(v, nn.Module)}
        )

    def forward(
        self,
        logits: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute the total multi-task loss and per-task breakdowns.

        Args:
            logits: ``{subdomain_key: (batch, n_classes)}`` model outputs.
            targets: ``{subdomain_key: (batch,)}`` integer class labels.

        Returns:
            Dictionary with keys:
                - ``"total"``: scalar total loss.
                - ``"per_task"``: dict of per-task scalar losses.
                - ``"weights"``: dict of effective per-task weights used.
        """
        per_task_losses: Dict[str, torch.Tensor] = {}
        effective_weights: Dict[str, float] = {}
        total_loss = torch.tensor(0.0, device=next(iter(logits.values())).device)

        for key in self.subdomain_keys:
            if key not in logits or key not in targets:
                continue

            task_loss = self.loss_fns[key](logits[key], targets[key])
            per_task_losses[key] = task_loss.detach()

            if self.auto_weight:
                # Uncertainty weighting: L_k / (2 * sigma_k^2) + log(sigma_k)
                precision = torch.exp(-self.log_vars[key])
                weighted = precision * task_loss + self.log_vars[key]
                effective_weights[key] = precision.item()
            else:
                weight = self.fixed_weights.get(key, 1.0)
                weighted = weight * task_loss
                effective_weights[key] = weight

            total_loss = total_loss + weighted

        return {
            "total": total_loss,
            "per_task": per_task_losses,
            "weights": effective_weights,
        }
