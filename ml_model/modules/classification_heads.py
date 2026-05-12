"""
Multi-Task Classification Heads
================================

Each sub-domain prediction task gets its own classification head — a
small MLP that maps the pooled encoder representation to class logits.
The heads are independent so that gradients from one task don't directly
interfere with another's parameters, while the shared encoder backbone
transfers knowledge between tasks.

The :class:`MultiTaskClassificationHead` wraps all individual heads and
provides a unified forward pass that returns a dictionary of logits
keyed by sub-domain name.

Includes an **attention-based pooler** (``AttentionPooling``) as an
alternative to CLS-token pooling, which can extract richer global
representations from the full sequence of hidden states.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Registry of sub-domain keys (imported from config at module level)
# ======================================================================
# Lazy import to avoid circular dependency — populated on first access.
SUBDOMAIN_REGISTRY: Dict[str, int] = {}


def _ensure_registry() -> Dict[str, int]:
    """Populate the sub-domain registry from config if empty.

    Returns:
        Mapping of sub-domain key to number of classes.
    """
    if not SUBDOMAIN_REGISTRY:
        from ..config import SUBDOMAIN_SPECS
        for key, spec in SUBDOMAIN_SPECS.items():
            SUBDOMAIN_REGISTRY[key] = len(spec.labels)
    return SUBDOMAIN_REGISTRY


# ======================================================================
# Pooling strategies
# ======================================================================

class CLSPooler(nn.Module):
    """Extract the [CLS] token representation and project it.

    BERT-style pooling: take the first token's hidden state and pass
    it through a dense layer + tanh activation.

    Args:
        d_model: Input hidden dimensionality.
    """

    def __init__(self, d_model: int) -> None:
        """Build the dense projection layer."""
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pool the CLS token.

        Args:
            hidden_states: ``(batch, seq_len, d_model)``

        Returns:
            ``(batch, d_model)`` pooled representation.
        """
        cls_token = hidden_states[:, 0, :]
        return self.activation(self.dense(cls_token))


class MeanPooler(nn.Module):
    """Average all non-padding token representations.

    Produces a more democratic representation than CLS pooling — every
    token contributes equally — which can be advantageous when the CLS
    token hasn't been explicitly trained as a summary vector.

    Args:
        d_model: Hidden dimensionality (unused, kept for interface symmetry).
    """

    def __init__(self, d_model: int) -> None:
        """Initialise (no learnable parameters)."""
        super().__init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute mean-pooled representation.

        Args:
            hidden_states: ``(batch, seq_len, d_model)``
            attention_mask: ``(batch, seq_len)`` with 1 for real tokens, 0 for padding.

        Returns:
            ``(batch, d_model)`` mean-pooled vector.
        """
        if attention_mask is None:
            return hidden_states.mean(dim=1)

        # Expand mask for broadcasting: (B, S) → (B, S, 1)
        mask = attention_mask.unsqueeze(-1).float()
        # Zero out padding positions and compute mean over real tokens
        summed = (hidden_states * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1e-9)
        return summed / count


class AttentionPooling(nn.Module):
    """Learned attention-weighted pooling over the sequence.

    Instead of treating all tokens equally (mean pool) or privileging
    a single token (CLS pool), this module learns which tokens are
    most informative for classification via a lightweight attention
    mechanism.

    .. math::
        \\alpha_i = \\text{softmax}(\\mathbf{w}^T \\tanh(W h_i + b))

        \\text{output} = \\sum_i \\alpha_i h_i

    Args:
        d_model: Hidden dimensionality.
    """

    def __init__(self, d_model: int) -> None:
        """Build the attention scoring network."""
        super().__init__()
        self.attention_vector = nn.Linear(d_model, 1, bias=False)
        self.projection = nn.Linear(d_model, d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention-weighted pool.

        Args:
            hidden_states: ``(batch, seq_len, d_model)``
            attention_mask: ``(batch, seq_len)``

        Returns:
            ``(batch, d_model)`` pooled representation.
        """
        projected = torch.tanh(self.projection(hidden_states))
        # (B, S, 1) → (B, S)
        scores = self.attention_vector(projected).squeeze(-1)

        # Mask out padding tokens before softmax
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        # Weighted sum: (B, 1, S) × (B, S, D) → (B, D)
        pooled = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return pooled


def get_pooler(strategy: str, d_model: int) -> nn.Module:
    """Factory function for pooling strategies.

    Args:
        strategy: One of ``"cls"``, ``"mean"``, ``"attention_pool"``.
        d_model: Hidden dimensionality.

    Returns:
        Pooling module instance.

    Raises:
        ValueError: If the strategy is unknown.
    """
    if strategy == "cls":
        return CLSPooler(d_model)
    elif strategy == "mean":
        return MeanPooler(d_model)
    elif strategy == "attention_pool":
        return AttentionPooling(d_model)
    else:
        raise ValueError(f"Unknown pooling strategy: {strategy}")


# ======================================================================
# Single classification head
# ======================================================================

class ClassificationHead(nn.Module):
    """MLP classification head for a single sub-domain task.

    Architecture::

        pooled → Dropout → Dense(d_model, hidden_dim) → GELU
               → Dropout → Dense(hidden_dim, n_classes) → logits

    The two-layer design with a bottleneck hidden dimension provides
    enough capacity for non-linear decision boundaries while keeping
    the head lightweight relative to the shared encoder.

    Args:
        d_model: Input dimensionality (from the pooler).
        hidden_dim: Bottleneck hidden dimension.
        n_classes: Number of output classes for this sub-domain.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        n_classes: int,
        dropout: float = 0.3,
    ) -> None:
        """Build the classification MLP."""
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

        # Xavier init for the linear layers
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """Produce class logits from a pooled representation.

        Args:
            pooled: ``(batch, d_model)`` pooled encoder output.

        Returns:
            ``(batch, n_classes)`` raw logits (pre-softmax).
        """
        return self.head(pooled)


# ======================================================================
# Multi-task head wrapper
# ======================================================================

class MultiTaskClassificationHead(nn.Module):
    """Wraps one :class:`ClassificationHead` per sub-domain.

    Provides a single ``forward()`` that returns a dictionary of logits
    for all sub-domains simultaneously, making the training loop cleaner.

    Args:
        d_model: Encoder output dimensionality.
        hidden_dim: Hidden dim for each head's MLP.
        dropout: Head dropout rate.
        subdomain_classes: Mapping of ``{subdomain_key: n_classes}``.
            If ``None``, uses the global registry from ``config.py``.
        pool_strategy: Pooling strategy name.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        subdomain_classes: Optional[Dict[str, int]] = None,
        pool_strategy: str = "cls",
    ) -> None:
        """Instantiate one classification head per sub-domain."""
        super().__init__()

        if subdomain_classes is None:
            subdomain_classes = _ensure_registry()

        self.subdomain_keys = sorted(subdomain_classes.keys())

        # Pooler shared across all heads (no need for separate poolers)
        self.pooler = get_pooler(pool_strategy, d_model)

        # One classification head per sub-domain
        self.heads = nn.ModuleDict({
            key: ClassificationHead(
                d_model=d_model,
                hidden_dim=hidden_dim,
                n_classes=n_classes,
                dropout=dropout,
            )
            for key, n_classes in subdomain_classes.items()
        })

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Produce logits for every sub-domain.

        Args:
            hidden_states: ``(batch, seq_len, d_model)`` encoder output.
            attention_mask: ``(batch, seq_len)`` padding mask.

        Returns:
            Dict of ``{subdomain_key: (batch, n_classes)`` logit tensors}.
        """
        # Pool the sequence into a single vector
        if isinstance(self.pooler, (MeanPooler, AttentionPooling)):
            pooled = self.pooler(hidden_states, attention_mask)
        else:
            pooled = self.pooler(hidden_states)

        # Pass through each task-specific head
        return {key: self.heads[key](pooled) for key in self.subdomain_keys}
