"""
Multi-Head Self-Attention
=========================

Implements the multi-head self-attention mechanism from "Attention Is All
You Need" (Vaswani et al., 2017) with enhancements:

* **Pre-norm** residual connections (more stable for deep networks).
* **Separate attention dropout** to regularise attention distributions
  independently of hidden-state dropout.
* **Attention weight caching** so that the inference pipeline can
  retrieve per-head attention maps for interpretability / report
  generation (identifying which articles or tokens influenced the
  prediction most).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Compute scaled dot-product attention.

    .. math::
        \\text{Attention}(Q, K, V) = \\text{softmax}\\left(
            \\frac{QK^T}{\\sqrt{d_k}}
        \\right) V

    Optionally applies a causal or padding mask before the softmax.

    Args:
        dropout: Dropout probability applied to attention weights.
    """

    def __init__(self, dropout: float = 0.1) -> None:
        """Initialise with the specified dropout rate."""
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention output and weights.

        Args:
            query: ``(batch, heads, seq_q, d_k)``
            key:   ``(batch, heads, seq_k, d_k)``
            value: ``(batch, heads, seq_k, d_v)``
            mask:  Broadcastable boolean mask where ``True`` means
                   *ignore* (will be filled with ``-inf``).

        Returns:
            Tuple of:
                - Context vectors ``(batch, heads, seq_q, d_v)``
                - Attention weights ``(batch, heads, seq_q, seq_k)``
        """
        d_k = query.size(-1)
        # Scaled dot-product: (B, H, Q, K)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask — set masked positions to large negative so softmax → 0
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        context = torch.matmul(attn_weights, value)

        return context, attn_weights


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional attention-weight caching.

    Projects the input into ``n_heads`` sets of query/key/value vectors,
    applies scaled dot-product attention in parallel, concatenates the
    results, and projects back to ``d_model``.

    When ``cache_attention=True``, the last forward pass's attention
    weights are stored in ``self.cached_attn_weights`` for later
    retrieval by the interpretability / report-generation pipeline.

    Args:
        d_model: Total model dimensionality.
        n_heads: Number of parallel attention heads.
        dropout: Hidden-state dropout.
        attention_dropout: Dropout on attention weights specifically.
        cache_attention: Whether to cache attention weights.

    Raises:
        ValueError: If ``d_model`` is not divisible by ``n_heads``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        cache_attention: bool = False,
    ) -> None:
        """Set up projection matrices and attention mechanism."""
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # dimension per head
        self.cache_attention = cache_attention

        # Linear projections for Q, K, V, and output
        self.W_q = nn.Linear(d_model, d_model, bias=True)
        self.W_k = nn.Linear(d_model, d_model, bias=True)
        self.W_v = nn.Linear(d_model, d_model, bias=True)
        self.W_o = nn.Linear(d_model, d_model, bias=True)

        self.attention = ScaledDotProductAttention(dropout=attention_dropout)
        self.dropout = nn.Dropout(p=dropout)

        # Storage for cached attention weights (populated during forward)
        self.cached_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute multi-head self-attention.

        Args:
            x: ``(batch, seq_len, d_model)`` input tensor.
            mask: Optional ``(batch, 1, 1, seq_len)`` padding mask.

        Returns:
            ``(batch, seq_len, d_model)`` attention output.
        """
        batch_size, seq_len, _ = x.size()

        # Project to Q, K, V and reshape for multi-head: (B, H, S, d_k)
        Q = self._reshape_to_heads(self.W_q(x), batch_size)
        K = self._reshape_to_heads(self.W_k(x), batch_size)
        V = self._reshape_to_heads(self.W_v(x), batch_size)

        # Scaled dot-product attention across all heads simultaneously
        context, attn_weights = self.attention(Q, K, V, mask=mask)

        # Cache attention weights for interpretability
        if self.cache_attention:
            self.cached_attn_weights = attn_weights.detach()

        # Concatenate heads: (B, S, H, d_k) → (B, S, d_model)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        # Final linear projection
        output = self.W_o(context)
        return self.dropout(output)

    def _reshape_to_heads(
        self, x: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Reshape a projected tensor from (B, S, D) to (B, H, S, d_k).

        Args:
            x: ``(batch, seq_len, d_model)`` projected tensor.
            batch_size: Batch size (for clarity in the reshape).

        Returns:
            ``(batch, n_heads, seq_len, d_k)`` multi-head tensor.
        """
        seq_len = x.size(1)
        return x.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
