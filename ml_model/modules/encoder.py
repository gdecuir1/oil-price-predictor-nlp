"""
Transformer Encoder Stack
=========================

Implements the encoder portion of the Transformer architecture:

* :class:`PositionWiseFeedForward` — the two-layer FFN applied to each
  position independently.
* :class:`TransformerEncoderBlock` — one encoder layer combining
  multi-head self-attention and feed-forward sub-layers with
  pre-LayerNorm residual connections.
* :class:`TransformerEncoderStack` — N stacked encoder blocks forming
  the complete encoder.

Design choices:
    * **Pre-norm** (LayerNorm before each sub-layer) rather than
      post-norm, following GPT-2 / modern best practice.  Pre-norm
      is more stable for deep networks and converges faster.
    * **GELU activation** by default (used by BERT/GPT), with ReLU
      as an option.
    * **Stochastic depth** (layer dropout) — randomly skips entire
      encoder blocks during training to regularise deep stacks.
"""

from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadSelfAttention


class PositionWiseFeedForward(nn.Module):
    """Two-layer position-wise feed-forward network.

    Applied identically to each position in the sequence:

    .. math::
        \\text{FFN}(x) = \\text{Activation}(x W_1 + b_1) W_2 + b_2

    The inner dimension ``d_ff`` is typically 4× the model dimension.

    Args:
        d_model: Input and output dimensionality.
        d_ff: Inner (hidden) dimensionality.
        dropout: Dropout after the first linear layer.
        activation: ``"gelu"`` or ``"relu"``.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        """Build the two linear layers with activation and dropout."""
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network to each position.

        Args:
            x: ``(batch, seq_len, d_model)`` input.

        Returns:
            ``(batch, seq_len, d_model)`` output.
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder layer with pre-norm residual connections.

    Architecture::

        x → LayerNorm → MultiHeadAttention → + → LayerNorm → FFN → +
        │                                    ↑    │                   ↑
        └────────────────────────────────────┘    └───────────────────┘
                    (residual)                        (residual)

    Args:
        d_model: Hidden dimensionality.
        n_heads: Number of attention heads.
        d_ff: Feed-forward inner dimension.
        dropout: General dropout rate.
        attention_dropout: Dropout on attention weights.
        activation: FFN activation function.
        layer_norm_eps: Epsilon for LayerNorm.
        cache_attention: Whether to cache attention weights.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-12,
        cache_attention: bool = False,
    ) -> None:
        """Assemble the attention, FFN, and normalisation sub-layers."""
        super().__init__()

        # Pre-norm layer normalisations
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Multi-head self-attention sub-layer
        self.self_attention = MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            cache_attention=cache_attention,
        )

        # Position-wise feed-forward sub-layer
        self.feed_forward = PositionWiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process one encoder layer.

        Args:
            x: ``(batch, seq_len, d_model)`` input hidden states.
            mask: Optional attention mask.

        Returns:
            ``(batch, seq_len, d_model)`` updated hidden states.
        """
        # Sub-layer 1: Self-attention with pre-norm residual
        residual = x
        x = self.norm1(x)
        x = self.self_attention(x, mask=mask)
        x = residual + self.dropout(x)

        # Sub-layer 2: Feed-forward with pre-norm residual
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = residual + self.dropout(x)

        return x


class TransformerEncoderStack(nn.Module):
    """Stack of N transformer encoder blocks with optional stochastic depth.

    Stochastic depth randomly drops entire layers during training
    (with a linearly increasing drop probability from the first to
    the last layer).  At test time all layers are active.  This
    technique regularises deep networks and improves generalisation
    on small datasets.

    Args:
        n_layers: Number of encoder blocks to stack.
        d_model: Hidden dimensionality.
        n_heads: Number of attention heads per block.
        d_ff: Feed-forward inner dimension.
        dropout: General dropout rate.
        attention_dropout: Attention-specific dropout.
        activation: FFN activation.
        layer_norm_eps: LayerNorm epsilon.
        stochastic_depth_rate: Maximum layer-drop probability (applied
            linearly — layer 0 has rate 0, last layer has this rate).
            Set to 0.0 to disable.
        cache_attention: Cache attention weights in the last layer.
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-12,
        stochastic_depth_rate: float = 0.1,
        cache_attention: bool = False,
    ) -> None:
        """Construct the encoder stack."""
        super().__init__()

        self.n_layers = n_layers
        self.stochastic_depth_rate = stochastic_depth_rate

        # Build encoder blocks — only cache attention in the final layer
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                cache_attention=(cache_attention and i == n_layers - 1),
            )
            for i in range(n_layers)
        ])

        # Final layer norm after the last block (pre-norm convention)
        self.final_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through all encoder layers.

        Args:
            x: ``(batch, seq_len, d_model)`` embedded input.
            mask: Optional attention mask.

        Returns:
            ``(batch, seq_len, d_model)`` final encoder hidden states.
        """
        for i, layer in enumerate(self.layers):
            # Stochastic depth: skip layers randomly during training
            if self.training and self.stochastic_depth_rate > 0:
                drop_prob = self.stochastic_depth_rate * (i / max(self.n_layers - 1, 1))
                if random.random() < drop_prob:
                    continue

            x = layer(x, mask=mask)

        return self.final_norm(x)
