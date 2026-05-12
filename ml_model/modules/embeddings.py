"""
Embedding Layers for the Custom Transformer
=============================================

Provides token embeddings, positional encodings, and their composition
for the from-scratch transformer encoder.  When using a pre-trained
backbone (FinBERT/BERT), these are not used — the backbone supplies
its own embedding layer.

Two positional encoding strategies are implemented:
    * **Learned** — a trainable embedding matrix (like BERT).
    * **Sinusoidal** — fixed sin/cos encoding (like the original Transformer).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Trainable token embedding lookup table with scaling.

    Scales embeddings by √d_model following Vaswani et al. (2017) to
    keep the magnitude of positional and token embeddings comparable.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        d_model: Embedding dimensionality.
        padding_idx: Index of the padding token (embeddings zeroed out).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: int = 0,
    ) -> None:
        """Initialise the embedding layer."""
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx,
        )
        # Xavier-uniform init for non-padding entries
        nn.init.normal_(self.embedding.weight, mean=0.0, std=d_model ** -0.5)
        if padding_idx is not None:
            nn.init.zeros_(self.embedding.weight[padding_idx])

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up token embeddings and apply scaling.

        Args:
            token_ids: ``(batch, seq_len)`` integer token IDs.

        Returns:
            ``(batch, seq_len, d_model)`` scaled embedding vectors.
        """
        return self.embedding(token_ids) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Injects positional information into token embeddings.

    Supports two modes:

    * ``learned`` — a trainable ``nn.Embedding`` (used by BERT-style
      models).  Allows the model to learn arbitrary position-dependent
      patterns.
    * ``sinusoidal`` — fixed sin/cos encodings from "Attention Is All
      You Need".  No extra parameters; generalises to unseen lengths.

    Args:
        d_model: Dimensionality of the embedding space.
        max_len: Maximum sequence length supported.
        dropout: Dropout applied after adding positional encoding.
        mode: ``"learned"`` or ``"sinusoidal"``.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1,
        mode: str = "learned",
    ) -> None:
        """Construct positional encoding of the specified type."""
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.mode = mode

        if mode == "learned":
            # Trainable position embeddings — each position gets its own vector
            self.position_embedding = nn.Embedding(max_len, d_model)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        else:
            # Sinusoidal — precomputed and registered as a non-trainable buffer
            pe = self._build_sinusoidal_encoding(d_model, max_len)
            self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to the input embeddings.

        Args:
            x: ``(batch, seq_len, d_model)`` token embeddings.

        Returns:
            ``(batch, seq_len, d_model)`` position-aware embeddings.
        """
        seq_len = x.size(1)

        if self.mode == "learned":
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            x = x + self.position_embedding(positions)
        else:
            x = x + self.pe[:, :seq_len, :]

        return self.dropout(x)

    @staticmethod
    def _build_sinusoidal_encoding(d_model: int, max_len: int) -> torch.Tensor:
        """Compute the sinusoidal positional encoding matrix.

        Uses the formula from Vaswani et al.:
            PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
            PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        Args:
            d_model: Embedding dimensionality.
            max_len: Maximum sequence length.

        Returns:
            ``(1, max_len, d_model)`` encoding tensor.
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Compute the division term in log-space for numerical stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # add batch dimension


class TransformerEmbedding(nn.Module):
    """Composed embedding: token lookup + positional encoding + LayerNorm.

    This is the complete input embedding stack for the custom (non-pretrained)
    transformer.  It mirrors BERT's input processing:

        output = LayerNorm(Dropout(TokenEmb(ids) + PosEnc(positions)))

    Args:
        vocab_size: Vocabulary size.
        d_model: Embedding dimensionality.
        max_len: Maximum sequence length.
        dropout: Dropout probability.
        padding_idx: Padding token index.
        position_mode: ``"learned"`` or ``"sinusoidal"``.
        layer_norm_eps: Epsilon for layer normalisation.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1,
        padding_idx: int = 0,
        position_mode: str = "learned",
        layer_norm_eps: float = 1e-12,
    ) -> None:
        """Compose the token and positional embedding layers."""
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model, padding_idx)
        self.position_encoding = PositionalEncoding(
            d_model, max_len, dropout, mode=position_mode
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed, encode positions, normalise, and apply dropout.

        Args:
            token_ids: ``(batch, seq_len)`` integer token IDs.

        Returns:
            ``(batch, seq_len, d_model)`` final embeddings.
        """
        x = self.token_embedding(token_ids)
        x = self.position_encoding(x)
        x = self.layer_norm(x)
        return self.dropout(x)
