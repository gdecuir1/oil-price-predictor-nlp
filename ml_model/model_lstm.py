"""
Bidirectional LSTM with additive attention for oil direction classification.
============================================================================

Maps a sequence of daily FinBERT embeddings ``(batch, window_days, 768)``
to 3-class logits (Down / Flat / Up) plus per-day attention weights for
interpretability.

Architecture
------------
::

    Input (B, T, 768)
      → Linear 768 → lstm_hidden
      → Dropout
      → BiLSTM → (B, T, lstm_hidden * 2)
      → Dropout
      → Additive attention → (B, lstm_hidden * 2)
      → MLP → logits (B, 3)

**Bidirectional** means two LSTM passes — forward and backward in time —
are concatenated.  The model can use both earlier and later days in the
window when forming each position's hidden state, which helps when a shock
on day 4 reframes context from day 1.

**Attention pooling** learns a soft weighting over days instead of using only
the last timestep.  High ``attention_weights[b, t]`` means day *t* strongly
influenced the prediction for batch item *b*.

**Projection layer** (768 → ``lstm_hidden``) shrinks the input before the LSTM,
reducing parameter count and training time on CPU — critical for the <30 min
budget at ~1k articles.

Additive (Bahdanau-style) attention math
----------------------------------------
For LSTM outputs ``h_t`` (shape ``lstm_hidden * 2``), learnable ``W``, ``v``:

    score_t = v^T · tanh(W · h_t)        (scalar per timestep t)
    alpha_t = softmax(score)_t              over t = 1 … T
    context = Σ_t alpha_t · h_t             (weighted sum → vector)

``forward`` returns ``alpha`` as ``attention_weights`` with shape ``(B, T)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pipeline_config import PipelineConfig


class OilLSTMPredictor(nn.Module):
  """BiLSTM + additive attention + 3-class head for oil direction.

  Args:
      config: :class:`~ml_model.pipeline_config.PipelineConfig` with architecture
          fields (``lstm_hidden``, ``lstm_layers``, ``dropout``, etc.).
  """

  def __init__(self, config: PipelineConfig) -> None:
    """Build projection, BiLSTM, attention, and classifier MLP.

    Args:
        config: Pipeline configuration instance.

    Note:
        Only parameters in this module are trained; FinBERT lives in
        ``window_builder`` and is frozen.
    """
    super().__init__()
    self.config = config
    self.window_days = config.window_days

    # Project FinBERT dim down before LSTM for efficiency.
    self.input_proj = nn.Linear(config.embed_dim, config.lstm_hidden)
    self.input_drop = nn.Dropout(config.dropout)

    self.lstm = nn.LSTM(
        input_size=config.lstm_hidden,
        hidden_size=config.lstm_hidden,
        num_layers=config.lstm_layers,
        batch_first=True,
        dropout=config.dropout if config.lstm_layers > 1 else 0.0,
        bidirectional=config.bidirectional,
    )
    self.lstm_drop = nn.Dropout(config.dropout)

    out_dim = config.lstm_output_dim
    # Bahdanau-style additive attention: W projects hidden, v scores each step.
    self.attn_w = nn.Linear(out_dim, out_dim, bias=True)
    self.attn_v = nn.Linear(out_dim, 1, bias=False)

    self.classifier = nn.Sequential(
        nn.Linear(out_dim, 64),
        nn.ReLU(),
        nn.Dropout(config.dropout),
        nn.Linear(64, 3),
    )

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full forward pass.

    Args:
        x: Input tensor ``(batch, window_days, embed_dim)``.

    Returns:
        Tuple ``(logits, attention_weights)``:

        * **logits** — ``(batch, 3)`` unnormalised class scores.
        * **attention_weights** — ``(batch, window_days)`` softmax weights
          over days (sum to 1 per row).

    Raises:
        ValueError: If input rank or feature dimension is wrong.
    """
    if x.dim() != 3:
      raise ValueError(f"Expected (B, T, D) input, got shape {tuple(x.shape)}")
    if x.size(-1) != self.config.embed_dim:
      raise ValueError(
          f"Expected embed_dim={self.config.embed_dim}, got {x.size(-1)}"
      )

    # (B, T, lstm_hidden)
    h = self.input_drop(F.relu(self.input_proj(x)))
    lstm_out, _ = self.lstm(h)
    lstm_out = self.lstm_drop(lstm_out)

    # Additive attention scores per timestep.
    # energy_t = v^T tanh(W h_t)
    energy = self.attn_v(torch.tanh(self.attn_w(lstm_out))).squeeze(-1)
    attention_weights = F.softmax(energy, dim=1)
    # context = sum_t alpha_t * h_t
    context = torch.bmm(
        attention_weights.unsqueeze(1), lstm_out
    ).squeeze(1)

    logits = self.classifier(context)
    return logits, attention_weights

  def count_parameters(self) -> int:
    """Count trainable parameters (LSTM + attention + head only).

    Returns:
        Number of elements with ``requires_grad=True``.
    """
    return sum(p.numel() for p in self.parameters() if p.requires_grad)
