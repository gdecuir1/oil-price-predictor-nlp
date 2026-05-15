"""
Bidirectional LSTM with masked attention (v3 compact + legacy loader).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pipeline_config import PipelineConfig


class OilLSTMPredictor(nn.Module):
    """Compact BiLSTM + masked attention; input = FinBERT + keyword features."""

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        in_dim = config.input_dim
        finbert_dim = config.finbert_dim

        self.input_norm = nn.LayerNorm(in_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, config.proj_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.proj_dim, config.lstm_hidden),
            nn.LayerNorm(config.lstm_hidden),
        )
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
        self.attn_w = nn.Linear(out_dim, out_dim, bias=True)
        self.attn_v = nn.Linear(out_dim, 1, bias=False)

        self.use_residual = config.use_residual
        if self.use_residual:
            self.residual_proj = nn.Linear(in_dim, out_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, config.mlp_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden, self.num_classes),
        )
        self._finbert_dim = finbert_dim

    def _day_presence_mask(self, x: torch.Tensor) -> torch.Tensor:
        """True when the FinBERT slice of the day vector is non-zero."""
        return x[..., : self._finbert_dim].norm(dim=-1) > 1e-6

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3 or x.size(-1) != self.config.input_dim:
            raise ValueError(
                f"Expected (B, T, {self.config.input_dim}), got {tuple(x.shape)}"
            )

        presence = self._day_presence_mask(x)
        h = self.input_drop(self.input_proj(self.input_norm(x)))
        lstm_out, _ = self.lstm(h)
        lstm_out = self.lstm_drop(lstm_out)

        energy = self.attn_v(torch.tanh(self.attn_w(lstm_out))).squeeze(-1)
        energy = energy.masked_fill(~presence, float("-inf"))
        attention_weights = F.softmax(energy, dim=1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

        context = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)

        if self.use_residual:
            mask = presence.unsqueeze(-1).float()
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = (x * mask).sum(dim=1) / denom
            context = context + self.residual_proj(pooled)

        return self.classifier(context), attention_weights

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class OilLSTMPredictorLegacy(nn.Module):
    """Original small 3-class model (768-dim only, no keywords). For old .pkl files."""

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__()
        self.config = config
        self.num_classes = 3
        dim = 768

        self.input_proj = nn.Linear(dim, config.lstm_hidden)
        self.input_drop = nn.Dropout(config.dropout)
        self.lstm = nn.LSTM(
            input_size=config.lstm_hidden,
            hidden_size=config.lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=True,
        )
        self.lstm_drop = nn.Dropout(config.dropout)
        out_dim = config.lstm_hidden * 2
        self.attn_w = nn.Linear(out_dim, out_dim, bias=True)
        self.attn_v = nn.Linear(out_dim, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_drop(F.relu(self.input_proj(x)))
        lstm_out, _ = self.lstm(h)
        lstm_out = self.lstm_drop(lstm_out)
        energy = self.attn_v(torch.tanh(self.attn_w(lstm_out))).squeeze(-1)
        attention_weights = F.softmax(energy, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)
        return self.classifier(context), attention_weights

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_model_from_checkpoint(
    payload: dict,
    config: PipelineConfig,
) -> nn.Module:
    """Instantiate v3, v2, or legacy architecture from checkpoint keys.

    Args:
        payload: Pickle dict with ``model_state_dict``.
        config: Reconstructed :class:`PipelineConfig`.

    Returns:
        Module in eval mode with weights loaded.
    """
    state = payload["model_state_dict"]
    if "input_proj.0.weight" in state:
        model = OilLSTMPredictor(config)
    elif "input_proj.weight" in state:
        model = OilLSTMPredictorLegacy(config)
    else:
        model = OilLSTMPredictor(config)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model
