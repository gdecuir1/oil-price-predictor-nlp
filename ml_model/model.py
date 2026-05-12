"""
Oil Market Prediction Transformer — Main Model
================================================

This module defines :class:`OilMarketTransformer`, the top-level ``nn.Module``
that orchestrates the full prediction pipeline:

    raw tokens → encoder → pooling → multi-task classification heads → logits

The model supports **two operating modes** controlled by the configuration:

1. **Fine-tune mode** (``use_pretrained=True``, default)
   Loads a pre-trained transformer encoder (FinBERT, BERT, DistilBERT, etc.)
   from HuggingFace and attaches fresh multi-task classification heads.
   This is the recommended mode given the relatively small dataset (~353
   articles) — transfer learning from a finance-domain model provides
   strong initialisation that pseudo-labelling alone cannot match.

2. **From-scratch mode** (``use_pretrained=False``)
   Builds a custom transformer encoder from the building blocks in
   :mod:`ml_model.modules` (custom embeddings, multi-head attention,
   encoder stack).  Useful for research / ablation studies or when the
   dataset grows large enough to train from scratch.

Both modes produce identical output shapes — a dict of logit tensors
keyed by sub-domain name — so the training loop and inference pipeline
are mode-agnostic.

Architecture diagram (fine-tune mode)::

    ┌─────────────────────────────────────────────────────┐
    │                  Pre-trained Encoder                 │
    │  (FinBERT / BERT)                                   │
    │  [CLS] tok1 tok2 ... tokN [SEP] [PAD] ...          │
    │    ↓                                                │
    │  (batch, seq_len, 768) hidden states                │
    └────────────────────┬────────────────────────────────┘
                         │
                    ┌────▼────┐
                    │ Pooler  │  (CLS / Mean / Attention)
                    └────┬────┘
                         │
              ┌──────────┼──────────┐──────────┐
              ▼          ▼          ▼          ▼
         ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
         │ Head:  │ │ Head:  │ │ Head:  │ │ Head:  │
         │ market │ │ price  │ │ senti- │ │ geo-   │
         │ dir.   │ │ magni. │ │ ment   │ │ risk   │
         │ (3 cls)│ │ (5 cls)│ │ (5 cls)│ │ (5 cls)│ ... (8 heads total)
         └────────┘ └────────┘ └────────┘ └────────┘
              ↓          ↓          ↓          ↓
           logits     logits     logits     logits

Usage::

    from ml_model.config import Config
    from ml_model.model import OilMarketTransformer

    cfg = Config()
    model = OilMarketTransformer(cfg)
    outputs = model(input_ids, attention_mask)
    # outputs: {"market_direction": (B, 3), "sentiment": (B, 5), ...}
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .config import Config, ModelConfig
from .modules.classification_heads import MultiTaskClassificationHead
from .modules.embeddings import TransformerEmbedding
from .modules.encoder import TransformerEncoderStack

logger = logging.getLogger(__name__)


class OilMarketTransformer(nn.Module):
    """Multi-task transformer for US oil market prediction.

    Combines a shared encoder backbone (pre-trained or custom) with
    independent classification heads for each prediction sub-domain.

    The model is designed to:
        * Accept tokenised article text as input.
        * Produce simultaneous predictions across all sub-domains.
        * Support encoder freezing during early training epochs.
        * Cache attention weights in the final encoder layer for
          interpretability / report generation.
        * Work with mixed-precision (FP16) training out of the box.

    Args:
        config: Top-level :class:`Config` object.  If ``None``, uses
            all defaults.

    Attributes:
        encoder: The transformer encoder (pre-trained or custom).
        classification_heads: :class:`MultiTaskClassificationHead` producing
            logits for all sub-domains.
        is_pretrained: Whether the encoder is a HuggingFace pre-trained model.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        """Construct the model from configuration.

        Depending on ``config.model.use_pretrained``, either loads a
        HuggingFace encoder or builds a custom transformer from scratch.
        Classification heads are always freshly initialised.
        """
        super().__init__()
        self.config = config or Config()
        self.model_config: ModelConfig = self.config.model
        self.is_pretrained = self.model_config.use_pretrained

        # -----------------------------------------------------------
        # Build the encoder backbone
        # -----------------------------------------------------------
        if self.is_pretrained:
            self.encoder = self._build_pretrained_encoder()
            # The pre-trained model's hidden size may differ from d_model
            encoder_dim = self.encoder.config.hidden_size
        else:
            self.encoder = self._build_custom_encoder()
            encoder_dim = self.model_config.d_model

        # -----------------------------------------------------------
        # Build multi-task classification heads
        # -----------------------------------------------------------
        subdomain_classes = self.config.total_output_classes
        self.classification_heads = MultiTaskClassificationHead(
            d_model=encoder_dim,
            hidden_dim=self.model_config.classifier_hidden_dim,
            dropout=self.model_config.classifier_dropout,
            subdomain_classes=subdomain_classes,
            pool_strategy=self.model_config.pool_strategy,
        )

        # Log model summary
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "OilMarketTransformer built: mode=%s, encoder_dim=%d, "
            "subdomains=%d, total_params=%s, trainable_params=%s",
            "pretrained" if self.is_pretrained else "custom",
            encoder_dim,
            len(subdomain_classes),
            f"{total_params:,}",
            f"{trainable_params:,}",
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass: encode → pool → classify.

        Args:
            input_ids: ``(batch, seq_len)`` integer token IDs.
            attention_mask: ``(batch, seq_len)`` mask with 1 for real
                tokens and 0 for padding.
            token_type_ids: ``(batch, seq_len)`` segment IDs (only used
                by some pre-trained models).

        Returns:
            Dictionary mapping sub-domain keys to logit tensors:
            ``{subdomain_key: (batch, n_classes)}``.
        """
        # Encode the input sequence
        hidden_states = self._encode(input_ids, attention_mask, token_type_ids)

        # Pass encoder output through multi-task classification heads
        logits = self.classification_heads(hidden_states, attention_mask)

        return logits

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Convenience method that returns class predictions and probabilities.

        Unlike :meth:`forward` which returns raw logits, this method
        applies softmax and returns the predicted class index, class
        label, and full probability distribution for each sub-domain.

        Args:
            input_ids: ``(batch, seq_len)`` token IDs.
            attention_mask: ``(batch, seq_len)`` attention mask.
            token_type_ids: ``(batch, seq_len)`` segment IDs.

        Returns:
            Nested dict: ``{subdomain_key: {"predicted_class": int,
            "predicted_label": str, "probabilities": Tensor,
            "confidence": float}}``.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask, token_type_ids)

        predictions: Dict[str, Dict[str, Any]] = {}
        for key, logit_tensor in logits.items():
            probs = torch.softmax(logit_tensor, dim=-1)
            pred_class = probs.argmax(dim=-1)
            confidence = probs.max(dim=-1).values

            # Map class indices to human-readable labels
            labels = self.config.subdomains[key].labels
            pred_labels = [labels[idx.item()] for idx in pred_class]

            predictions[key] = {
                "predicted_class": pred_class,
                "predicted_label": pred_labels,
                "probabilities": probs,
                "confidence": confidence,
            }

        return predictions

    # ------------------------------------------------------------------
    # Encoder construction
    # ------------------------------------------------------------------

    def _build_pretrained_encoder(self) -> nn.Module:
        """Load a HuggingFace pre-trained transformer encoder.

        Uses AutoModel to support any BERT-like architecture.  The
        model is loaded with its original weights; fine-tuning happens
        during training.

        Returns:
            HuggingFace model (e.g. ``BertModel``).
        """
        from transformers import AutoModel

        model_name = self.model_config.pretrained_model_name
        logger.info("Loading pre-trained encoder: %s", model_name)

        encoder = AutoModel.from_pretrained(model_name)
        return encoder

    def _build_custom_encoder(self) -> nn.Module:
        """Build a custom transformer encoder from scratch.

        Constructs the embedding layer and encoder stack using the
        building blocks defined in :mod:`ml_model.modules`.

        Returns:
            :class:`CustomTransformerEncoder` wrapping embeddings + stack.
        """
        logger.info(
            "Building custom encoder: %d layers, d_model=%d, %d heads",
            self.model_config.n_encoder_layers,
            self.model_config.d_model,
            self.model_config.n_heads,
        )

        return CustomTransformerEncoder(
            vocab_size=self.model_config.vocab_size,
            d_model=self.model_config.d_model,
            n_heads=self.model_config.n_heads,
            n_layers=self.model_config.n_encoder_layers,
            d_ff=self.model_config.d_feedforward,
            max_len=self.model_config.max_position_embeddings,
            dropout=self.model_config.dropout,
            attention_dropout=self.model_config.attention_dropout,
            activation=self.model_config.activation,
            layer_norm_eps=self.model_config.layer_norm_eps,
        )

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        token_type_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Route encoding through the appropriate backend.

        Args:
            input_ids: Token IDs.
            attention_mask: Padding mask.
            token_type_ids: Segment IDs.

        Returns:
            ``(batch, seq_len, d_model)`` hidden states from the encoder.
        """
        if self.is_pretrained:
            # HuggingFace models return a BaseModelOutput with .last_hidden_state
            kwargs: Dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids

            outputs = self.encoder(**kwargs)
            return outputs.last_hidden_state
        else:
            # Custom encoder takes token IDs and an inverted mask
            return self.encoder(input_ids, attention_mask)

    # ------------------------------------------------------------------
    # Encoder management utilities
    # ------------------------------------------------------------------

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters (disable gradient computation).

        Used during the initial training epochs to let the classification
        heads warm up before fine-tuning disturbs the pre-trained weights.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder parameters frozen")

    def unfreeze_encoder(self) -> None:
        """Unfreeze all encoder parameters for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        logger.info("Encoder parameters unfrozen")

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Retrieve cached attention weights from the last encoder layer.

        Only available when ``cache_attention=True`` in the custom
        encoder.  For pre-trained models, use
        ``output_attentions=True`` in the forward pass instead.

        Returns:
            Attention weight tensor or ``None`` if not available.
        """
        if not self.is_pretrained and hasattr(self.encoder, "encoder_stack"):
            last_layer = self.encoder.encoder_stack.layers[-1]
            return last_layer.self_attention.cached_attn_weights
        return None


class CustomTransformerEncoder(nn.Module):
    """Wraps :class:`TransformerEmbedding` + :class:`TransformerEncoderStack`.

    This provides the same interface as a HuggingFace model (takes
    ``input_ids`` and ``attention_mask``, returns hidden states) so
    that the parent :class:`OilMarketTransformer` can use either
    backend transparently.

    Args:
        vocab_size: Tokenizer vocabulary size.
        d_model: Hidden dimensionality.
        n_heads: Number of attention heads.
        n_layers: Number of encoder layers.
        d_ff: Feed-forward inner dimension.
        max_len: Maximum sequence length.
        dropout: Dropout rate.
        attention_dropout: Attention-specific dropout.
        activation: FFN activation function.
        layer_norm_eps: LayerNorm epsilon.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_len: int = 512,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-12,
    ) -> None:
        """Construct the embedding layer and encoder stack."""
        super().__init__()

        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
            padding_idx=0,
            position_mode="learned",
            layer_norm_eps=layer_norm_eps,
        )

        self.encoder_stack = TransformerEncoderStack(
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            stochastic_depth_rate=0.1,
            cache_attention=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embed tokens and pass through the encoder stack.

        Args:
            input_ids: ``(batch, seq_len)`` token IDs.
            attention_mask: ``(batch, seq_len)`` with 1 for real tokens.

        Returns:
            ``(batch, seq_len, d_model)`` encoder hidden states.
        """
        # Create the 4-D attention mask expected by the attention layers
        # Shape: (batch, 1, 1, seq_len) — broadcastable across heads and query positions
        extended_mask = None
        if attention_mask is not None:
            # Convert 0/1 mask to True/False where True = masked (ignored)
            extended_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(1)

        x = self.embedding(input_ids)
        x = self.encoder_stack(x, mask=extended_mask)
        return x
