"""
Text Preprocessor
=================

Handles tokenisation, encoding, and augmentation of extracted article text
for consumption by the transformer model.  Wraps a HuggingFace tokenizer
and adds domain-specific preprocessing steps (financial term normalisation,
numeric entity handling, data augmentation).

The preprocessor is the bridge between raw text strings and the integer
tensors that the model consumes.
"""

from __future__ import annotations

import logging
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Tokenises and encodes article text for the transformer.

    Handles three concerns:

    1. **Domain-specific text cleaning** — normalises financial jargon,
       masks specific numeric values to help the model generalise, and
       removes encoding artefacts.
    2. **Tokenisation** — delegates to a HuggingFace tokenizer with
       configurable max length and padding strategy.
    3. **Data augmentation** — optional per-sample augmentations (random
       sentence dropout, synonym replacement) to regularise training on
       the small dataset.

    Args:
        tokenizer_name: HuggingFace tokenizer identifier.
        max_length: Maximum token count (including special tokens).
        padding: Padding strategy — ``"max_length"`` or ``"longest"``.
        truncation: Whether to truncate sequences exceeding ``max_length``.
        normalize_numbers: Replace specific dollar/barrel values with
            placeholders so the model learns patterns rather than
            memorising prices.
        augment: Enable training-time augmentation.
        augment_prob: Per-sample probability of applying each augmentation.

    Example::

        preprocessor = TextPreprocessor(
            tokenizer_name="ProsusAI/finbert",
            max_length=512,
        )
        encoded = preprocessor.encode("Oil prices rose 3% today...")
    """

    # Regex patterns for domain-specific normalisation
    _DOLLAR_PRICE = re.compile(
        r"\$\s*[\d,]+(?:\.\d{1,2})?\s*(?:per\s+barrel|/bbl|/b)?",
        re.IGNORECASE,
    )
    _PERCENTAGE = re.compile(r"[\d,]+(?:\.\d{1,2})?\s*%")
    _LARGE_NUMBER = re.compile(r"\b\d{1,3}(?:,\d{3})+\b")
    _BARREL_AMOUNT = re.compile(
        r"[\d,]+(?:\.\d+)?\s*(?:million|billion|thousand)?\s*(?:barrels|bbl|bpd|b/d)",
        re.IGNORECASE,
    )

    # Common oil / financial abbreviations to expand
    _ABBREVIATIONS = {
        "bpd": "barrels per day",
        "b/d": "barrels per day",
        "bbl": "barrel",
        "WTI": "West Texas Intermediate",
        "OPEC": "Organization of the Petroleum Exporting Countries",
        "EIA": "Energy Information Administration",
        "SPR": "Strategic Petroleum Reserve",
        "API": "American Petroleum Institute",
        "NYMEX": "New York Mercantile Exchange",
        "ICE": "Intercontinental Exchange",
    }

    def __init__(
        self,
        tokenizer_name: str = "ProsusAI/finbert",
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
        normalize_numbers: bool = True,
        augment: bool = False,
        augment_prob: float = 0.15,
    ) -> None:
        """Load the tokenizer and configure preprocessing options."""
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            tokenizer_name
        )
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.normalize_numbers = normalize_numbers
        self.augment = augment
        self.augment_prob = augment_prob

        logger.info(
            "Preprocessor initialised: tokenizer=%s, max_length=%d, augment=%s",
            tokenizer_name,
            max_length,
            augment,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, text: str) -> Dict[str, torch.Tensor]:
        """Preprocess and tokenise a single article text.

        Args:
            text: Raw or lightly-cleaned article text.

        Returns:
            Dictionary with keys ``input_ids``, ``attention_mask``, and
            (if the tokenizer produces them) ``token_type_ids``.  Each
            value is a 1-D ``torch.LongTensor`` of length ``max_length``.
        """
        text = self._preprocess(text)

        if self.augment:
            text = self._apply_augmentations(text)

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
        )

        # Squeeze batch dimension (single sample)
        return {k: v.squeeze(0) for k, v in encoding.items()}

    def encode_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenise a batch of article texts.

        Args:
            texts: List of raw article text strings.

        Returns:
            Dictionary with batched tensors of shape ``(B, max_length)``.
        """
        processed = [self._preprocess(t) for t in texts]

        if self.augment:
            processed = [self._apply_augmentations(t) for t in processed]

        encoding = self.tokenizer(
            processed,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
        )
        return dict(encoding)

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to text (useful for debugging).

        Args:
            token_ids: 1-D tensor of token IDs.

        Returns:
            Decoded text string.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @property
    def vocab_size(self) -> int:
        """Return the tokenizer's vocabulary size."""
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int:
        """Return the padding token ID."""
        return self.tokenizer.pad_token_id

    # ------------------------------------------------------------------
    # Preprocessing pipeline
    # ------------------------------------------------------------------

    def _preprocess(self, text: str) -> str:
        """Apply all domain-specific cleaning steps.

        Pipeline order matters — abbreviation expansion must happen before
        numeric normalisation so that expanded forms are properly handled.

        Args:
            text: Input text string.

        Returns:
            Cleaned text ready for tokenisation.
        """
        # Remove residual HTML entities and unicode artefacts
        text = self._clean_encoding_artefacts(text)

        # Expand domain abbreviations for richer semantic signal
        text = self._expand_abbreviations(text)

        # Optionally normalise specific numeric values to placeholders
        if self.normalize_numbers:
            text = self._normalise_numeric_entities(text)

        # Collapse excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @staticmethod
    def _clean_encoding_artefacts(text: str) -> str:
        """Remove HTML entities, zero-width characters, and other noise.

        Args:
            text: Input text.

        Returns:
            Text with encoding artefacts removed.
        """
        import html as html_lib

        text = html_lib.unescape(text)
        # Remove zero-width spaces and other invisible unicode
        text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", text)
        # Remove isolated control characters (but keep newlines/tabs)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return text

    def _expand_abbreviations(self, text: str) -> str:
        """Replace oil/financial abbreviations with full forms.

        This gives the tokenizer more semantic content to work with,
        especially for sub-word tokenizers that may split abbreviations
        into meaningless fragments.

        Args:
            text: Input text.

        Returns:
            Text with abbreviations expanded.
        """
        for abbrev, expansion in self._ABBREVIATIONS.items():
            # Only replace whole-word occurrences to avoid false positives
            pattern = re.compile(rf"\b{re.escape(abbrev)}\b")
            text = pattern.sub(f"{abbrev} ({expansion})", text, count=1)
        return text

    def _normalise_numeric_entities(self, text: str) -> str:
        """Replace specific numeric values with semantic placeholders.

        Raw prices ($72.45/barrel) change daily and would cause the model
        to overfit to specific values.  Replacing them with category
        tokens ([PRICE], [PERCENTAGE], etc.) helps the model learn the
        *structure* of financial language instead.

        Args:
            text: Input text.

        Returns:
            Text with numeric entities replaced by placeholders.
        """
        text = self._DOLLAR_PRICE.sub("[OIL_PRICE]", text)
        text = self._BARREL_AMOUNT.sub("[BARREL_AMOUNT]", text)
        text = self._PERCENTAGE.sub("[PERCENTAGE]", text)
        text = self._LARGE_NUMBER.sub("[LARGE_NUMBER]", text)
        return text

    # ------------------------------------------------------------------
    # Data augmentation
    # ------------------------------------------------------------------

    def _apply_augmentations(self, text: str) -> str:
        """Apply stochastic augmentations to regularise training.

        Each augmentation is applied independently with probability
        ``self.augment_prob``, so a sample might receive zero, one,
        or multiple augmentations.

        Args:
            text: Preprocessed text.

        Returns:
            Augmented text.
        """
        if random.random() < self.augment_prob:
            text = self._random_sentence_dropout(text)

        if random.random() < self.augment_prob:
            text = self._random_word_swap(text)

        if random.random() < self.augment_prob:
            text = self._random_insertion(text)

        return text

    @staticmethod
    def _random_sentence_dropout(text: str, drop_prob: float = 0.1) -> str:
        """Randomly drop sentences to simulate partial information.

        This teaches the model to make predictions even when some context
        is missing, improving robustness.

        Args:
            text: Input text.
            drop_prob: Probability of dropping each sentence.

        Returns:
            Text with some sentences randomly removed.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) <= 2:
            return text
        kept = [s for s in sentences if random.random() > drop_prob]
        return " ".join(kept) if kept else text

    @staticmethod
    def _random_word_swap(text: str, n_swaps: int = 2) -> str:
        """Randomly swap adjacent words to add positional noise.

        Args:
            text: Input text.
            n_swaps: Number of swap operations to perform.

        Returns:
            Text with some adjacent words swapped.
        """
        words = text.split()
        if len(words) < 4:
            return text
        for _ in range(n_swaps):
            idx = random.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
        return " ".join(words)

    @staticmethod
    def _random_insertion(text: str) -> str:
        """Insert a domain-relevant filler token at a random position.

        Uses neutral filler words that don't change semantics but force
        the model to be robust to minor textual variations.

        Args:
            text: Input text.

        Returns:
            Text with a filler word inserted.
        """
        fillers = [
            "reportedly", "according to analysts", "market observers note",
            "industry sources say", "data shows", "meanwhile",
        ]
        words = text.split()
        if len(words) < 5:
            return text
        insert_pos = random.randint(1, len(words) - 1)
        words.insert(insert_pos, random.choice(fillers))
        return " ".join(words)
