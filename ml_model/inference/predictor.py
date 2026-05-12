"""
Prediction Pipeline
====================

Provides :class:`OilMarketPredictor` — the high-level inference interface
that takes raw HTML article paths and produces structured predictions
across all sub-domains.

Features:
    * End-to-end pipeline: HTML → text extraction → tokenisation → model → predictions.
    * **MC-Dropout uncertainty estimation** — runs multiple stochastic forward
      passes with dropout enabled to produce prediction variance / confidence
      intervals, going beyond a single-pass point estimate.
    * **Batch processing** for efficient GPU utilisation.
    * **Attention-weight extraction** for identifying the most influential
      tokens / articles (interpretability).
    * Returns a rich :class:`PredictionResult` dataclass that feeds into
      the :class:`ReportGenerator`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from ..config import Config
from ..data.html_extractor import HTMLArticleExtractor
from ..data.preprocessor import TextPreprocessor
from ..model import OilMarketTransformer

logger = logging.getLogger(__name__)


@dataclass
class ArticlePrediction:
    """Prediction results for a single article.

    Attributes:
        filename: Source HTML filename.
        title: Extracted article title.
        source: Publication name or domain.
        subdomain_predictions: Mapping of sub-domain key to prediction details.
        confidence: Overall prediction confidence (primary task).
        uncertainty: Standard deviation from MC-Dropout (if enabled).
    """

    filename: str
    title: Optional[str]
    source: Optional[str]
    subdomain_predictions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    confidence: float = 0.0
    uncertainty: float = 0.0


@dataclass
class PredictionResult:
    """Aggregated prediction result across all input articles.

    This is the top-level output object that feeds into the
    :class:`ReportGenerator`.

    Attributes:
        primary_prediction: Overall market direction prediction (up/unchanged/down).
        primary_confidence: Confidence in the primary prediction.
        primary_probabilities: Class probability distribution for the primary task.
        subdomain_aggregations: Aggregated predictions for each sub-domain
            (majority vote, mean probabilities, confidence).
        article_predictions: Per-article detailed predictions.
        ensemble_std: Standard deviation from MC-Dropout ensemble (uncertainty).
        n_articles_processed: Number of articles that were successfully processed.
        n_articles_total: Total number of input articles (including failures).
    """

    primary_prediction: str
    primary_confidence: float
    primary_probabilities: Dict[str, float]
    subdomain_aggregations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    article_predictions: List[ArticlePrediction] = field(default_factory=list)
    ensemble_std: float = 0.0
    n_articles_processed: int = 0
    n_articles_total: int = 0


class OilMarketPredictor:
    """End-to-end prediction pipeline from raw HTML to structured output.

    Usage::

        predictor = OilMarketPredictor.from_checkpoint("outputs/checkpoints/best_model.pt")
        result = predictor.predict_from_directory(Path("../raw_articles"))
        print(result.primary_prediction)  # "up", "unchanged", or "down"

    Args:
        model: Trained :class:`OilMarketTransformer`.
        config: Model and inference configuration.
        extractor: HTML text extraction engine.
        preprocessor: Text tokenisation preprocessor.
        device: Torch device for inference.
    """

    def __init__(
        self,
        model: OilMarketTransformer,
        config: Config,
        extractor: HTMLArticleExtractor,
        preprocessor: TextPreprocessor,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialise the predictor with all pipeline components."""
        self.model = model
        self.config = config
        self.ic = config.inference
        self.extractor = extractor
        self.preprocessor = preprocessor
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        config: Optional[Config] = None,
    ) -> "OilMarketPredictor":
        """Construct a predictor from a saved checkpoint.

        Loads model weights, builds the extractor and preprocessor,
        and returns a ready-to-use predictor.

        Args:
            checkpoint_path: Path to the ``.pt`` checkpoint file.
            config: Optional config override.  If ``None``, uses
                the config saved in the checkpoint.

        Returns:
            Configured :class:`OilMarketPredictor` instance.
        """
        config = config or Config()

        # Resolve device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Build model and load weights
        model = OilMarketTransformer(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Loaded model from %s (epoch %d)", checkpoint_path, checkpoint["epoch"])

        # Build data pipeline components
        extractor = HTMLArticleExtractor(
            primary_backend=config.data.extraction_backend,
            min_chars=config.data.min_article_chars,
            include_metadata_header=True,
            parsed_articles_dir=config.data.parsed_articles_dir,
        )
        preprocessor = TextPreprocessor(
            tokenizer_name=config.data.tokenizer_name,
            max_length=config.data.max_sequence_length,
            augment=False,  # no augmentation at inference
        )

        return cls(model, config, extractor, preprocessor, device)

    # ------------------------------------------------------------------
    # High-level prediction methods
    # ------------------------------------------------------------------

    def predict_from_directory(self, articles_dir: Path) -> PredictionResult:
        """Run prediction on all HTML articles in a directory.

        Args:
            articles_dir: Directory containing ``*.html`` article files.

        Returns:
            Aggregated :class:`PredictionResult`.
        """
        logger.info("Extracting articles from %s", articles_dir)
        articles = self.extractor.extract_all(articles_dir)
        return self.predict_from_articles(articles, n_total=len(list(articles_dir.glob("*.html"))))

    def predict_from_articles(
        self,
        articles: List[Dict[str, Any]],
        n_total: Optional[int] = None,
    ) -> PredictionResult:
        """Run prediction on a list of extracted article dicts.

        Args:
            articles: List of article dicts with at least a ``"text"`` key.
            n_total: Total articles attempted (for reporting extraction rate).

        Returns:
            Aggregated :class:`PredictionResult`.
        """
        if not articles:
            logger.warning("No articles to process")
            return PredictionResult(
                primary_prediction="unchanged",
                primary_confidence=0.0,
                primary_probabilities={"up": 0.33, "unchanged": 0.34, "down": 0.33},
                n_articles_processed=0,
                n_articles_total=n_total or 0,
            )

        # Get per-article predictions (optionally with MC-Dropout)
        article_predictions = self._predict_batch(articles)

        # Aggregate across all articles
        aggregated = self._aggregate_predictions(article_predictions)

        return PredictionResult(
            primary_prediction=aggregated["market_direction"]["prediction"],
            primary_confidence=aggregated["market_direction"]["confidence"],
            primary_probabilities=aggregated["market_direction"]["probabilities"],
            subdomain_aggregations=aggregated,
            article_predictions=article_predictions,
            ensemble_std=aggregated["market_direction"].get("uncertainty", 0.0),
            n_articles_processed=len(articles),
            n_articles_total=n_total or len(articles),
        )

    # ------------------------------------------------------------------
    # Batch prediction with MC-Dropout
    # ------------------------------------------------------------------

    def _predict_batch(
        self, articles: List[Dict[str, Any]]
    ) -> List[ArticlePrediction]:
        """Tokenise and predict a batch of articles.

        If ``ensemble_passes > 1`` in config, uses MC-Dropout to
        estimate prediction uncertainty.

        Args:
            articles: List of extracted article dicts.

        Returns:
            List of :class:`ArticlePrediction` objects.
        """
        results: List[ArticlePrediction] = []

        # Process in batches
        batch_size = self.ic.batch_size
        for i in range(0, len(articles), batch_size):
            batch_articles = articles[i : i + batch_size]
            texts = [a["text"] for a in batch_articles]

            # Tokenise the batch
            encoded = self.preprocessor.encode_batch(texts)
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            token_type_ids = encoded.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            if self.ic.ensemble_passes > 1:
                # MC-Dropout: multiple forward passes with dropout enabled
                all_logits = self._mc_dropout_forward(
                    input_ids, attention_mask, token_type_ids,
                    n_passes=self.ic.ensemble_passes,
                )
            else:
                # Single deterministic forward pass
                self.model.eval()
                with torch.no_grad():
                    logits = self.model(input_ids, attention_mask, token_type_ids)
                all_logits = {k: [v] for k, v in logits.items()}

            # Convert logits to predictions for each article in the batch
            for j, article in enumerate(batch_articles):
                pred = self._logits_to_prediction(
                    article, all_logits, sample_idx=j
                )
                results.append(pred)

        return results

    def _mc_dropout_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor],
        n_passes: int,
    ) -> Dict[str, List[torch.Tensor]]:
        """Run multiple forward passes with dropout enabled for uncertainty.

        MC-Dropout treats dropout as approximate Bayesian inference:
        each pass samples a different sub-network, and the variance
        across passes estimates epistemic uncertainty.

        Args:
            input_ids: Tokenised input.
            attention_mask: Attention mask.
            token_type_ids: Token type IDs.
            n_passes: Number of stochastic forward passes.

        Returns:
            Dict mapping sub-domain keys to lists of logit tensors
            (one per pass).
        """
        # Enable dropout during inference for MC-Dropout
        self.model.train()
        collected: Dict[str, List[torch.Tensor]] = {}

        with torch.no_grad():
            for _ in range(n_passes):
                logits = self.model(input_ids, attention_mask, token_type_ids)
                for key, val in logits.items():
                    collected.setdefault(key, []).append(val.cpu())

        self.model.eval()
        return collected

    def _logits_to_prediction(
        self,
        article: Dict[str, Any],
        all_logits: Dict[str, List[torch.Tensor]],
        sample_idx: int,
    ) -> ArticlePrediction:
        """Convert model logits to a structured prediction for one article.

        When multiple MC-Dropout passes are available, computes mean
        probabilities and standard deviation for uncertainty.

        Args:
            article: Article metadata dict.
            all_logits: Multi-pass logits from MC-Dropout.
            sample_idx: Index of this article in the batch.

        Returns:
            :class:`ArticlePrediction` with filled sub-domain predictions.
        """
        subdomain_preds: Dict[str, Dict[str, Any]] = {}
        primary_confidence = 0.0
        primary_uncertainty = 0.0

        for key in self.config.all_subdomain_keys:
            if key not in all_logits:
                continue

            # Stack all passes for this sample: (n_passes, n_classes)
            logit_stack = torch.stack([
                l[sample_idx] for l in all_logits[key]
            ])
            prob_stack = F.softmax(logit_stack, dim=-1)

            # Mean probabilities across passes
            mean_probs = prob_stack.mean(dim=0)
            std_probs = prob_stack.std(dim=0) if prob_stack.size(0) > 1 else torch.zeros_like(mean_probs)

            pred_class = mean_probs.argmax().item()
            confidence = mean_probs[pred_class].item()
            uncertainty = std_probs.mean().item()

            labels = self.config.subdomains[key].labels
            pred_label = labels[pred_class] if pred_class < len(labels) else "unknown"

            subdomain_preds[key] = {
                "predicted_class": pred_class,
                "predicted_label": pred_label,
                "confidence": confidence,
                "uncertainty": uncertainty,
                "probabilities": {
                    labels[i]: mean_probs[i].item()
                    for i in range(min(len(labels), len(mean_probs)))
                },
            }

            if key == "market_direction":
                primary_confidence = confidence
                primary_uncertainty = uncertainty

        return ArticlePrediction(
            filename=article.get("filename", "unknown"),
            title=article.get("title"),
            source=article.get("source"),
            subdomain_predictions=subdomain_preds,
            confidence=primary_confidence,
            uncertainty=primary_uncertainty,
        )

    # ------------------------------------------------------------------
    # Aggregation across articles
    # ------------------------------------------------------------------

    def _aggregate_predictions(
        self, predictions: List[ArticlePrediction]
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregate per-article predictions into overall market assessment.

        Uses probability-weighted voting: each article's prediction
        probabilities are averaged (weighted by confidence) to produce
        a consensus prediction.

        Args:
            predictions: List of per-article predictions.

        Returns:
            Dict of aggregated results per sub-domain.
        """
        aggregated: Dict[str, Dict[str, Any]] = {}

        for key in self.config.all_subdomain_keys:
            labels = list(self.config.subdomains[key].labels)

            # Collect probability distributions, weighted by confidence
            weighted_probs = np.zeros(len(labels))
            total_weight = 0.0
            uncertainties: List[float] = []

            for pred in predictions:
                if key not in pred.subdomain_predictions:
                    continue

                sp = pred.subdomain_predictions[key]
                conf = sp["confidence"]
                unc = sp.get("uncertainty", 0.0)

                for i, label in enumerate(labels):
                    prob = sp["probabilities"].get(label, 0.0)
                    weighted_probs[i] += prob * conf

                total_weight += conf
                uncertainties.append(unc)

            # Normalise
            if total_weight > 0:
                weighted_probs /= total_weight
            else:
                weighted_probs = np.ones(len(labels)) / len(labels)

            pred_idx = int(np.argmax(weighted_probs))
            pred_label = labels[pred_idx]
            confidence = float(weighted_probs[pred_idx])
            avg_uncertainty = float(np.mean(uncertainties)) if uncertainties else 0.0

            aggregated[key] = {
                "prediction": pred_label,
                "predicted_class": pred_idx,
                "confidence": confidence,
                "uncertainty": avg_uncertainty,
                "probabilities": {
                    labels[i]: float(weighted_probs[i])
                    for i in range(len(labels))
                },
                "n_articles": len([
                    p for p in predictions
                    if key in p.subdomain_predictions
                ]),
            }

        return aggregated
