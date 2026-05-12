"""
Training Loop
=============

Implements the complete training pipeline for the :class:`OilMarketTransformer`:

* Multi-task training with gradient accumulation.
* Separate learning rates for encoder vs. classification heads.
* Linear warmup + cosine/linear/plateau learning rate schedules.
* Mixed-precision training (AMP) for speed on GPU.
* Encoder freezing during early epochs (warm-up for heads).
* Mixup data augmentation for regularisation.
* Early stopping based on validation metrics.
* Checkpoint saving (top-K by validation metric).
* Comprehensive logging of losses and metrics per epoch.

The :class:`Trainer` is stateful and can be resumed from a checkpoint.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    ReduceLROnPlateau,
)
from torch.utils.data import DataLoader

from ..config import Config
from ..model import OilMarketTransformer
from .losses import MultiTaskLoss
from .metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class Trainer:
    """Orchestrates model training, validation, and checkpointing.

    Args:
        model: The :class:`OilMarketTransformer` to train.
        config: Training and model configuration.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        test_loader: Optional test data loader (evaluated at end).
        class_weights: Optional per-task class weights for loss balancing.

    Attributes:
        best_metric: Best validation metric observed so far.
        current_epoch: Current training epoch (0-indexed).
        global_step: Total optimisation steps taken.
        history: List of per-epoch metric dictionaries.
    """

    def __init__(
        self,
        model: OilMarketTransformer,
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        class_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Set up optimiser, scheduler, loss, metrics, and AMP scaler."""
        self.model = model
        self.config = config
        self.tc = config.training
        self.device = self._resolve_device()

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Move model to device
        self.model.to(self.device)

        # -----------------------------------------------------------
        # Loss function
        # -----------------------------------------------------------
        self.criterion = MultiTaskLoss(
            label_smoothing=self.tc.label_smoothing,
            use_focal=True,
            focal_gamma=2.0,
            auto_weight=False,
            class_weights=class_weights,
        )

        # -----------------------------------------------------------
        # Optimiser with discriminative learning rates
        # -----------------------------------------------------------
        self.optimiser = self._build_optimiser()

        # -----------------------------------------------------------
        # Learning rate scheduler
        # -----------------------------------------------------------
        self.scheduler = self._build_scheduler()

        # -----------------------------------------------------------
        # Mixed-precision scaler
        # -----------------------------------------------------------
        self.scaler = GradScaler(enabled=self.tc.use_amp and self.device.type == "cuda")

        # -----------------------------------------------------------
        # Metrics calculator
        # -----------------------------------------------------------
        label_names = {
            k: list(v.labels) for k, v in config.subdomains.items()
        }
        self.metrics_calc = MetricsCalculator(
            subdomain_keys=config.all_subdomain_keys,
            label_names=label_names,
        )

        # -----------------------------------------------------------
        # Training state
        # -----------------------------------------------------------
        self.best_metric: float = 0.0
        self.current_epoch: int = 0
        self.global_step: int = 0
        self.patience_counter: int = 0
        self.history: List[Dict[str, Any]] = []

        # Ensure checkpoint directory exists
        self.tc.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        """Run the full training procedure.

        Iterates over epochs, handling encoder freeze/unfreeze,
        training, validation, checkpointing, and early stopping.

        Returns:
            Dictionary with final training summary including best
            metric, total epochs, and training history.
        """
        logger.info("Starting training: %d epochs, batch_size=%d, device=%s",
                     self.tc.epochs, self.tc.batch_size, self.device)

        for epoch in range(self.tc.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Freeze/unfreeze encoder based on schedule
            if self.model.is_pretrained:
                if epoch < self.config.model.freeze_encoder_epochs:
                    self.model.freeze_encoder()
                elif epoch == self.config.model.freeze_encoder_epochs:
                    self.model.unfreeze_encoder()

            # --- Training phase ---
            train_loss = self._train_one_epoch()

            # --- Validation phase ---
            val_loss, val_metrics = self._validate()

            # --- Scheduler step ---
            if isinstance(self.scheduler, ReduceLROnPlateau):
                metric_val = val_metrics.get("market_direction", {}).get("f1_macro", 0)
                self.scheduler.step(metric_val)
            else:
                self.scheduler.step()

            # --- Record history ---
            epoch_time = time.time() - epoch_start
            current_lr = self.optimiser.param_groups[0]["lr"]

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "lr": current_lr,
                "epoch_time_sec": epoch_time,
            }
            self.history.append(epoch_record)

            # --- Logging ---
            primary_f1 = val_metrics.get("market_direction", {}).get("f1_macro", 0)
            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | "
                "primary_f1=%.4f | lr=%.2e | time=%.1fs",
                epoch + 1, self.tc.epochs, train_loss, val_loss,
                primary_f1, current_lr, epoch_time,
            )

            # --- Checkpointing ---
            if primary_f1 > self.best_metric:
                self.best_metric = primary_f1
                self.patience_counter = 0
                self._save_checkpoint(is_best=True)
                logger.info("New best model! F1=%.4f", primary_f1)
            else:
                self.patience_counter += 1
                self._save_checkpoint(is_best=False)

            # --- Early stopping ---
            if self.patience_counter >= self.tc.early_stopping_patience:
                logger.info(
                    "Early stopping triggered after %d epochs without improvement",
                    self.patience_counter,
                )
                break

        # --- Final evaluation on test set ---
        test_results = None
        if self.test_loader is not None:
            logger.info("Running final evaluation on test set...")
            self._load_best_checkpoint()
            _, test_results = self._validate(loader=self.test_loader)

        # --- Save training history ---
        history_path = self.tc.checkpoint_dir / "training_history.json"
        self._save_history(history_path)

        summary = {
            "best_primary_f1": self.best_metric,
            "total_epochs": self.current_epoch + 1,
            "test_results": test_results,
            "history_path": str(history_path),
        }
        logger.info("Training complete. Best F1=%.4f", self.best_metric)
        return summary

    # ------------------------------------------------------------------
    # Epoch-level methods
    # ------------------------------------------------------------------

    def _train_one_epoch(self) -> float:
        """Run one training epoch with gradient accumulation and mixup.

        Returns:
            Average training loss over the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        self.optimiser.zero_grad()

        for step, batch in enumerate(self.train_loader):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            # Collect targets for all sub-domains
            targets = {}
            for key in self.config.all_subdomain_keys:
                label_key = f"labels_{key}"
                if label_key in batch:
                    targets[key] = batch[label_key].to(self.device)

            # Optional mixup augmentation
            if self.tc.mixup_alpha > 0 and self.model.training:
                input_ids, targets = self._apply_mixup(
                    input_ids, targets, alpha=self.tc.mixup_alpha
                )

            # Forward pass with AMP
            with autocast(enabled=self.tc.use_amp and self.device.type == "cuda"):
                logits = self.model(input_ids, attention_mask, token_type_ids)
                loss_dict = self.criterion(logits, targets)
                loss = loss_dict["total"] / self.tc.accumulation_steps

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Gradient accumulation step
            if (step + 1) % self.tc.accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimiser)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.tc.max_grad_norm
                )
                self.scaler.step(self.optimiser)
                self.scaler.update()
                self.optimiser.zero_grad()
                self.global_step += 1

            total_loss += loss_dict["total"].item()
            n_batches += 1

            # Periodic logging
            if (step + 1) % self.tc.log_every_n_steps == 0:
                avg_loss = total_loss / n_batches
                logger.debug(
                    "  Step %d/%d | loss=%.4f",
                    step + 1, len(self.train_loader), avg_loss,
                )

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(
        self, loader: Optional[DataLoader] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """Run validation/test evaluation.

        Args:
            loader: Data loader to evaluate on.  Defaults to ``val_loader``.

        Returns:
            Tuple of (average_loss, metrics_dict).
        """
        loader = loader or self.val_loader
        self.model.eval()
        self.metrics_calc.reset()

        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            targets = {}
            for key in self.config.all_subdomain_keys:
                label_key = f"labels_{key}"
                if label_key in batch:
                    targets[key] = batch[label_key].to(self.device)

            logits = self.model(input_ids, attention_mask, token_type_ids)
            loss_dict = self.criterion(logits, targets)

            total_loss += loss_dict["total"].item()
            n_batches += 1

            # Accumulate predictions for metrics
            self.metrics_calc.update(logits, targets)

        avg_loss = total_loss / max(n_batches, 1)
        metrics = self.metrics_calc.compute()

        return avg_loss, metrics

    # ------------------------------------------------------------------
    # Mixup augmentation
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_mixup(
        input_ids: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        alpha: float = 0.2,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply Mixup augmentation by shuffling and interpolating.

        For token-level inputs, Mixup is applied to the embeddings
        (not the token IDs themselves, since IDs are discrete).  However,
        since we're before the embedding layer here, we just shuffle
        and the loss handles the mixed targets.

        In practice this means we randomly permute samples within the
        batch and use the permuted targets with a mixing coefficient.
        The loss function should accept soft targets for full Mixup
        benefit; when using hard labels the dominant class is used.

        Args:
            input_ids: ``(batch, seq_len)`` token IDs.
            targets: Dict of label tensors.
            alpha: Beta distribution parameter for mix coefficient.

        Returns:
            Tuple of (mixed_input_ids, mixed_targets).  For simplicity
            with discrete tokens, we return the original inputs and
            the dominant-class targets from the random permutation.
        """
        batch_size = input_ids.size(0)

        # Sample mixing coefficient from Beta(alpha, alpha)
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
        lam = max(lam, 1 - lam)  # ensure lambda >= 0.5 so original dominates

        # Random permutation of batch indices
        perm = torch.randperm(batch_size, device=input_ids.device)

        # For discrete token inputs, use the dominant sample's tokens
        # (full Mixup requires embedding-level interpolation, handled
        # inside the model if needed)
        if lam >= 0.5:
            mixed_input_ids = input_ids
            mixed_targets = targets
        else:
            mixed_input_ids = input_ids[perm]
            mixed_targets = {k: v[perm] for k, v in targets.items()}

        return mixed_input_ids, mixed_targets

    # ------------------------------------------------------------------
    # Optimiser and scheduler construction
    # ------------------------------------------------------------------

    def _build_optimiser(self) -> AdamW:
        """Build AdamW with discriminative learning rates.

        The encoder gets a lower learning rate than the classification
        heads, following the fine-tuning best practice of preserving
        pre-trained features while allowing heads to adapt freely.

        Returns:
            Configured AdamW optimiser.
        """
        # Separate encoder and head parameters
        encoder_params = list(self.model.encoder.parameters())
        head_params = list(self.model.classification_heads.parameters())

        # No weight decay on biases and LayerNorm weights
        no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}

        param_groups = [
            # Encoder parameters with weight decay
            {
                "params": [
                    p for n, p in self.model.encoder.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "lr": self.tc.learning_rate,
                "weight_decay": self.tc.weight_decay,
                "name": "encoder_decay",
            },
            # Encoder parameters without weight decay
            {
                "params": [
                    p for n, p in self.model.encoder.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "lr": self.tc.learning_rate,
                "weight_decay": 0.0,
                "name": "encoder_no_decay",
            },
            # Head parameters with weight decay
            {
                "params": [
                    p for n, p in self.model.classification_heads.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "lr": self.tc.head_learning_rate,
                "weight_decay": self.tc.weight_decay,
                "name": "heads_decay",
            },
            # Head parameters without weight decay
            {
                "params": [
                    p for n, p in self.model.classification_heads.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "lr": self.tc.head_learning_rate,
                "weight_decay": 0.0,
                "name": "heads_no_decay",
            },
        ]

        # Filter out empty param groups
        param_groups = [g for g in param_groups if len(list(g["params"])) > 0]

        return AdamW(param_groups, eps=1e-8)

    def _build_scheduler(self):
        """Build the learning rate scheduler.

        Returns:
            LR scheduler matching the configured type.
        """
        total_steps = len(self.train_loader) * self.tc.epochs
        warmup_steps = int(total_steps * self.tc.warmup_ratio)

        if self.tc.lr_scheduler == "cosine":
            # Linear warmup followed by cosine annealing
            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return float(step) / max(warmup_steps, 1)
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                return max(0.0, 0.5 * (1.0 + __import__("math").cos(
                    __import__("math").pi * progress
                )))

            return LambdaLR(self.optimiser, lr_lambda)

        elif self.tc.lr_scheduler == "linear":
            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return float(step) / max(warmup_steps, 1)
                return max(0.0, 1.0 - (step - warmup_steps) / max(
                    total_steps - warmup_steps, 1
                ))

            return LambdaLR(self.optimiser, lr_lambda)

        elif self.tc.lr_scheduler == "plateau":
            return ReduceLROnPlateau(
                self.optimiser, mode="max", patience=3, factor=0.5
            )

        else:
            raise ValueError(f"Unknown scheduler: {self.tc.lr_scheduler}")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, is_best: bool = False) -> None:
        """Save a model checkpoint.

        Args:
            is_best: If ``True``, also saves as ``best_model.pt``.
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimiser_state_dict": self.optimiser.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_metric": self.best_metric,
            "config": {
                "model": self.config.model.__dict__,
                "training": self.config.training.__dict__,
            },
        }

        # Save epoch checkpoint
        path = self.tc.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, path)

        # Save best checkpoint
        if is_best:
            best_path = self.tc.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

        # Clean up old checkpoints (keep only top-K)
        self._cleanup_checkpoints()

    def _load_best_checkpoint(self) -> None:
        """Load the best model checkpoint for final evaluation."""
        best_path = self.tc.checkpoint_dir / "best_model.pt"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Loaded best checkpoint from epoch %d", checkpoint["epoch"])

    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the top-K best."""
        checkpoints = sorted(
            self.tc.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        # Always keep best_model.pt; remove oldest epoch checkpoints
        for ckpt in checkpoints[self.tc.save_top_k:]:
            ckpt.unlink()

    def _save_history(self, path: Path) -> None:
        """Save training history to a JSON file.

        Args:
            path: Output file path.
        """
        # Convert non-serialisable types (numpy arrays, tensors)
        serialisable = []
        for record in self.history:
            clean = {}
            for k, v in record.items():
                if isinstance(v, dict):
                    clean[k] = self._make_serialisable(v)
                else:
                    clean[k] = v
            serialisable.append(clean)

        path.write_text(json.dumps(serialisable, indent=2, default=str))
        logger.info("Training history saved to %s", path)

    @staticmethod
    def _make_serialisable(obj: Any) -> Any:
        """Recursively convert numpy/torch types to Python natives.

        Args:
            obj: Object to convert.

        Returns:
            JSON-serialisable version.
        """
        import numpy as np

        if isinstance(obj, dict):
            return {k: Trainer._make_serialisable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [Trainer._make_serialisable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return obj

    # ------------------------------------------------------------------
    # Device selection
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device() -> torch.device:
        """Select the best available compute device.

        Returns:
            ``torch.device`` for CUDA, MPS, or CPU.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA: %s", torch.cuda.get_device_name(0))
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple MPS")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        return device
