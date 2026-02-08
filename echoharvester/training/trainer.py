"""Training loop with checkpoint management, validation, and logging."""

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn

from echoharvester.training.config import TrainingConfig
from echoharvester.training.dataset import create_dataloader
from echoharvester.training.model import create_model
from echoharvester.training.tokenizer import CharTokenizer
from echoharvester.training.utils import NoamScheduler, resolve_device, set_seed

logger = logging.getLogger(__name__)


class Trainer:
    """Conformer CTC trainer with checkpointing and TensorBoard support."""

    def __init__(self, config: TrainingConfig, progress_callback=None):
        self.config = config
        self.device = resolve_device(config.device)
        self.exp_dir = Path(config.exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.progress_callback = progress_callback
        self._stop_requested = False

        logger.info(f"Training device: {self.device}")

    def stop(self):
        """Request training to stop at the next epoch boundary."""
        self._stop_requested = True

    def _emit(self, event: dict):
        """Emit a progress event if callback is set."""
        if self.progress_callback:
            self.progress_callback(event)

    def _load_pretrained(self, model, path: str) -> tuple[int, int]:
        """Load pretrained weights with key-by-key shape matching.

        Returns (loaded, skipped) counts.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        source_state = checkpoint.get("model_state_dict", checkpoint)
        target_state = model.state_dict()
        loaded, skipped = 0, 0

        for key, value in source_state.items():
            if key in target_state and target_state[key].shape == value.shape:
                target_state[key] = value
                loaded += 1
            else:
                reason = (
                    "not in model"
                    if key not in target_state
                    else f"shape {value.shape} vs {target_state[key].shape}"
                )
                logger.warning(f"Pretrained skip: {key} ({reason})")
                skipped += 1

        model.load_state_dict(target_state)
        logger.info(f"Loaded {loaded}/{loaded + skipped} params from {path}")
        return loaded, skipped

    def _freeze_encoder(self, model):
        """Freeze all parameters except output_proj (decoder head)."""
        for name, param in model.named_parameters():
            if not name.startswith("output_proj"):
                param.requires_grad = False

    def _unfreeze_all(self, model):
        """Unfreeze all model parameters."""
        for param in model.parameters():
            param.requires_grad = True

    def train(self, resume_checkpoint: str | None = None,
              pretrained_checkpoint: str | None = None,
              freeze_encoder_epochs: int = 0) -> dict:
        """Run the full training loop.

        Args:
            resume_checkpoint: Path to checkpoint to resume from.
            pretrained_checkpoint: Path to pretrained checkpoint for fine-tuning.
                Ignored if resume_checkpoint is set.
            freeze_encoder_epochs: Number of epochs to freeze encoder (0 = no freeze).

        Returns:
            Final training statistics.
        """
        params = self.config.training_params
        set_seed(self.config.split.seed)

        # Load tokenizer
        tokens_path = Path(self.config.data_dir) / "lang_char" / "tokens.txt"
        if not tokens_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {tokens_path}. Run 'train prepare' first."
            )
        tokenizer = CharTokenizer(tokens_path)
        logger.info(f"Tokenizer loaded: {tokenizer.vocab_size} tokens")

        # Create model
        model = create_model(
            num_classes=tokenizer.vocab_size,
            input_dim=self.config.features.num_mel_bins,
            config=self.config.model,
        )
        model = model.to(self.device)
        num_params = model.get_num_params()
        logger.info(f"Model parameters: {num_params:,} ({num_params / 1e6:.1f}M)")

        # Load pretrained weights (only for fresh fine-tuning, not resume)
        if pretrained_checkpoint and not resume_checkpoint:
            loaded, skipped = self._load_pretrained(model, pretrained_checkpoint)
            self._emit({
                "event": "pretrained_loaded",
                "path": pretrained_checkpoint,
                "loaded": loaded,
                "skipped": skipped,
            })

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.0,  # Controlled by scheduler
            weight_decay=params.weight_decay,
        )

        # CTC loss
        ctc_loss = nn.CTCLoss(blank=tokenizer.blank_id, reduction="mean", zero_infinity=True)

        # Create data loaders (needed before scheduler for warmup_epochs calculation)
        logger.info("Creating data loaders...")
        train_dl = create_dataloader("train", self.config, tokenizer, is_training=True)
        val_dl = create_dataloader("val", self.config, tokenizer, is_training=False)

        # LR scheduler — d_model depends on model type
        if hasattr(self.config.model, "attention_dim"):
            d_model = self.config.model.attention_dim
        else:
            d_model = self.config.model.max_encoder_dim

        # Compute warm_step: warmup_epochs overrides if > 0
        warm_step = params.warm_step
        if params.warmup_epochs > 0:
            steps_per_epoch = len(train_dl)
            warm_step = max(1, int(params.warmup_epochs * steps_per_epoch))
            logger.info(
                f"warmup_epochs={params.warmup_epochs} × {steps_per_epoch} steps/epoch "
                f"→ warm_step={warm_step}"
            )

        scheduler = NoamScheduler(
            optimizer,
            d_model=d_model,
            warm_step=warm_step,
            factor=params.lr_factor,
        )

        # Resume from checkpoint
        start_epoch = 1
        best_val_loss = float("inf")
        if resume_checkpoint:
            start_epoch, best_val_loss = self._load_checkpoint(
                resume_checkpoint, model, optimizer, scheduler
            )
            start_epoch += 1

        # TensorBoard writer (optional)
        tb_writer = self._create_tb_writer()

        # Training loop
        global_step = 0
        training_stats = {
            "epochs": [],
            "best_val_loss": best_val_loss,
            "best_epoch": 0,
        }

        logger.info(f"Starting training from epoch {start_epoch} to {params.num_epochs}")

        self._emit({
            "event": "train_start",
            "total_epochs": params.num_epochs,
            "start_epoch": start_epoch,
            "model_type": self.config.model.type,
            "num_params": num_params,
        })

        for epoch in range(start_epoch, params.num_epochs + 1):
            # Check for stop request at epoch boundary
            if self._stop_requested:
                logger.info(f"Training stopped by user at epoch {epoch}")
                self._emit({"event": "train_stopped", "epoch": epoch - 1})
                break

            # Encoder freeze/unfreeze for fine-tuning
            if freeze_encoder_epochs > 0:
                if epoch == start_epoch:
                    self._freeze_encoder(model)
                    logger.info(
                        f"Encoder frozen for epochs {start_epoch}-"
                        f"{start_epoch + freeze_encoder_epochs - 1}"
                    )
                    self._emit({
                        "event": "encoder_frozen",
                        "until_epoch": start_epoch + freeze_encoder_epochs - 1,
                    })
                elif epoch == start_epoch + freeze_encoder_epochs:
                    self._unfreeze_all(model)
                    logger.info(f"Encoder unfrozen at epoch {epoch}")
                    self._emit({"event": "encoder_unfrozen", "epoch": epoch})

            epoch_start = time.time()
            self._emit({"event": "epoch_start", "epoch": epoch})

            # Train one epoch
            train_loss, global_step = self._train_epoch(
                model, train_dl, optimizer, scheduler, ctc_loss,
                epoch, global_step, tb_writer,
            )

            # Validate
            val_loss = float("inf")
            if epoch % params.valid_interval == 0:
                val_loss = self._validate(model, val_dl, ctc_loss, epoch, tb_writer)

            elapsed = time.time() - epoch_start
            lr = scheduler.get_last_lr()

            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                training_stats["best_val_loss"] = round(best_val_loss, 4)
                training_stats["best_epoch"] = epoch

            epoch_stats = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4) if val_loss != float("inf") else None,
                "lr": lr,
                "elapsed_sec": round(elapsed, 1),
            }
            training_stats["epochs"].append(epoch_stats)

            logger.info(
                f"Epoch {epoch}/{params.num_epochs} - "
                f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, "
                f"lr: {lr:.2e}, time: {elapsed:.1f}s"
            )

            self._emit({
                "event": "epoch_complete",
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4) if val_loss != float("inf") else None,
                "lr": lr,
                "elapsed_sec": round(elapsed, 1),
                "is_best": is_best,
            })

            self._save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_loss, val_loss, is_best,
            )

            # Cleanup old checkpoints
            self._cleanup_checkpoints(params.keep_last_n)

        # Close TensorBoard
        if tb_writer:
            tb_writer.close()

        # Save final stats
        stats_path = self.exp_dir / "training_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(training_stats, f, indent=2)
        logger.info(f"Training complete. Stats saved to {stats_path}")

        if not self._stop_requested:
            self._emit({
                "event": "train_complete",
                "best_val_loss": training_stats["best_val_loss"],
                "best_epoch": training_stats["best_epoch"],
            })

        return training_stats

    def _train_epoch(
        self, model, train_dl, optimizer, scheduler, ctc_loss,
        epoch, global_step, tb_writer,
    ) -> tuple[float, int]:
        """Train one epoch."""
        model.train()
        total_loss = 0.0
        num_batches = 0
        params = self.config.training_params

        for batch in train_dl:
            features = batch["features"].to(self.device)
            feature_lens = batch["feature_lens"].to(self.device)
            targets = batch["targets"].to(self.device)
            target_lens = batch["target_lens"].to(self.device)

            # Forward
            log_probs, out_lens = model(features, feature_lens)

            # CTC loss expects (time, batch, classes)
            log_probs_t = log_probs.permute(1, 0, 2)

            loss = ctc_loss(log_probs_t, targets, out_lens, target_lens)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), params.clip_grad_norm
            )

            optimizer.step()
            scheduler.step()
            global_step += 1

            total_loss += loss.item()
            num_batches += 1

            # Log
            if global_step % params.log_interval == 0:
                avg_loss = total_loss / num_batches
                lr = scheduler.get_last_lr()
                logger.info(
                    f"  [Epoch {epoch}] step {global_step}, "
                    f"loss: {loss.item():.4f}, avg_loss: {avg_loss:.4f}, lr: {lr:.2e}"
                )
                if tb_writer:
                    tb_writer.add_scalar("train/loss", loss.item(), global_step)
                    tb_writer.add_scalar("train/lr", lr, global_step)
                self._emit({
                    "event": "batch_update",
                    "epoch": epoch,
                    "step": global_step,
                    "loss": round(loss.item(), 4),
                    "lr": lr,
                })

        avg_loss = total_loss / max(num_batches, 1)
        if tb_writer:
            tb_writer.add_scalar("train/epoch_loss", avg_loss, epoch)

        return avg_loss, global_step

    @torch.no_grad()
    def _validate(self, model, val_dl, ctc_loss, epoch, tb_writer) -> float:
        """Run validation."""
        model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in val_dl:
            features = batch["features"].to(self.device)
            feature_lens = batch["feature_lens"].to(self.device)
            targets = batch["targets"].to(self.device)
            target_lens = batch["target_lens"].to(self.device)

            log_probs, out_lens = model(features, feature_lens)
            log_probs_t = log_probs.permute(1, 0, 2)
            loss = ctc_loss(log_probs_t, targets, out_lens, target_lens)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        if tb_writer:
            tb_writer.add_scalar("val/loss", avg_loss, epoch)

        return avg_loss

    def _save_checkpoint(
        self, model, optimizer, scheduler, epoch,
        train_loss, val_loss, is_best,
    ) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": self.config.model_dump(mode="json"),
        }

        path = self.exp_dir / f"epoch-{epoch}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

        if is_best:
            best_path = self.exp_dir / "best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model updated: {best_path} (val_loss: {val_loss:.4f})")

    def _load_checkpoint(
        self, path: str, model, optimizer, scheduler
    ) -> tuple[int, float]:
        """Load checkpoint and return (epoch, best_val_loss)."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        epoch = checkpoint.get("epoch", 0)
        val_loss = checkpoint.get("val_loss", float("inf"))
        logger.info(f"Resumed from {path} (epoch {epoch}, val_loss: {val_loss:.4f})")
        return epoch, val_loss

    def _cleanup_checkpoints(self, keep_last_n: int) -> None:
        """Remove old checkpoints, keeping the last N and best."""
        checkpoints = sorted(
            self.exp_dir.glob("epoch-*.pt"),
            key=lambda p: int(p.stem.split("-")[1]),
        )

        if len(checkpoints) <= keep_last_n:
            return

        for old_ckpt in checkpoints[:-keep_last_n]:
            old_ckpt.unlink()
            logger.debug(f"Removed old checkpoint: {old_ckpt}")

    def _create_tb_writer(self):
        """Create TensorBoard writer if available."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = self.exp_dir / "tensorboard"
            writer = SummaryWriter(log_dir=str(log_dir))
            logger.info(f"TensorBoard logging to {log_dir}")
            return writer
        except ImportError:
            logger.info("TensorBoard not available, skipping TensorBoard logging")
            return None
