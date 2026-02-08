"""CTC greedy decoding and CER evaluation."""

import json
import logging
from pathlib import Path

import torch

from echoharvester.training.config import TrainingConfig
from echoharvester.training.dataset import create_dataloader
from echoharvester.training.model import create_model
from echoharvester.training.tokenizer import CharTokenizer
from echoharvester.training.utils import compute_cer, resolve_device

logger = logging.getLogger(__name__)


def ctc_greedy_decode(
    log_probs: torch.Tensor,
    out_lens: torch.Tensor,
    tokenizer: CharTokenizer,
) -> list[str]:
    """CTC greedy decoding: argmax -> remove blanks -> remove consecutive duplicates.

    Args:
        log_probs: (batch, time, vocab_size)
        out_lens: (batch,) output lengths
        tokenizer: CharTokenizer for ID-to-token mapping

    Returns:
        List of decoded strings.
    """
    # Argmax
    predictions = log_probs.argmax(dim=-1)  # (batch, time)
    batch_size = predictions.size(0)
    results = []

    for b in range(batch_size):
        length = out_lens[b].item()
        pred_ids = predictions[b, :length].tolist()

        # Remove blanks and consecutive duplicates
        decoded_ids = []
        prev_id = -1
        for idx in pred_ids:
            if idx != tokenizer.blank_id and idx != prev_id:
                decoded_ids.append(idx)
            prev_id = idx

        text = tokenizer.decode(decoded_ids)
        results.append(text)

    return results


class Decoder:
    """Decode test set and compute CER."""

    def __init__(self, config: TrainingConfig, progress_callback=None):
        self.config = config
        self.device = resolve_device(config.device)
        self.progress_callback = progress_callback

    def _emit(self, event: dict):
        """Emit a progress event if callback is set."""
        if self.progress_callback:
            self.progress_callback(event)

    def decode(self, checkpoint_path: str, split: str = "test", max_samples: int = 0) -> dict:
        """Decode a split and compute CER.

        Args:
            checkpoint_path: Path to model checkpoint.
            split: Data split to decode ('test', 'val').
            max_samples: Max samples to save in results (0 = all).

        Returns:
            Results dict with CER and decoded samples.
        """
        # Load tokenizer
        tokens_path = Path(self.config.data_dir) / "lang_char" / "tokens.txt"
        tokenizer = CharTokenizer(tokens_path)
        logger.info(f"Tokenizer loaded: {tokenizer.vocab_size} tokens")

        # Load model
        model = self._load_model(checkpoint_path, tokenizer)
        model.eval()

        # Create dataloader
        dl = create_dataloader(split, self.config, tokenizer, is_training=False)

        # Decode
        all_refs = []
        all_hyps = []
        samples = []

        logger.info(f"Decoding {split} set...")
        self._emit({"event": "decode_start", "split": split})
        sample_limit = max_samples if max_samples > 0 else float("inf")

        with torch.no_grad():
            for batch in dl:
                features = batch["features"].to(self.device)
                feature_lens = batch["feature_lens"].to(self.device)
                targets = batch["targets"]
                target_lens = batch["target_lens"]

                log_probs, out_lens = model(features, feature_lens)
                hypotheses = ctc_greedy_decode(log_probs, out_lens, tokenizer)

                # Decode references
                for i in range(targets.size(0)):
                    ref_ids = targets[i, : target_lens[i]].tolist()
                    ref_text = tokenizer.decode(ref_ids)
                    hyp_text = hypotheses[i]

                    all_refs.append(ref_text)
                    all_hyps.append(hyp_text)

                    if len(samples) < sample_limit:
                        samples.append({
                            "ref": ref_text,
                            "hyp": hyp_text,
                            "cer": round(compute_cer(ref_text, hyp_text), 4),
                        })

                self._emit({
                    "event": "decode_progress",
                    "processed": len(all_refs),
                })

        # Compute overall CER
        total_ref_len = sum(len(r.replace(" ", "")) for r in all_refs)
        total_errors = sum(
            compute_cer(r, h) * len(r.replace(" ", ""))
            for r, h in zip(all_refs, all_hyps)
        )
        overall_cer = total_errors / max(total_ref_len, 1)

        results = {
            "split": split,
            "checkpoint": checkpoint_path,
            "num_utterances": len(all_refs),
            "overall_cer": round(overall_cer, 4),
            "samples": samples,
        }

        # Save results
        exp_dir = Path(self.config.exp_dir)
        results_path = exp_dir / f"decode_{split}.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Decode results: CER={overall_cer:.4f} "
            f"({len(all_refs)} utterances) -> {results_path}"
        )

        self._emit({
            "event": "decode_complete",
            "overall_cer": round(overall_cer, 4),
            "num_utterances": len(all_refs),
        })

        # Print samples
        logger.info("Sample predictions:")
        for s in samples[:5]:
            logger.info(f"  REF: {s['ref']}")
            logger.info(f"  HYP: {s['hyp']}")
            logger.info(f"  CER: {s['cer']}")
            logger.info("")

        return results

    def _load_model(
        self, checkpoint_path: str, tokenizer: CharTokenizer
    ) -> torch.nn.Module:
        """Load model from checkpoint."""
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        model = create_model(
            num_classes=tokenizer.vocab_size,
            input_dim=self.config.features.num_mel_bins,
            config=self.config.model,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)

        epoch = checkpoint.get("epoch", "?")
        logger.info(f"Model loaded from {checkpoint_path} (epoch {epoch})")

        return model
