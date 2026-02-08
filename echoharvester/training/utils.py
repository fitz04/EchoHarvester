"""Training utilities: device resolution, seeding, learning rate scheduler."""

import logging
import math
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def resolve_device(device: str) -> torch.device:
    """Resolve 'auto' device to the best available option."""
    if device != "auto":
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


class NoamScheduler:
    """Noam learning rate scheduler (Attention Is All You Need).

    lr = factor * (d_model^-0.5) * min(step^-0.5, step * warmup^-1.5)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warm_step: int,
        factor: float = 1.0,
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warm_step = warm_step
        self.factor = factor
        self._step = 0

    def step(self):
        """Update learning rate."""
        self._step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _get_lr(self) -> float:
        step = max(self._step, 1)
        return self.factor * (
            self.d_model ** (-0.5)
            * min(step ** (-0.5), step * self.warm_step ** (-1.5))
        )

    def get_last_lr(self) -> float:
        return self._get_lr()

    def state_dict(self) -> dict:
        return {"step": self._step}

    def load_state_dict(self, state_dict: dict):
        self._step = state_dict["step"]


def compute_cer(ref: str, hyp: str) -> float:
    """Compute Character Error Rate between reference and hypothesis.

    Uses edit distance at character level (ignoring spaces).
    """
    ref_chars = list(ref.replace(" ", ""))
    hyp_chars = list(hyp.replace(" ", ""))

    if len(ref_chars) == 0:
        return 1.0 if len(hyp_chars) > 0 else 0.0

    # Edit distance (dynamic programming)
    d = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]

    for i in range(len(ref_chars) + 1):
        d[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d[0][j] = j

    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,      # deletion
                    d[i][j - 1] + 1,      # insertion
                    d[i - 1][j - 1] + 1,  # substitution
                )

    return d[len(ref_chars)][len(hyp_chars)] / len(ref_chars)
