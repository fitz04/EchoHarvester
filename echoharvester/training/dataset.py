"""Dataset and DataLoader for Lhotse CutSet with on-the-fly Fbank extraction."""

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from echoharvester.training.config import TrainingConfig
from echoharvester.training.tokenizer import CharTokenizer

logger = logging.getLogger(__name__)


class AsrDataset(torch.utils.data.Dataset):
    """ASR dataset that extracts Fbank features and encodes text on-the-fly.

    Each item returns:
        - features: (time, num_mel_bins) Fbank tensor
        - feature_lens: int, number of frames
        - targets: list[int], encoded token IDs
        - target_lens: int, length of targets
    """

    def __init__(self, tokenizer: CharTokenizer, num_mel_bins: int = 80):
        super().__init__()
        self.tokenizer = tokenizer
        self.num_mel_bins = num_mel_bins
        self._extractor = None

    @property
    def extractor(self):
        """Lazy initialization of Fbank extractor."""
        if self._extractor is None:
            from lhotse import Fbank, FbankConfig

            self._extractor = Fbank(
                FbankConfig(num_mel_bins=self.num_mel_bins)
            )
        return self._extractor

    def __getitem__(self, cuts) -> dict:
        """Process a batch of cuts from the sampler.

        Args:
            cuts: Lhotse CutSet (a batch from DynamicBucketingSampler)

        Returns:
            Dict with features, feature_lens, targets, target_lens tensors.
        """
        features_list = []
        feature_lens = []
        targets_list = []
        target_lens = []

        for cut in cuts:
            # Extract Fbank features on-the-fly
            feats = cut.compute_features(extractor=self.extractor)
            features_list.append(torch.from_numpy(feats))
            feature_lens.append(feats.shape[0])

            # Encode text
            text = cut.supervisions[0].text if cut.supervisions else ""
            token_ids = self.tokenizer.encode(text)
            targets_list.append(torch.tensor(token_ids, dtype=torch.long))
            target_lens.append(len(token_ids))

        # Pad features to same length
        max_feat_len = max(feature_lens)
        num_mel = self.num_mel_bins
        features = torch.zeros(len(features_list), max_feat_len, num_mel)
        for i, feat in enumerate(features_list):
            features[i, : feat.size(0)] = feat

        # Pad targets to same length
        max_target_len = max(target_lens) if target_lens else 0
        targets = torch.zeros(len(targets_list), max_target_len, dtype=torch.long)
        for i, tgt in enumerate(targets_list):
            targets[i, : tgt.size(0)] = tgt

        return {
            "features": features,
            "feature_lens": torch.tensor(feature_lens, dtype=torch.long),
            "targets": targets,
            "target_lens": torch.tensor(target_lens, dtype=torch.long),
        }


def create_dataloader(
    split: str,
    config: TrainingConfig,
    tokenizer: CharTokenizer,
    is_training: bool = True,
) -> DataLoader:
    """Create a DataLoader for the given split.

    Uses Lhotse's DynamicBucketingSampler for efficient batching by duration.

    Args:
        split: 'train', 'val', or 'test'
        config: Training configuration
        tokenizer: Character tokenizer
        is_training: Whether this is for training (enables shuffling)

    Returns:
        DataLoader yielding batched dicts.
    """
    from lhotse import CutSet
    from lhotse.dataset import SimpleCutSampler

    data_dir = Path(config.data_dir)
    cuts_path = data_dir / f"{split}_cuts.jsonl.gz"

    if not cuts_path.exists():
        raise FileNotFoundError(f"Split not found: {cuts_path}")

    cuts = CutSet.from_jsonl_lazy(cuts_path)

    # Filter out excessively long cuts that exceed PE max_len (10000 frames)
    # 30s * 100 frames/s / 4x subsampling = 750 frames (well within limit)
    max_cut_duration = 30.0
    cuts = cuts.filter(lambda c: c.duration <= max_cut_duration)

    # SimpleCutSampler is more robust for small/medium datasets
    sampler = SimpleCutSampler(
        cuts,
        max_duration=config.training_params.max_duration,
        shuffle=is_training,
    )

    dataset = AsrDataset(
        tokenizer=tokenizer,
        num_mel_bins=config.features.num_mel_bins,
    )

    dl = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=None,  # Sampler handles batching
        num_workers=0,  # On-the-fly feature extraction; keep in main process
    )

    return dl
