"""Data preparation: load Shar sources, merge, and split into train/val/test."""

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from echoharvester.training.config import TrainingConfig

logger = logging.getLogger(__name__)


class DataPreparer:
    """Load multiple Shar sources, merge, and create stratified splits."""

    def __init__(self, config: TrainingConfig, progress_callback=None):
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.progress_callback = progress_callback

    def _emit(self, event: dict):
        """Emit a progress event if callback is set."""
        if self.progress_callback:
            self.progress_callback(event)

    def prepare(self) -> dict:
        """Run full data preparation pipeline.

        Returns:
            Statistics dict with split sizes and duration info.
        """
        from lhotse import CutSet

        self._emit({"event": "prepare_start"})

        # 1. Load and merge all shar sources
        logger.info("Loading Shar sources...")
        self._emit({"event": "prepare_progress", "step": "loading", "detail": "Loading Shar sources..."})
        all_cuts = self._load_shar_sources()
        total = len(all_cuts)
        logger.info(f"Total cuts loaded: {total}")

        if total == 0:
            raise ValueError("No cuts found in shar sources")

        # Filter out excessively long cuts (PE max_len and memory concerns)
        max_cut_duration = 30.0  # seconds
        before = len(all_cuts)
        all_cuts = [c for c in all_cuts if c.duration <= max_cut_duration]
        filtered = before - len(all_cuts)
        if filtered > 0:
            logger.info(f"Filtered {filtered} cuts exceeding {max_cut_duration}s (kept {len(all_cuts)}/{before})")
            total = len(all_cuts)

        # 2. Save audio to disk (Shar loads audio in-memory)
        logger.info("Saving audio to disk...")
        self._emit({"event": "prepare_progress", "step": "saving_audio", "detail": f"Saving {total} audio files to disk..."})
        all_cuts = self._save_audio_to_disk(all_cuts)

        # 3. Stratified split by media_id (prevent data leakage)
        logger.info("Splitting by media_id...")
        self._emit({"event": "prepare_progress", "step": "splitting", "detail": "Stratified split by media_id..."})
        train_cuts, val_cuts, test_cuts = self._stratified_split(all_cuts)

        # 4. Save splits
        self.data_dir.mkdir(parents=True, exist_ok=True)

        train_path = self.data_dir / "train_cuts.jsonl.gz"
        val_path = self.data_dir / "val_cuts.jsonl.gz"
        test_path = self.data_dir / "test_cuts.jsonl.gz"

        train_cs = CutSet.from_cuts(train_cuts)
        val_cs = CutSet.from_cuts(val_cuts)
        test_cs = CutSet.from_cuts(test_cuts)

        train_cs.to_jsonl(train_path)
        val_cs.to_jsonl(val_path)
        test_cs.to_jsonl(test_path)

        logger.info(f"Train: {len(train_cs)} cuts -> {train_path}")
        logger.info(f"Val:   {len(val_cs)} cuts -> {val_path}")
        logger.info(f"Test:  {len(test_cs)} cuts -> {test_path}")

        # 5. Compute stats
        stats = self._compute_stats(train_cs, val_cs, test_cs)
        stats_path = self.data_dir / "stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"Stats saved to {stats_path}")

        self._emit({"event": "prepare_complete", "stats": stats})

        return stats

    def _save_audio_to_disk(self, cuts: list) -> list:
        """Save in-memory audio from Shar cuts to WAV files on disk.

        Shar-loaded cuts have audio in memory. To allow lazy JSONL loading
        during training, we write each cut's audio to disk and update
        the Recording to point to the file.
        """
        import soundfile as sf
        from lhotse import AudioSource, Recording

        audio_dir = self.data_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        updated_cuts = []
        for i, cut in enumerate(cuts):
            audio_path = audio_dir / f"{cut.id}.wav"

            if not audio_path.exists():
                # Load audio from Shar (in-memory) and write to disk
                samples = cut.load_audio()  # (channels, samples)
                sf.write(
                    str(audio_path),
                    samples[0],  # mono
                    cut.recording.sampling_rate,
                )

            # Create new Recording pointing to the file on disk
            new_recording = Recording.from_file(
                str(audio_path.resolve()), recording_id=cut.recording.id
            )

            # Update cut with disk-backed recording
            cut.recording = new_recording

            # Clean up non-serializable fields added by Shar loading
            if cut.custom and "shard_origin" in cut.custom:
                cut.custom["shard_origin"] = str(cut.custom["shard_origin"])
            if cut.custom and "dataloading_info" in cut.custom:
                del cut.custom["dataloading_info"]

            updated_cuts.append(cut)

            if (i + 1) % 500 == 0:
                logger.info(f"  Saved {i + 1}/{len(cuts)} audio files")

        logger.info(f"Saved {len(updated_cuts)} audio files to {audio_dir}")
        return updated_cuts

    def _load_shar_sources(self) -> list:
        """Load CutSets from all configured shar sources."""
        from lhotse import CutSet

        all_cuts = []
        for source_path in self.config.shar_sources:
            source = Path(source_path)
            if not source.exists():
                logger.warning(f"Shar source not found: {source}")
                continue

            # Check for shar format (files like cuts.000000.jsonl.gz)
            # Use explicit file lists to avoid .bak files being matched
            # (Lhotse's in_dir mode uses extension_contains which matches .bak)
            shar_cuts = sorted(source.glob("cuts.*.jsonl.gz"))
            shar_cuts = [p for p in shar_cuts if p.suffix == ".gz"]
            if shar_cuts:
                logger.info(f"Loading Shar from {source} ({len(shar_cuts)} shards)")
                # Build explicit fields dict to bypass Lhotse's in_dir globbing
                fields = {}
                for p in sorted(source.iterdir()):
                    if p.name.endswith(".bak"):
                        continue
                    field_name = p.name.split(".")[0]
                    if field_name not in fields:
                        fields[field_name] = []
                    fields[field_name].append(p)
                for k in fields:
                    fields[k] = sorted(fields[k])
                cs = CutSet.from_shar(fields=fields)
                cuts_list = list(cs)
                all_cuts.extend(cuts_list)
                logger.info(f"  Loaded {len(cuts_list)} cuts from {source}")
            else:
                # Try loading as regular JSONL manifest
                for manifest in source.glob("*.jsonl.gz"):
                    logger.info(f"Loading manifest from {manifest}")
                    cs = CutSet.from_jsonl(manifest)
                    cuts_list = list(cs)
                    all_cuts.extend(cuts_list)
                    logger.info(f"  Loaded {len(cuts_list)} cuts from {manifest}")

        return all_cuts

    def _stratified_split(self, cuts: list) -> tuple[list, list, list]:
        """Split cuts by media_id to prevent data leakage.

        All segments from the same media_id go to the same split.
        """
        cfg = self.config.split
        rng = random.Random(cfg.seed)

        # Group cuts by media_id
        groups = defaultdict(list)
        for cut in cuts:
            media_id = getattr(cut, "custom", {}).get("media_id", cut.id) if hasattr(cut, "custom") and cut.custom else cut.id
            groups[media_id].append(cut)

        # Shuffle media_ids
        media_ids = list(groups.keys())
        rng.shuffle(media_ids)

        # Calculate split points based on cumulative cut count
        total_cuts = len(cuts)
        train_target = int(total_cuts * cfg.train_ratio)
        val_target = int(total_cuts * (cfg.train_ratio + cfg.val_ratio))

        train_cuts, val_cuts, test_cuts = [], [], []
        running_count = 0

        for mid in media_ids:
            group = groups[mid]
            if running_count < train_target:
                train_cuts.extend(group)
            elif running_count < val_target:
                val_cuts.extend(group)
            else:
                test_cuts.extend(group)
            running_count += len(group)

        # Ensure val and test are not empty
        if not val_cuts and train_cuts:
            # Move last group from train to val
            val_cuts = train_cuts[-1:]
            train_cuts = train_cuts[:-1]
        if not test_cuts and train_cuts:
            test_cuts = val_cuts[-1:] if len(val_cuts) > 1 else train_cuts[-1:]
            if len(val_cuts) > 1:
                val_cuts = val_cuts[:-1]
            else:
                train_cuts = train_cuts[:-1]

        logger.info(
            f"Split: train={len(train_cuts)}, val={len(val_cuts)}, "
            f"test={len(test_cuts)} (from {len(media_ids)} media groups)"
        )

        return train_cuts, val_cuts, test_cuts

    def _compute_stats(self, train_cs, val_cs, test_cs) -> dict:
        """Compute statistics for the prepared data."""
        def split_stats(cs, name):
            durations = [c.duration for c in cs]
            total_dur = sum(durations)
            return {
                "name": name,
                "num_cuts": len(cs),
                "total_duration_sec": round(total_dur, 2),
                "total_duration_hours": round(total_dur / 3600, 4),
                "avg_duration_sec": round(total_dur / len(cs), 2) if cs else 0,
                "min_duration_sec": round(min(durations), 2) if durations else 0,
                "max_duration_sec": round(max(durations), 2) if durations else 0,
            }

        return {
            "total_cuts": len(train_cs) + len(val_cs) + len(test_cs),
            "splits": {
                "train": split_stats(train_cs, "train"),
                "val": split_stats(val_cs, "val"),
                "test": split_stats(test_cs, "test"),
            },
        }
