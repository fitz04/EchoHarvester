"""Voice Activity Detection using Silero VAD."""

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SpeechSegment(NamedTuple):
    """A detected speech segment."""

    start_sec: float
    end_sec: float


class SileroVAD:
    """Silero VAD wrapper for speech detection."""

    def __init__(self, threshold: float = 0.5):
        """Initialize Silero VAD.

        Args:
            threshold: Speech probability threshold (0.0 to 1.0)
        """
        self.threshold = threshold
        self._model = None
        self._utils = None

    def _load_model(self):
        """Lazy load the VAD model."""
        if self._model is not None:
            return

        logger.info("Loading Silero VAD model...")
        self._model, self._utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        logger.info("Silero VAD model loaded")

    def get_speech_timestamps(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
    ) -> list[SpeechSegment]:
        """Detect speech segments in audio.

        Args:
            audio: Audio samples (float32, mono)
            sample_rate: Sample rate in Hz
            min_speech_duration_ms: Minimum speech segment duration
            min_silence_duration_ms: Minimum silence between segments

        Returns:
            List of speech segments with start/end times
        """
        self._load_model()

        # Convert to torch tensor
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio

        # Get speech timestamps using Silero utility
        get_speech_timestamps = self._utils[0]

        timestamps = get_speech_timestamps(
            audio_tensor,
            self._model,
            sampling_rate=sample_rate,
            threshold=self.threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
        )

        # Convert to SpeechSegment objects
        segments = []
        for ts in timestamps:
            start_sec = ts["start"] / sample_rate
            end_sec = ts["end"] / sample_rate
            segments.append(SpeechSegment(start_sec=start_sec, end_sec=end_sec))

        return segments

    def calculate_speech_ratio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> float:
        """Calculate the ratio of speech to total duration.

        Args:
            audio: Audio samples
            sample_rate: Sample rate

        Returns:
            Speech ratio (0.0 to 1.0)
        """
        total_duration = len(audio) / sample_rate
        if total_duration == 0:
            return 0.0

        segments = self.get_speech_timestamps(audio, sample_rate)
        speech_duration = sum(seg.end_sec - seg.start_sec for seg in segments)

        return min(1.0, speech_duration / total_duration)

    def is_speech(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        min_speech_ratio: float = 0.5,
    ) -> bool:
        """Check if audio contains sufficient speech.

        Args:
            audio: Audio samples
            sample_rate: Sample rate
            min_speech_ratio: Minimum required speech ratio

        Returns:
            True if speech ratio meets threshold
        """
        speech_ratio = self.calculate_speech_ratio(audio, sample_rate)
        return speech_ratio >= min_speech_ratio


# Global VAD instance
_vad: SileroVAD | None = None


def get_vad(threshold: float = 0.5) -> SileroVAD:
    """Get the global VAD instance."""
    global _vad
    if _vad is None:
        _vad = SileroVAD(threshold=threshold)
    return _vad


def calculate_speech_ratio(
    audio: np.ndarray,
    sample_rate: int = 16000,
    threshold: float = 0.5,
) -> float:
    """Calculate speech ratio using global VAD instance."""
    vad = get_vad(threshold)
    return vad.calculate_speech_ratio(audio, sample_rate)


def calculate_speech_ratio_from_file(
    file_path: Path | str,
    threshold: float = 0.5,
) -> float:
    """Calculate speech ratio from audio file."""
    import soundfile as sf

    audio, sr = sf.read(str(file_path), dtype="float32")

    # Convert stereo to mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    return calculate_speech_ratio(audio, sr, threshold)


def is_valid_speech_segment(
    audio: np.ndarray,
    sample_rate: int = 16000,
    min_speech_ratio: float = 0.5,
    threshold: float = 0.5,
) -> bool:
    """Check if audio segment contains valid speech."""
    vad = get_vad(threshold)
    return vad.is_speech(audio, sample_rate, min_speech_ratio)
