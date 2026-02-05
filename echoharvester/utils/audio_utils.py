"""Audio processing utilities using ffmpeg and soundfile."""

import asyncio
import hashlib
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass
class AudioInfo:
    """Audio file information."""

    duration_sec: float
    sample_rate: int
    channels: int
    format: str


async def get_audio_info(file_path: Path | str) -> AudioInfo:
    """Get audio file information using ffprobe."""
    file_path = Path(file_path)

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(file_path),
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {stderr.decode()}")

    import json

    data = json.loads(stdout.decode())

    # Find audio stream
    audio_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "audio":
            audio_stream = stream
            break

    if audio_stream is None:
        raise ValueError(f"No audio stream found in {file_path}")

    duration = float(data.get("format", {}).get("duration", 0))
    if duration == 0 and "duration" in audio_stream:
        duration = float(audio_stream["duration"])

    return AudioInfo(
        duration_sec=duration,
        sample_rate=int(audio_stream.get("sample_rate", 16000)),
        channels=int(audio_stream.get("channels", 1)),
        format=audio_stream.get("codec_name", "unknown"),
    )


async def convert_to_wav(
    input_path: Path | str,
    output_path: Path | str,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """Convert audio/video file to WAV format.

    Args:
        input_path: Input file path
        output_path: Output WAV file path
        sample_rate: Target sample rate (default: 16kHz)
        channels: Number of channels (default: 1 for mono)

    Returns:
        Path to the output file
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i",
        str(input_path),
        "-vn",  # No video
        "-acodec",
        "pcm_s16le",  # 16-bit PCM
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        str(output_path),
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {stderr.decode()}")

    return output_path


async def extract_segment(
    input_path: Path | str,
    output_path: Path | str,
    start_sec: float,
    end_sec: float,
    padding_sec: float = 0.15,
    sample_rate: int = 16000,
) -> Path:
    """Extract a segment from audio file.

    Args:
        input_path: Input audio file
        output_path: Output segment file
        start_sec: Start time in seconds
        end_sec: End time in seconds
        padding_sec: Padding before and after segment
        sample_rate: Target sample rate

    Returns:
        Path to the output file
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Apply padding
    actual_start = max(0, start_sec - padding_sec)
    actual_end = end_sec + padding_sec
    duration = actual_end - actual_start

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(actual_start),
        "-i",
        str(input_path),
        "-t",
        str(duration),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(output_path),
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg segment extraction failed: {stderr.decode()}")

    return output_path


def extract_segment_sync(
    input_path: Path | str,
    output_path: Path | str,
    start_sec: float,
    end_sec: float,
    padding_sec: float = 0.15,
    sample_rate: int = 16000,
) -> Path:
    """Synchronous version of extract_segment."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    actual_start = max(0, start_sec - padding_sec)
    actual_end = end_sec + padding_sec
    duration = actual_end - actual_start

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(actual_start),
        "-i",
        str(input_path),
        "-t",
        str(duration),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg segment extraction failed: {result.stderr.decode()}")

    return output_path


def load_audio(file_path: Path | str) -> tuple[np.ndarray, int]:
    """Load audio file and return samples and sample rate."""
    file_path = Path(file_path)
    audio, sr = sf.read(str(file_path), dtype="float32")

    # Convert stereo to mono if needed
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    return audio, sr


def calculate_snr(audio: np.ndarray, frame_length: int = 2048) -> float:
    """Calculate Signal-to-Noise Ratio (SNR) in dB.

    Uses a simple energy-based approach:
    - Signal energy: RMS of the entire audio
    - Noise energy: estimated from the quietest frames

    Args:
        audio: Audio samples (float32, normalized)
        frame_length: Frame length for analysis

    Returns:
        SNR in dB
    """
    if len(audio) < frame_length:
        return 0.0

    # Calculate RMS for each frame
    num_frames = len(audio) // frame_length
    frames = audio[: num_frames * frame_length].reshape(num_frames, frame_length)
    frame_rms = np.sqrt(np.mean(frames**2, axis=1))

    # Signal: overall RMS
    signal_rms = np.sqrt(np.mean(audio**2))

    # Noise: estimate from quietest 10% of frames
    sorted_rms = np.sort(frame_rms)
    noise_frames = max(1, num_frames // 10)
    noise_rms = np.mean(sorted_rms[:noise_frames])

    # Avoid division by zero
    if noise_rms < 1e-10:
        noise_rms = 1e-10

    # SNR in dB
    snr_db = 20 * np.log10(signal_rms / noise_rms)

    return float(snr_db)


def calculate_snr_from_file(file_path: Path | str) -> float:
    """Calculate SNR from audio file."""
    audio, _ = load_audio(file_path)
    return calculate_snr(audio)


def get_audio_duration(file_path: Path | str) -> float:
    """Get audio duration in seconds using soundfile."""
    with sf.SoundFile(str(file_path)) as f:
        return f.frames / f.samplerate


def generate_file_hash(file_path: Path | str) -> str:
    """Generate MD5 hash of file for unique identification."""
    file_path = Path(file_path)

    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()[:16]


def is_audio_file(file_path: Path | str) -> bool:
    """Check if file is a supported audio format."""
    audio_extensions = {
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a",
        ".aac",
        ".wma",
        ".opus",
    }
    return Path(file_path).suffix.lower() in audio_extensions


def is_video_file(file_path: Path | str) -> bool:
    """Check if file is a supported video format."""
    video_extensions = {
        ".mp4",
        ".mkv",
        ".avi",
        ".mov",
        ".wmv",
        ".flv",
        ".webm",
        ".m4v",
        ".mpeg",
        ".mpg",
    }
    return Path(file_path).suffix.lower() in video_extensions


def is_media_file(file_path: Path | str) -> bool:
    """Check if file is a supported media format (audio or video)."""
    return is_audio_file(file_path) or is_video_file(file_path)


def find_matching_subtitle(media_path: Path | str) -> Path | None:
    """Find matching subtitle file for a media file.

    Looks for subtitle files with same name in same directory:
    - video.ko.vtt, video.ko.srt
    - video.vtt, video.srt
    """
    media_path = Path(media_path)
    stem = media_path.stem
    parent = media_path.parent

    # Try different patterns
    patterns = [
        f"{stem}.ko.vtt",
        f"{stem}.ko.srt",
        f"{stem}.vtt",
        f"{stem}.srt",
        f"{stem}.ko.auto.vtt",
        f"{stem}.ko.auto.srt",
    ]

    for pattern in patterns:
        subtitle_path = parent / pattern
        if subtitle_path.exists():
            return subtitle_path

    return None
