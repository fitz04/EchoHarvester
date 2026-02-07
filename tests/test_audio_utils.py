"""Tests for audio utility functions."""

import numpy as np
import pytest

from echoharvester.utils.audio_utils import (
    AudioInfo,
    calculate_snr,
    find_matching_subtitle,
    generate_file_hash,
    get_audio_duration,
    is_audio_file,
    is_media_file,
    is_video_file,
    load_audio,
)


class TestIsAudioFile:
    def test_audio_extensions(self):
        assert is_audio_file("test.wav") is True
        assert is_audio_file("test.mp3") is True
        assert is_audio_file("test.flac") is True
        assert is_audio_file("test.ogg") is True
        assert is_audio_file("test.m4a") is True
        assert is_audio_file("test.opus") is True

    def test_non_audio(self):
        assert is_audio_file("test.txt") is False
        assert is_audio_file("test.mp4") is False
        assert is_audio_file("test.py") is False

    def test_case_insensitive(self):
        assert is_audio_file("test.WAV") is True
        assert is_audio_file("test.Mp3") is True


class TestIsVideoFile:
    def test_video_extensions(self):
        assert is_video_file("test.mp4") is True
        assert is_video_file("test.mkv") is True
        assert is_video_file("test.avi") is True
        assert is_video_file("test.webm") is True

    def test_non_video(self):
        assert is_video_file("test.wav") is False
        assert is_video_file("test.txt") is False


class TestIsMediaFile:
    def test_audio(self):
        assert is_media_file("test.wav") is True

    def test_video(self):
        assert is_media_file("test.mp4") is True

    def test_non_media(self):
        assert is_media_file("test.txt") is False


class TestCalculateSnr:
    def test_sine_wave(self):
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        snr = calculate_snr(audio)
        assert snr > 0  # A sine wave should have positive SNR

    def test_silence(self):
        audio = np.zeros(16000, dtype=np.float32)
        snr = calculate_snr(audio)
        # All zeros → signal_rms=0, log10(0)=-inf
        assert snr <= 0.0

    def test_short_audio(self):
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        snr = calculate_snr(audio, frame_length=2048)
        assert snr == 0.0  # Too short for analysis

    def test_noisy_audio(self):
        sr = 16000
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.5, sr).astype(np.float32)
        snr = calculate_snr(noise)
        # Noise should have lower SNR than a clean sine
        assert isinstance(snr, float)


class TestLoadAudio:
    def test_load_mono(self, sample_audio_mono):
        audio, sr = load_audio(sample_audio_mono)
        assert sr == 16000
        assert len(audio.shape) == 1  # Mono
        assert len(audio) == 16000  # 1 second

    def test_load_returns_float32(self, sample_audio_mono):
        audio, _ = load_audio(sample_audio_mono)
        assert audio.dtype == np.float32


class TestGetAudioDuration:
    def test_one_second(self, sample_audio_mono):
        duration = get_audio_duration(sample_audio_mono)
        assert duration == pytest.approx(1.0, abs=0.01)


class TestGenerateFileHash:
    def test_consistent_hash(self, sample_audio_mono):
        h1 = generate_file_hash(sample_audio_mono)
        h2 = generate_file_hash(sample_audio_mono)
        assert h1 == h2

    def test_hash_length(self, sample_audio_mono):
        h = generate_file_hash(sample_audio_mono)
        assert len(h) == 16

    def test_different_files(self, sample_audio_mono, sample_audio_silent):
        h1 = generate_file_hash(sample_audio_mono)
        h2 = generate_file_hash(sample_audio_silent)
        assert h1 != h2


class TestFindMatchingSubtitle:
    def test_find_ko_vtt(self, tmp_path):
        media = tmp_path / "video.mp4"
        media.touch()
        sub = tmp_path / "video.ko.vtt"
        sub.touch()
        assert find_matching_subtitle(media) == sub

    def test_find_srt(self, tmp_path):
        media = tmp_path / "video.mp4"
        media.touch()
        sub = tmp_path / "video.ko.srt"
        sub.touch()
        assert find_matching_subtitle(media) == sub

    def test_no_subtitle(self, tmp_path):
        media = tmp_path / "video.mp4"
        media.touch()
        assert find_matching_subtitle(media) is None

    def test_priority(self, tmp_path):
        media = tmp_path / "video.mp4"
        media.touch()
        vtt = tmp_path / "video.ko.vtt"
        vtt.touch()
        srt = tmp_path / "video.ko.srt"
        srt.touch()
        # VTT should be preferred
        assert find_matching_subtitle(media) == vtt
