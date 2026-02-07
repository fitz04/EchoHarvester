"""Shared test fixtures for EchoHarvester tests."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import pytest_asyncio

import echoharvester.config as config_module
from echoharvester.config import Config
from echoharvester.db import Database


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory for test files."""
    return tmp_path


@pytest.fixture
def sample_config(tmp_path):
    """Create a minimal Config for testing."""
    # Reset global config
    config_module._config = None

    cfg = Config(
        sources=[],
        paths={
            "work_dir": str(tmp_path / "work"),
            "output_dir": str(tmp_path / "output"),
            "archive_dir": str(tmp_path / "archive"),
            "db_path": str(tmp_path / "test.db"),
        },
        logging={"level": "WARNING"},
    )
    config_module._config = cfg
    cfg.paths.ensure_dirs()
    yield cfg
    config_module._config = None


@pytest_asyncio.fixture
async def db(sample_config):
    """Create a test database."""
    database = Database(db_path=sample_config.paths.db_path)
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
def sample_audio_mono(tmp_path):
    """Create a sample mono WAV file (1 second of sine wave at 440Hz)."""
    import soundfile as sf

    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    path = tmp_path / "sample.wav"
    sf.write(str(path), audio, sr)
    return path


@pytest.fixture
def sample_audio_silent(tmp_path):
    """Create a silent WAV file."""
    import soundfile as sf

    sr = 16000
    audio = np.zeros(sr, dtype=np.float32)  # 1 second of silence

    path = tmp_path / "silent.wav"
    sf.write(str(path), audio, sr)
    return path


@pytest.fixture
def sample_vtt_file(tmp_path):
    """Create a sample VTT subtitle file."""
    content = """WEBVTT

00:00:01.000 --> 00:00:04.000
안녕하세요 여러분

00:00:05.000 --> 00:00:08.000
오늘은 좋은 날씨입니다

00:00:10.000 --> 00:00:14.000
감사합니다
"""
    path = tmp_path / "sample.ko.vtt"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def sample_srt_file(tmp_path):
    """Create a sample SRT subtitle file."""
    content = """1
00:00:01,000 --> 00:00:04,000
안녕하세요 여러분

2
00:00:05,000 --> 00:00:08,000
오늘은 좋은 날씨입니다

3
00:00:10,000 --> 00:00:14,000
감사합니다
"""
    path = tmp_path / "sample.srt"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def sample_config_yaml(tmp_path):
    """Create a sample config YAML file."""
    content = """sources:
  - type: youtube_video
    url: "https://www.youtube.com/watch?v=test123"
    label: "Test Video"

subtitles:
  languages: ["ko"]
  include_auto_generated: true
  prefer_manual: true

filters:
  min_duration_sec: 0.5
  max_duration_sec: 30.0
  min_snr_db: 10.0
  cer_threshold_manual: 0.15

paths:
  work_dir: "{work_dir}"
  output_dir: "{output_dir}"
  db_path: "{db_path}"

logging:
  level: WARNING
""".format(
        work_dir=str(tmp_path / "work"),
        output_dir=str(tmp_path / "output"),
        db_path=str(tmp_path / "test.db"),
    )
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path
