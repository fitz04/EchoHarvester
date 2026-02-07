"""Tests for configuration management."""

import pytest

import echoharvester.config as config_module
from echoharvester.config import (
    AudioConfig,
    Config,
    DownloadConfig,
    FilterConfig,
    ForcedAlignmentConfig,
    LoggingConfig,
    PathConfig,
    PipelineConfig,
    SourceConfig,
    SubtitleConfig,
    ValidationConfig,
    WebConfig,
    get_config,
    load_config,
)


@pytest.fixture(autouse=True)
def reset_global_config():
    """Reset the global config before each test."""
    config_module._config = None
    yield
    config_module._config = None


class TestSourceConfig:
    def test_youtube_video(self):
        cfg = SourceConfig(type="youtube_video", url="https://youtube.com/watch?v=test")
        assert cfg.type == "youtube_video"
        assert cfg.url == "https://youtube.com/watch?v=test"

    def test_local_file(self):
        cfg = SourceConfig(type="local_file", path="/tmp/audio.wav")
        assert cfg.type == "local_file"
        assert cfg.path == "/tmp/audio.wav"

    def test_invalid_type(self):
        with pytest.raises(Exception):
            SourceConfig(type="invalid_type")

    def test_default_pattern(self):
        cfg = SourceConfig(type="local_directory", path="/tmp")
        assert cfg.pattern == "*.*"
        assert cfg.recursive is True


class TestFilterConfig:
    def test_defaults(self):
        cfg = FilterConfig()
        assert cfg.min_duration_sec == 0.5
        assert cfg.max_duration_sec == 30.0
        assert cfg.min_snr_db == 10.0
        assert cfg.min_speech_ratio == 0.5
        assert cfg.cer_threshold_manual == 0.15
        assert cfg.cer_threshold_auto == 0.10

    def test_custom_values(self):
        cfg = FilterConfig(min_duration_sec=1.0, max_duration_sec=20.0)
        assert cfg.min_duration_sec == 1.0
        assert cfg.max_duration_sec == 20.0


class TestAudioConfig:
    def test_defaults(self):
        cfg = AudioConfig()
        assert cfg.sample_rate == 16000
        assert cfg.format == "wav"
        assert cfg.channels == 1
        assert cfg.segment_padding_sec == 0.15


class TestPathConfig:
    def test_defaults(self):
        cfg = PathConfig()
        assert cfg.work_dir.name == "work"
        assert cfg.output_dir.name == "output"

    def test_ensure_dirs(self, tmp_path):
        cfg = PathConfig(
            work_dir=tmp_path / "work",
            output_dir=tmp_path / "output",
            db_path=tmp_path / "test.db",
        )
        cfg.ensure_dirs()
        assert (tmp_path / "work").exists()
        assert (tmp_path / "output").exists()


class TestConfig:
    def test_default_config(self):
        cfg = Config()
        assert cfg.sources == []
        assert isinstance(cfg.filters, FilterConfig)
        assert isinstance(cfg.audio, AudioConfig)

    def test_sources_none_to_empty(self):
        """When YAML sources is null/None, it should become []."""
        cfg = Config(sources=None)
        assert cfg.sources == []

    def test_from_yaml(self, sample_config_yaml, tmp_path):
        cfg = Config.from_yaml(sample_config_yaml)
        assert len(cfg.sources) == 1
        assert cfg.sources[0].type == "youtube_video"
        assert cfg.subtitles.languages == ["ko"]

    def test_from_yaml_missing_file(self):
        with pytest.raises(FileNotFoundError):
            Config.from_yaml("/nonexistent/path.yaml")

    def test_to_yaml(self, tmp_path):
        cfg = Config()
        path = tmp_path / "output_config.yaml"
        cfg.to_yaml(path)
        assert path.exists()

        # Re-read and verify
        cfg2 = Config.from_yaml(path)
        assert cfg2.filters.min_duration_sec == cfg.filters.min_duration_sec

    def test_setup(self, tmp_path):
        cfg = Config(
            paths={
                "work_dir": str(tmp_path / "work"),
                "output_dir": str(tmp_path / "output"),
                "db_path": str(tmp_path / "test.db"),
            },
            logging={"level": "WARNING"},
        )
        cfg.setup()
        assert (tmp_path / "work").exists()
        assert (tmp_path / "output").exists()


class TestGetConfig:
    def test_not_initialized(self):
        with pytest.raises(RuntimeError, match="Configuration not initialized"):
            get_config()

    def test_after_load(self, tmp_path):
        # Create a minimal YAML
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(
            f"paths:\n  work_dir: {tmp_path / 'work'}\n"
            f"  output_dir: {tmp_path / 'output'}\n"
            f"  db_path: {tmp_path / 'test.db'}\n"
            f"logging:\n  level: WARNING\n",
            encoding="utf-8",
        )
        cfg = load_config(yaml_path)
        assert get_config() is cfg


class TestLoadConfig:
    def test_load_default(self, tmp_path):
        cfg = load_config()
        assert isinstance(cfg, Config)

    def test_load_from_yaml(self, sample_config_yaml, tmp_path):
        cfg = load_config(sample_config_yaml)
        assert len(cfg.sources) == 1


class TestValidationConfig:
    def test_defaults(self):
        cfg = ValidationConfig()
        assert cfg.backend == "qwen-asr"
        assert cfg.batch_size == 16
        assert cfg.beam_size == 5
