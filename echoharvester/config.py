"""Configuration management using Pydantic."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class SourceConfig(BaseModel):
    """Input source configuration."""

    type: Literal[
        "youtube_channel",
        "youtube_playlist",
        "youtube_video",
        "local_file",
        "local_directory",
    ]
    url: str | None = None
    path: str | None = None
    subtitle_path: str | None = None
    pattern: str = "*.*"
    recursive: bool = True
    label: str = ""

    @field_validator("url", "path", mode="before")
    @classmethod
    def validate_source(cls, v, info):
        return v


class SubtitleConfig(BaseModel):
    """Subtitle processing configuration."""

    languages: list[str] = ["ko"]
    include_auto_generated: bool = True
    prefer_manual: bool = True


class FilterConfig(BaseModel):
    """Filtering thresholds configuration."""

    min_duration_sec: float = 0.5
    max_duration_sec: float = 30.0
    min_snr_db: float = 10.0
    min_speech_ratio: float = 0.5
    cer_threshold_manual: float = 0.15
    cer_threshold_auto: float = 0.10


class DownloadConfig(BaseModel):
    """YouTube download configuration."""

    max_concurrent: int = 3
    rate_limit: str = "1M"
    retry_count: int = 3
    retry_delay_sec: int = 5


class AudioConfig(BaseModel):
    """Audio processing configuration."""

    sample_rate: int = 16000
    format: str = "wav"
    channels: int = 1
    segment_padding_sec: float = 0.15


class ValidationConfig(BaseModel):
    """GPU validation configuration."""

    model: str = "seastar105/whisper-medium-komixv2"
    device: Literal["cuda", "cpu"] = "cuda"
    compute_type: Literal["float16", "int8", "float32"] = "float16"
    batch_size: int = 16
    beam_size: int = 5
    language: str = "ko"


class PathConfig(BaseModel):
    """Path configuration."""

    work_dir: Path = Path("./work")
    output_dir: Path = Path("./output")
    archive_dir: Path | None = Path("./archive")
    db_path: Path = Path("./echoharvester.db")

    def ensure_dirs(self):
        """Create directories if they don't exist."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.archive_dir:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


class PipelineConfig(BaseModel):
    """Pipeline execution configuration."""

    num_cpu_workers: int = 4
    gpu_queue_size: int = 100
    checkpoint_interval: int = 50
    auto_resume: bool = True


class WebConfig(BaseModel):
    """Web server configuration."""

    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False
    reload: bool = False


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Path | None = None


class Config(BaseSettings):
    """Main configuration class."""

    sources: list[SourceConfig] = Field(default_factory=list)
    subtitles: SubtitleConfig = Field(default_factory=SubtitleConfig)
    filters: FilterConfig = Field(default_factory=FilterConfig)
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.model_dump(mode="json"),
                f,
                default_flow_style=False,
                allow_unicode=True,
            )

    def setup(self) -> None:
        """Initialize directories and logging."""
        self.paths.ensure_dirs()
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging based on settings."""
        import logging

        logger = logging.getLogger("echoharvester")
        logger.setLevel(getattr(logging, self.logging.level))

        formatter = logging.Formatter(self.logging.format)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (if specified)
        if self.logging.file:
            self.logging.file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.logging.file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)


# Global config instance
_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        raise RuntimeError("Configuration not initialized. Call load_config() first.")
    return _config


def load_config(path: str | Path | None = None) -> Config:
    """Load and initialize configuration."""
    global _config

    if path:
        _config = Config.from_yaml(path)
    else:
        _config = Config()

    _config.setup()
    return _config
