"""Base classes for media sources."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SourceType(str, Enum):
    """Types of media sources."""

    YOUTUBE_CHANNEL = "youtube_channel"
    YOUTUBE_PLAYLIST = "youtube_playlist"
    YOUTUBE_VIDEO = "youtube_video"
    LOCAL_FILE = "local_file"
    LOCAL_DIRECTORY = "local_directory"


class SubtitleType(str, Enum):
    """Types of subtitles."""

    MANUAL = "manual"
    AUTO = "auto"
    EXTERNAL = "external"  # User-provided subtitle file
    NONE = "none"


@dataclass
class MediaItem:
    """Represents a media item (video/audio) to be processed."""

    id: str
    source_id: int
    title: str = ""
    duration_sec: float | None = None
    subtitle_type: SubtitleType = SubtitleType.NONE
    source_type: SourceType = SourceType.LOCAL_FILE

    # URLs or paths
    url: str | None = None  # For YouTube
    original_path: Path | None = None  # For local files

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedMedia:
    """Media prepared for processing (audio extracted, subtitle loaded)."""

    item: MediaItem
    audio_path: Path
    subtitle_path: Path | None = None
    subtitle_type: SubtitleType = SubtitleType.NONE


class BaseSource(ABC):
    """Abstract base class for media sources."""

    source_type: SourceType
    label: str

    def __init__(self, source_id: int, label: str = ""):
        self.source_id = source_id
        self.label = label

    @abstractmethod
    async def discover(self) -> list[MediaItem]:
        """Discover media items from this source.

        Returns:
            List of MediaItem objects to be processed
        """
        pass

    @abstractmethod
    async def prepare(self, item: MediaItem, work_dir: Path) -> PreparedMedia:
        """Prepare a media item for processing.

        Downloads/converts audio and subtitle files.

        Args:
            item: MediaItem to prepare
            work_dir: Working directory for temporary files

        Returns:
            PreparedMedia with paths to audio and subtitle files
        """
        pass

    @classmethod
    def from_config(cls, source_id: int, config: dict) -> "BaseSource":
        """Create a source instance from configuration.

        Args:
            source_id: Database source ID
            config: Source configuration dictionary

        Returns:
            Appropriate source instance based on type
        """
        source_type = config.get("type")

        # Import here to avoid circular imports
        from echoharvester.sources.local_directory import LocalDirectorySource
        from echoharvester.sources.local_file import LocalFileSource
        from echoharvester.sources.youtube import (
            YouTubeChannelSource,
            YouTubePlaylistSource,
            YouTubeVideoSource,
        )

        source_classes = {
            SourceType.YOUTUBE_CHANNEL.value: YouTubeChannelSource,
            SourceType.YOUTUBE_PLAYLIST.value: YouTubePlaylistSource,
            SourceType.YOUTUBE_VIDEO.value: YouTubeVideoSource,
            SourceType.LOCAL_FILE.value: LocalFileSource,
            SourceType.LOCAL_DIRECTORY.value: LocalDirectorySource,
        }

        source_class = source_classes.get(source_type)
        if source_class is None:
            raise ValueError(f"Unknown source type: {source_type}")

        return source_class.from_config(source_id, config)
