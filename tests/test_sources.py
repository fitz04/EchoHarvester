"""Tests for source implementations."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from echoharvester.sources.base import (
    BaseSource,
    MediaItem,
    PreparedMedia,
    SourceType,
    SubtitleType,
)
from echoharvester.sources.youtube import extract_video_id


class TestExtractVideoId:
    def test_standard_url(self):
        assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_short_url(self):
        assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_embed_url(self):
        assert extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_shorts_url(self):
        assert extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_invalid_url(self):
        assert extract_video_id("https://example.com") is None
        assert extract_video_id("not a url") is None

    def test_url_with_params(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30"
        assert extract_video_id(url) == "dQw4w9WgXcQ"


class TestSourceType:
    def test_values(self):
        assert SourceType.YOUTUBE_VIDEO.value == "youtube_video"
        assert SourceType.YOUTUBE_PLAYLIST.value == "youtube_playlist"
        assert SourceType.LOCAL_FILE.value == "local_file"
        assert SourceType.LOCAL_DIRECTORY.value == "local_directory"


class TestSubtitleType:
    def test_values(self):
        assert SubtitleType.MANUAL.value == "manual"
        assert SubtitleType.AUTO.value == "auto"
        assert SubtitleType.NONE.value == "none"


class TestMediaItem:
    def test_creation(self):
        item = MediaItem(
            id="test_id",
            source_id=1,
            title="Test Video",
            duration_sec=120.0,
            subtitle_type=SubtitleType.MANUAL,
            source_type=SourceType.YOUTUBE_VIDEO,
        )
        assert item.id == "test_id"
        assert item.title == "Test Video"
        assert item.metadata == {}

    def test_defaults(self):
        item = MediaItem(id="test", source_id=1)
        assert item.title == ""
        assert item.duration_sec is None
        assert item.subtitle_type == SubtitleType.NONE
        assert item.url is None
        assert item.original_path is None


class TestPreparedMedia:
    def test_creation(self):
        item = MediaItem(id="test", source_id=1)
        pm = PreparedMedia(
            item=item,
            audio_path=Path("/tmp/audio.wav"),
            subtitle_path=Path("/tmp/sub.vtt"),
            subtitle_type=SubtitleType.MANUAL,
        )
        assert pm.audio_path == Path("/tmp/audio.wav")
        assert pm.subtitle_type == SubtitleType.MANUAL


class TestBaseSourceFactory:
    def test_from_config_youtube_video(self, sample_config):
        config = {
            "type": "youtube_video",
            "url": "https://www.youtube.com/watch?v=test123",
            "label": "Test",
        }
        source = BaseSource.from_config(1, config)
        assert source.source_type == SourceType.YOUTUBE_VIDEO

    def test_from_config_local_file(self, sample_config):
        config = {
            "type": "local_file",
            "path": "/tmp/audio.wav",
            "label": "Local",
        }
        source = BaseSource.from_config(1, config)
        assert source.source_type == SourceType.LOCAL_FILE

    def test_from_config_local_directory(self, sample_config):
        config = {
            "type": "local_directory",
            "path": "/tmp/audio_dir",
            "label": "Dir",
        }
        source = BaseSource.from_config(1, config)
        assert source.source_type == SourceType.LOCAL_DIRECTORY

    def test_from_config_unknown_type(self, sample_config):
        config = {"type": "unknown_type", "url": "test"}
        with pytest.raises(ValueError, match="Unknown source type"):
            BaseSource.from_config(1, config)


class TestLocalFileSource:
    @pytest.mark.asyncio
    async def test_discover_nonexistent(self, sample_config):
        from echoharvester.sources.local_file import LocalFileSource

        source = LocalFileSource(1, "/nonexistent/file.wav")
        with pytest.raises(FileNotFoundError):
            await source.discover()

    @pytest.mark.asyncio
    async def test_discover_non_media(self, sample_config, tmp_path):
        from echoharvester.sources.local_file import LocalFileSource

        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not media")
        source = LocalFileSource(1, str(txt_file))
        with pytest.raises(ValueError, match="Not a supported media file"):
            await source.discover()


class TestLocalDirectorySource:
    @pytest.mark.asyncio
    async def test_discover_nonexistent(self, sample_config):
        from echoharvester.sources.local_directory import LocalDirectorySource

        source = LocalDirectorySource(1, "/nonexistent/dir")
        with pytest.raises(FileNotFoundError):
            await source.discover()

    @pytest.mark.asyncio
    async def test_discover_not_directory(self, sample_config, tmp_path):
        from echoharvester.sources.local_directory import LocalDirectorySource

        f = tmp_path / "file.txt"
        f.write_text("not a dir")
        source = LocalDirectorySource(1, str(f))
        with pytest.raises(ValueError, match="Not a directory"):
            await source.discover()

    @pytest.mark.asyncio
    async def test_discover_empty_dir(self, sample_config, tmp_path):
        from echoharvester.sources.local_directory import LocalDirectorySource

        source = LocalDirectorySource(1, str(tmp_path))
        items = await source.discover()
        assert items == []
