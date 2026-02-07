"""YouTube source implementations."""

import asyncio
import json
import logging
import re
from pathlib import Path

from echoharvester.config import get_config
from echoharvester.sources.base import (
    BaseSource,
    MediaItem,
    PreparedMedia,
    SourceType,
    SubtitleType,
)

logger = logging.getLogger(__name__)


def extract_video_id(url: str) -> str | None:
    """Extract video ID from various YouTube URL formats."""
    patterns = [
        r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
        r"(?:shorts/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


class YouTubeBaseSource(BaseSource):
    """Base class for YouTube sources."""

    def __init__(
        self,
        source_id: int,
        url: str,
        label: str = "",
        languages: list[str] | None = None,
        include_auto: bool = True,
        prefer_manual: bool = True,
    ):
        super().__init__(source_id, label)
        self.url = url
        self.languages = languages or ["ko"]
        self.include_auto = include_auto
        self.prefer_manual = prefer_manual

    async def _run_ytdlp(self, *args: str, timeout: int = 120) -> str:
        """Run yt-dlp command and return stdout."""
        import shutil
        import sys

        # Find yt-dlp: prefer venv bin, then PATH
        ytdlp_bin = shutil.which("yt-dlp", path=str(Path(sys.executable).parent))
        if not ytdlp_bin:
            ytdlp_bin = shutil.which("yt-dlp")
        if not ytdlp_bin:
            raise FileNotFoundError("yt-dlp not found. Install it with: pip install yt-dlp")

        cmd = [ytdlp_bin, *args]
        logger.debug(f"Running: {' '.join(cmd)}")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise TimeoutError(f"yt-dlp timed out after {timeout}s")

        if proc.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise RuntimeError(f"yt-dlp failed: {error_msg}")

        return stdout.decode()

    async def _get_subtitle_info(self, video_id: str) -> tuple[SubtitleType, str | None]:
        """Get subtitle information for a video.

        Returns:
            Tuple of (subtitle_type, language_code)
        """
        try:
            output = await self._run_ytdlp(
                "--list-subs",
                "--skip-download",
                f"https://www.youtube.com/watch?v={video_id}",
                timeout=30,
            )

            has_manual = False
            has_auto = False

            for lang in self.languages:
                if re.search(rf"\b{lang}\b.*\bvtt\b", output, re.IGNORECASE):
                    # Check if it's in the auto-generated section
                    lines = output.split("\n")
                    in_auto_section = False
                    for line in lines:
                        if "auto-generated" in line.lower():
                            in_auto_section = True
                        elif "available subtitles" in line.lower():
                            in_auto_section = False
                        elif lang in line.lower():
                            if in_auto_section:
                                has_auto = True
                            else:
                                has_manual = True

            if has_manual:
                return SubtitleType.MANUAL, self.languages[0]
            elif has_auto and self.include_auto:
                return SubtitleType.AUTO, self.languages[0]
            else:
                return SubtitleType.NONE, None

        except Exception as e:
            logger.warning(f"Failed to get subtitle info for {video_id}: {e}")
            return SubtitleType.NONE, None

    async def prepare(self, item: MediaItem, work_dir: Path) -> PreparedMedia:
        """Download audio and subtitles for a YouTube video."""
        config = get_config()
        video_id = item.id
        output_dir = work_dir / "youtube" / video_id
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_path = output_dir / f"{video_id}.wav"
        subtitle_path = None
        subtitle_type = SubtitleType.NONE

        # Build yt-dlp arguments
        args = [
            "-x",  # Extract audio
            "--audio-format",
            "wav",
            "--postprocessor-args",
            f"ffmpeg:-ar {config.audio.sample_rate} -ac 1",
            "-o",
            str(output_dir / "%(id)s.%(ext)s"),
        ]

        # Add subtitle options
        lang_str = ",".join(self.languages)
        if self.prefer_manual:
            args.extend(["--write-sub", "--sub-lang", lang_str, "--sub-format", "vtt"])
        if self.include_auto:
            args.extend(["--write-auto-sub"])

        # Rate limiting
        if config.download.rate_limit:
            args.extend(["--limit-rate", config.download.rate_limit])

        # Add URL
        args.append(f"https://www.youtube.com/watch?v={video_id}")

        # Download with retries
        for attempt in range(config.download.retry_count):
            try:
                await self._run_ytdlp(*args, timeout=300)
                break
            except Exception as e:
                if attempt < config.download.retry_count - 1:
                    logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(config.download.retry_delay_sec * (attempt + 1))
                else:
                    raise

        # Find downloaded subtitle file
        for lang in self.languages:
            # Try manual subtitle first
            for pattern in [f"{video_id}.{lang}.vtt", f"{video_id}.{lang}.*.vtt"]:
                for sub_file in output_dir.glob(pattern):
                    if "auto" not in sub_file.name.lower():
                        subtitle_path = sub_file
                        subtitle_type = SubtitleType.MANUAL
                        break
                if subtitle_path:
                    break

            # Try auto-generated subtitle
            if not subtitle_path and self.include_auto:
                for pattern in [f"{video_id}.{lang}*.vtt"]:
                    for sub_file in output_dir.glob(pattern):
                        subtitle_path = sub_file
                        subtitle_type = SubtitleType.AUTO
                        break
                    if subtitle_path:
                        break

            if subtitle_path:
                break

        return PreparedMedia(
            item=item,
            audio_path=audio_path,
            subtitle_path=subtitle_path,
            subtitle_type=subtitle_type,
        )


class YouTubeVideoSource(YouTubeBaseSource):
    """Single YouTube video source."""

    source_type = SourceType.YOUTUBE_VIDEO

    async def discover(self) -> list[MediaItem]:
        """Get info for single video."""
        video_id = extract_video_id(self.url)
        if not video_id:
            raise ValueError(f"Invalid YouTube URL: {self.url}")

        try:
            output = await self._run_ytdlp(
                "--dump-json",
                "--skip-download",
                self.url,
            )
            info = json.loads(output)

            subtitle_type, _ = await self._get_subtitle_info(video_id)

            return [
                MediaItem(
                    id=video_id,
                    source_id=self.source_id,
                    title=info.get("title", ""),
                    duration_sec=info.get("duration"),
                    subtitle_type=subtitle_type,
                    source_type=SourceType.YOUTUBE_VIDEO,
                    url=self.url,
                    metadata={
                        "channel": info.get("channel"),
                        "upload_date": info.get("upload_date"),
                        "view_count": info.get("view_count"),
                    },
                )
            ]
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            raise

    @classmethod
    def from_config(cls, source_id: int, config: dict) -> "YouTubeVideoSource":
        from echoharvester.config import get_config

        app_config = get_config()
        return cls(
            source_id=source_id,
            url=config["url"],
            label=config.get("label", ""),
            languages=app_config.subtitles.languages,
            include_auto=app_config.subtitles.include_auto_generated,
            prefer_manual=app_config.subtitles.prefer_manual,
        )


class YouTubePlaylistSource(YouTubeBaseSource):
    """YouTube playlist source."""

    source_type = SourceType.YOUTUBE_PLAYLIST

    async def discover(self) -> list[MediaItem]:
        """Get all videos from playlist."""
        try:
            output = await self._run_ytdlp(
                "--flat-playlist",
                "--dump-json",
                self.url,
                timeout=300,
            )

            items = []
            for line in output.strip().split("\n"):
                if not line:
                    continue
                info = json.loads(line)
                video_id = info.get("id")
                if not video_id:
                    continue

                # Get subtitle info for each video
                subtitle_type, _ = await self._get_subtitle_info(video_id)

                items.append(
                    MediaItem(
                        id=video_id,
                        source_id=self.source_id,
                        title=info.get("title", ""),
                        duration_sec=info.get("duration"),
                        subtitle_type=subtitle_type,
                        source_type=SourceType.YOUTUBE_PLAYLIST,
                        url=f"https://www.youtube.com/watch?v={video_id}",
                        metadata={
                            "playlist_title": info.get("playlist_title"),
                            "playlist_index": info.get("playlist_index"),
                        },
                    )
                )

            logger.info(f"Discovered {len(items)} videos from playlist")
            return items

        except Exception as e:
            logger.error(f"Failed to get playlist info: {e}")
            raise

    @classmethod
    def from_config(cls, source_id: int, config: dict) -> "YouTubePlaylistSource":
        from echoharvester.config import get_config

        app_config = get_config()
        return cls(
            source_id=source_id,
            url=config["url"],
            label=config.get("label", ""),
            languages=app_config.subtitles.languages,
            include_auto=app_config.subtitles.include_auto_generated,
            prefer_manual=app_config.subtitles.prefer_manual,
        )


class YouTubeChannelSource(YouTubeBaseSource):
    """YouTube channel source."""

    source_type = SourceType.YOUTUBE_CHANNEL

    async def discover(self) -> list[MediaItem]:
        """Get all videos from channel."""
        try:
            # Get channel videos URL
            channel_url = self.url
            if not channel_url.endswith("/videos"):
                channel_url = channel_url.rstrip("/") + "/videos"

            output = await self._run_ytdlp(
                "--flat-playlist",
                "--dump-json",
                channel_url,
                timeout=600,
            )

            items = []
            for line in output.strip().split("\n"):
                if not line:
                    continue
                info = json.loads(line)
                video_id = info.get("id")
                if not video_id:
                    continue

                subtitle_type, _ = await self._get_subtitle_info(video_id)

                items.append(
                    MediaItem(
                        id=video_id,
                        source_id=self.source_id,
                        title=info.get("title", ""),
                        duration_sec=info.get("duration"),
                        subtitle_type=subtitle_type,
                        source_type=SourceType.YOUTUBE_CHANNEL,
                        url=f"https://www.youtube.com/watch?v={video_id}",
                        metadata={
                            "channel": info.get("channel"),
                            "upload_date": info.get("upload_date"),
                        },
                    )
                )

            logger.info(f"Discovered {len(items)} videos from channel")
            return items

        except Exception as e:
            logger.error(f"Failed to get channel info: {e}")
            raise

    @classmethod
    def from_config(cls, source_id: int, config: dict) -> "YouTubeChannelSource":
        from echoharvester.config import get_config

        app_config = get_config()
        return cls(
            source_id=source_id,
            url=config["url"],
            label=config.get("label", ""),
            languages=app_config.subtitles.languages,
            include_auto=app_config.subtitles.include_auto_generated,
            prefer_manual=app_config.subtitles.prefer_manual,
        )
