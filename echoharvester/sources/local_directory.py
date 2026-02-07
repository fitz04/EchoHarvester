"""Local directory source implementation."""

import fnmatch
import logging
from pathlib import Path

from echoharvester.config import get_config
from echoharvester.sources.base import (
    BaseSource,
    MediaItem,
    PreparedMedia,
    SourceType,
    SubtitleType,
)
from echoharvester.utils.audio_utils import (
    convert_to_wav,
    find_matching_subtitle,
    generate_file_hash,
    get_audio_info,
    is_media_file,
)
from echoharvester.utils.subtitle_parser import detect_subtitle_type

logger = logging.getLogger(__name__)


class LocalDirectorySource(BaseSource):
    """Local directory source for batch processing."""

    source_type = SourceType.LOCAL_DIRECTORY

    def __init__(
        self,
        source_id: int,
        path: str | Path,
        pattern: str = "*.*",
        recursive: bool = True,
        label: str = "",
    ):
        super().__init__(source_id, label)
        self.path = Path(path)
        self.pattern = pattern
        self.recursive = recursive

    async def discover(self) -> list[MediaItem]:
        """Discover all media files in directory."""
        if not self.path.exists():
            raise FileNotFoundError(f"Directory not found: {self.path}")

        if not self.path.is_dir():
            raise ValueError(f"Not a directory: {self.path}")

        items = []

        # Find matching files
        if self.recursive:
            files = list(self.path.rglob("*"))
        else:
            files = list(self.path.glob("*"))

        # Filter by pattern and media type
        media_files = []
        for f in files:
            if not f.is_file():
                continue
            if not fnmatch.fnmatch(f.name, self.pattern):
                continue
            if not is_media_file(f):
                continue
            media_files.append(f)

        logger.info(f"Found {len(media_files)} media files in {self.path}")

        for file_path in sorted(media_files):
            # Generate unique ID
            file_hash = generate_file_hash(file_path)
            item_id = f"local_{file_hash}"

            # Get audio info
            try:
                info = await get_audio_info(file_path)
                duration = info.duration_sec
            except Exception as e:
                logger.warning(f"Failed to get audio info for {file_path}: {e}")
                duration = None

            # Check for subtitle
            subtitle_file = find_matching_subtitle(file_path)
            if subtitle_file and subtitle_file.exists():
                sub_type_str = detect_subtitle_type(subtitle_file)
                subtitle_type = (
                    SubtitleType.MANUAL
                    if sub_type_str == "manual"
                    else SubtitleType.AUTO
                    if sub_type_str == "auto"
                    else SubtitleType.EXTERNAL
                )
            else:
                subtitle_type = SubtitleType.NONE

            items.append(
                MediaItem(
                    id=item_id,
                    source_id=self.source_id,
                    title=file_path.stem,
                    duration_sec=duration,
                    subtitle_type=subtitle_type,
                    source_type=SourceType.LOCAL_DIRECTORY,
                    original_path=file_path,
                    metadata={
                        "relative_path": str(file_path.relative_to(self.path)),
                        "file_size": file_path.stat().st_size,
                    },
                )
            )

        logger.info(f"Discovered {len(items)} media items from directory")
        return items

    async def prepare(self, item: MediaItem, work_dir: Path) -> PreparedMedia:
        """Prepare local file for processing."""
        config = get_config()
        output_dir = work_dir / "local" / item.id
        output_dir.mkdir(parents=True, exist_ok=True)

        source_path = item.original_path
        audio_path = output_dir / f"{item.id}.wav"

        # Convert to WAV
        if source_path.suffix.lower() == ".wav":
            try:
                info = await get_audio_info(source_path)
                if (
                    info.sample_rate == config.audio.sample_rate
                    and info.channels == config.audio.channels
                ):
                    import shutil

                    shutil.copy2(source_path, audio_path)
                else:
                    await convert_to_wav(
                        source_path,
                        audio_path,
                        sample_rate=config.audio.sample_rate,
                        channels=config.audio.channels,
                    )
            except Exception:
                await convert_to_wav(
                    source_path,
                    audio_path,
                    sample_rate=config.audio.sample_rate,
                    channels=config.audio.channels,
                )
        else:
            await convert_to_wav(
                source_path,
                audio_path,
                sample_rate=config.audio.sample_rate,
                channels=config.audio.channels,
            )

        # Find subtitle
        subtitle_path = find_matching_subtitle(source_path)
        subtitle_type = SubtitleType.NONE

        if subtitle_path and subtitle_path.exists():
            sub_type_str = detect_subtitle_type(subtitle_path)
            subtitle_type = (
                SubtitleType.MANUAL
                if sub_type_str == "manual"
                else SubtitleType.AUTO
                if sub_type_str == "auto"
                else SubtitleType.EXTERNAL
            )

        return PreparedMedia(
            item=item,
            audio_path=audio_path,
            subtitle_path=subtitle_path,
            subtitle_type=subtitle_type,
        )

    @classmethod
    def from_config(cls, source_id: int, config: dict) -> "LocalDirectorySource":
        return cls(
            source_id=source_id,
            path=config["path"],
            pattern=config.get("pattern", "*.*"),
            recursive=config.get("recursive", True),
            label=config.get("label", ""),
        )
