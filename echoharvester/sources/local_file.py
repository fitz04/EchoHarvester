"""Local file source implementation."""

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


class LocalFileSource(BaseSource):
    """Single local media file source."""

    source_type = SourceType.LOCAL_FILE

    def __init__(
        self,
        source_id: int,
        path: str | Path,
        subtitle_path: str | Path | None = None,
        label: str = "",
    ):
        super().__init__(source_id, label)
        self.path = Path(path)
        self.subtitle_path = Path(subtitle_path) if subtitle_path else None

    async def discover(self) -> list[MediaItem]:
        """Discover the single media file."""
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")

        if not is_media_file(self.path):
            raise ValueError(f"Not a supported media file: {self.path}")

        # Generate unique ID from file hash
        file_hash = generate_file_hash(self.path)
        item_id = f"local_{file_hash}"

        # Get audio info
        try:
            info = await get_audio_info(self.path)
            duration = info.duration_sec
        except Exception as e:
            logger.warning(f"Failed to get audio info: {e}")
            duration = None

        # Check for subtitle file
        subtitle_file = self.subtitle_path or find_matching_subtitle(self.path)
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
            subtitle_file = None

        return [
            MediaItem(
                id=item_id,
                source_id=self.source_id,
                title=self.path.stem,
                duration_sec=duration,
                subtitle_type=subtitle_type,
                source_type=SourceType.LOCAL_FILE,
                original_path=self.path,
                metadata={
                    "file_size": self.path.stat().st_size,
                    "subtitle_file": str(subtitle_file) if subtitle_file else None,
                },
            )
        ]

    async def prepare(self, item: MediaItem, work_dir: Path) -> PreparedMedia:
        """Prepare local file for processing."""
        config = get_config()
        output_dir = work_dir / "local" / item.id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert to WAV if needed
        source_path = item.original_path
        audio_path = output_dir / f"{item.id}.wav"

        if source_path.suffix.lower() == ".wav":
            # Check if already correct format
            try:
                info = await get_audio_info(source_path)
                if (
                    info.sample_rate == config.audio.sample_rate
                    and info.channels == config.audio.channels
                ):
                    # Link or copy instead of converting
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

        # Get subtitle path
        subtitle_path = None
        subtitle_type = SubtitleType.NONE

        if self.subtitle_path and self.subtitle_path.exists():
            subtitle_path = self.subtitle_path
            sub_type_str = detect_subtitle_type(subtitle_path)
            subtitle_type = (
                SubtitleType.MANUAL
                if sub_type_str == "manual"
                else SubtitleType.AUTO
                if sub_type_str == "auto"
                else SubtitleType.EXTERNAL
            )
        else:
            # Try to find matching subtitle
            found_sub = find_matching_subtitle(source_path)
            if found_sub:
                subtitle_path = found_sub
                sub_type_str = detect_subtitle_type(found_sub)
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
    def from_config(cls, source_id: int, config: dict) -> "LocalFileSource":
        return cls(
            source_id=source_id,
            path=config["path"],
            subtitle_path=config.get("subtitle_path"),
            label=config.get("label", ""),
        )
