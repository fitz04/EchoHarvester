"""Utility modules for EchoHarvester."""

from echoharvester.utils.audio_utils import (
    AudioInfo,
    calculate_snr,
    calculate_snr_from_file,
    convert_to_wav,
    extract_segment,
    extract_segment_sync,
    find_matching_subtitle,
    generate_file_hash,
    get_audio_duration,
    get_audio_info,
    is_audio_file,
    is_media_file,
    is_video_file,
    load_audio,
)
from echoharvester.utils.cer import (
    calculate_cer,
    get_alignment_details,
    is_acceptable_cer,
    normalize_for_cer,
)
from echoharvester.utils.subtitle_parser import (
    SubtitleSegment,
    detect_subtitle_type,
    merge_overlapping_segments,
    parse_subtitle,
    split_long_segments,
)
from echoharvester.utils.text_normalize import (
    is_valid_text,
    normalize_text,
    prepare_for_cer,
)
from echoharvester.utils.forced_alignment import (
    AlignedWord,
    ForcedAligner,
    align_subtitles_with_audio,
)
from echoharvester.utils.retry import async_retry, retry
from echoharvester.utils.vad import (
    SileroVAD,
    calculate_speech_ratio,
    calculate_speech_ratio_from_file,
    get_vad,
    is_valid_speech_segment,
)

__all__ = [
    # Audio utilities
    "AudioInfo",
    "get_audio_info",
    "convert_to_wav",
    "extract_segment",
    "extract_segment_sync",
    "load_audio",
    "calculate_snr",
    "calculate_snr_from_file",
    "get_audio_duration",
    "generate_file_hash",
    "is_audio_file",
    "is_video_file",
    "is_media_file",
    "find_matching_subtitle",
    # Subtitle parsing
    "SubtitleSegment",
    "parse_subtitle",
    "merge_overlapping_segments",
    "split_long_segments",
    "detect_subtitle_type",
    # Text normalization
    "normalize_text",
    "is_valid_text",
    "prepare_for_cer",
    # CER calculation
    "calculate_cer",
    "normalize_for_cer",
    "is_acceptable_cer",
    "get_alignment_details",
    # Forced alignment
    "AlignedWord",
    "ForcedAligner",
    "align_subtitles_with_audio",
    # VAD
    "SileroVAD",
    "get_vad",
    "calculate_speech_ratio",
    "calculate_speech_ratio_from_file",
    "is_valid_speech_segment",
    # Retry utilities
    "retry",
    "async_retry",
]
