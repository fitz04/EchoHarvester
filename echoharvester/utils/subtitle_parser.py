"""Subtitle parsing utilities for VTT and SRT formats."""

import re
from dataclasses import dataclass
from pathlib import Path

import webvtt


@dataclass
class SubtitleSegment:
    """A single subtitle segment."""

    start_sec: float
    end_sec: float
    text: str

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


def parse_timestamp(timestamp: str) -> float:
    """Parse timestamp string to seconds.

    Supports formats:
    - HH:MM:SS.mmm (VTT)
    - HH:MM:SS,mmm (SRT)
    - MM:SS.mmm
    """
    timestamp = timestamp.strip().replace(",", ".")

    parts = timestamp.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    elif len(parts) == 2:
        minutes, seconds = parts
        return float(minutes) * 60 + float(seconds)
    else:
        return float(timestamp)


def clean_vtt_text(text: str) -> str:
    """Remove VTT formatting tags and clean text.

    Removes:
    - Position/alignment tags: <00:00:01.000>
    - Voice tags: <v Speaker>
    - Style tags: <c>, <i>, <b>, <u>
    - Ruby annotations: <ruby>, <rt>
    """
    # Remove timestamp tags like <00:00:01.000>
    text = re.sub(r"<\d{2}:\d{2}:\d{2}[.,]\d{3}>", "", text)

    # Remove voice tags <v Speaker>text</v>
    text = re.sub(r"<v[^>]*>", "", text)
    text = re.sub(r"</v>", "", text)

    # Remove style tags
    text = re.sub(r"</?[cibu](?:\.[^>]*)?>", "", text)

    # Remove ruby annotations
    text = re.sub(r"</?ruby>", "", text)
    text = re.sub(r"<rt>[^<]*</rt>", "", text)

    # Remove any remaining HTML-like tags
    text = re.sub(r"<[^>]+>", "", text)

    # Clean up whitespace
    text = " ".join(text.split())

    return text.strip()


def _is_youtube_rolling_vtt(file_path: Path | str, captions: list) -> bool:
    """Detect if VTT uses YouTube's rolling display format.

    YouTube VTT has transition cues (~10ms) alternating with content cues,
    and inline word timestamps in the raw file (<00:00:00.640><c> word</c>).
    The webvtt library strips inline timestamps, so we also check the raw file.
    """
    # Check for transition cues (very short duration)
    transition_count = 0
    for cap in captions[:20]:
        duration = parse_timestamp(cap.end) - parse_timestamp(cap.start)
        if duration < 0.05:
            transition_count += 1
    if transition_count >= 3:
        return True

    # Fallback: check raw file for inline timestamps
    try:
        raw = Path(file_path).read_text(encoding="utf-8", errors="replace")[:3000]
        return bool(re.search(r"<\d{2}:\d{2}:\d{2}[.,]\d{3}><c>", raw))
    except Exception:
        return False


def _parse_youtube_rolling_vtt(captions: list) -> list[SubtitleSegment]:
    """Parse YouTube rolling VTT, extracting only new text from each cue.

    YouTube VTT uses a 2-line rolling display:
    - Content cues (~2-4s): line 1 = repeated previous text, line 2 = new text
    - Transition cues (~10ms): bridge between displays, skipped

    Since webvtt strips inline timestamps, we identify new content by
    taking only the last non-empty line of each content cue.
    The first content cue's line 1 is also included since it contains
    the opening text that has no prior cue to repeat from.
    """
    segments = []
    is_first_content = True

    for caption in captions:
        start = parse_timestamp(caption.start)
        end = parse_timestamp(caption.end)

        # Skip transition cues (< 50ms)
        if end - start < 0.05:
            continue

        # Get non-empty lines
        lines = [l.strip() for l in caption.text.split("\n") if l.strip()]
        if not lines:
            continue

        # First content cue: also include line 1 (opening text)
        if is_first_content and len(lines) >= 2:
            opening_text = clean_vtt_text(lines[0])
            if opening_text:
                segments.append(SubtitleSegment(
                    start_sec=start,
                    end_sec=start,
                    text=opening_text,
                ))
        is_first_content = False

        # In rolling display: last line is the new content
        new_text = clean_vtt_text(lines[-1])
        if not new_text:
            continue

        segments.append(SubtitleSegment(
            start_sec=start,
            end_sec=end,
            text=new_text,
        ))

    return segments


def parse_vtt(file_path: Path | str) -> list[SubtitleSegment]:
    """Parse VTT subtitle file. Auto-detects YouTube rolling format."""
    file_path = Path(file_path)

    try:
        captions = list(webvtt.read(str(file_path)))
    except Exception as e:
        raise ValueError(f"Failed to parse VTT file {file_path}: {e}")

    if not captions:
        return []

    # Auto-detect YouTube rolling VTT and use specialized parser
    if _is_youtube_rolling_vtt(file_path, captions):
        return _parse_youtube_rolling_vtt(captions)

    # Standard VTT parsing
    segments = []
    for caption in captions:
        text = clean_vtt_text(caption.text)
        if not text:
            continue

        segment = SubtitleSegment(
            start_sec=parse_timestamp(caption.start),
            end_sec=parse_timestamp(caption.end),
            text=text,
        )
        segments.append(segment)

    return segments


def parse_srt(file_path: Path | str) -> list[SubtitleSegment]:
    """Parse SRT subtitle file."""
    file_path = Path(file_path)

    # SRT format:
    # 1
    # 00:00:01,000 --> 00:00:04,000
    # Text line 1
    # Text line 2
    #
    # 2
    # ...

    segments = []
    content = file_path.read_text(encoding="utf-8", errors="replace")

    # Split by double newline (subtitle blocks)
    blocks = re.split(r"\n\s*\n", content.strip())

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        # Skip index line, get timestamp line
        timestamp_line = lines[1] if lines[0].isdigit() else lines[0]

        # Parse timestamp: 00:00:01,000 --> 00:00:04,000
        match = re.match(
            r"(\d{1,2}:\d{2}:\d{2}[,.]?\d*)\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]?\d*)",
            timestamp_line,
        )
        if not match:
            continue

        start_sec = parse_timestamp(match.group(1))
        end_sec = parse_timestamp(match.group(2))

        # Get text (remaining lines)
        text_start = 2 if lines[0].isdigit() else 1
        text = " ".join(lines[text_start:])
        text = clean_vtt_text(text)

        if not text:
            continue

        segments.append(SubtitleSegment(start_sec=start_sec, end_sec=end_sec, text=text))

    return segments


def parse_subtitle(file_path: Path | str) -> list[SubtitleSegment]:
    """Parse subtitle file (auto-detect format)."""
    file_path = Path(file_path)

    suffix = file_path.suffix.lower()
    if suffix == ".vtt":
        return parse_vtt(file_path)
    elif suffix == ".srt":
        return parse_srt(file_path)
    else:
        # Try VTT first, then SRT
        try:
            return parse_vtt(file_path)
        except Exception:
            return parse_srt(file_path)


def merge_overlapping_segments(
    segments: list[SubtitleSegment],
    max_gap_sec: float = 0.1,
) -> list[SubtitleSegment]:
    """Merge overlapping or nearly adjacent segments.

    Args:
        segments: List of subtitle segments
        max_gap_sec: Maximum gap between segments to merge

    Returns:
        List of merged segments
    """
    if not segments:
        return []

    # Sort by start time
    sorted_segments = sorted(segments, key=lambda s: s.start_sec)
    merged = [sorted_segments[0]]

    for segment in sorted_segments[1:]:
        last = merged[-1]

        # Check if segments overlap or are close enough to merge
        if segment.start_sec <= last.end_sec + max_gap_sec:
            # Merge: extend end time and combine text
            merged[-1] = SubtitleSegment(
                start_sec=last.start_sec,
                end_sec=max(last.end_sec, segment.end_sec),
                text=f"{last.text} {segment.text}".strip(),
            )
        else:
            merged.append(segment)

    return merged


def split_long_segments(
    segments: list[SubtitleSegment],
    max_duration_sec: float = 30.0,
) -> list[SubtitleSegment]:
    """Split segments that exceed maximum duration.

    Long segments are split at sentence boundaries if possible.
    """
    result = []

    for segment in segments:
        if segment.duration_sec <= max_duration_sec:
            result.append(segment)
            continue

        # Try to split at sentence boundaries
        sentences = re.split(r"([.!?。！？])\s*", segment.text)

        # Rebuild sentences with punctuation
        text_parts = []
        current = ""
        for i, part in enumerate(sentences):
            if i % 2 == 0:  # Text part
                current += part
            else:  # Punctuation
                current += part
                text_parts.append(current.strip())
                current = ""
        if current.strip():
            text_parts.append(current.strip())

        if len(text_parts) <= 1:
            # Can't split by sentences, just keep as is
            result.append(segment)
            continue

        # Distribute time proportionally to text length
        total_chars = sum(len(p) for p in text_parts)
        current_time = segment.start_sec

        for text in text_parts:
            if not text:
                continue
            portion = len(text) / total_chars
            duration = segment.duration_sec * portion
            result.append(
                SubtitleSegment(
                    start_sec=current_time,
                    end_sec=current_time + duration,
                    text=text,
                )
            )
            current_time += duration

    return result


def detect_subtitle_type(file_path: Path | str) -> str:
    """Detect if subtitle is manual or auto-generated based on filename patterns.

    Returns:
        'manual', 'auto', or 'unknown'
    """
    file_path = Path(file_path)
    name = file_path.name.lower()

    # YouTube auto-generated patterns
    auto_patterns = [
        ".auto.",
        "_auto.",
        "-auto.",
        "auto-generated",
        "autogen",
        ".a.vtt",  # YouTube pattern
    ]

    for pattern in auto_patterns:
        if pattern in name:
            return "auto"

    # If it has a language code without 'auto', likely manual
    if re.search(r"\.[a-z]{2}(-[a-z]{2})?\.vtt$", name):
        return "manual"

    return "unknown"
