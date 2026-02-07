"""Forced alignment using Qwen3-ForcedAligner for precise audio-text timestamp alignment."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from echoharvester.utils.subtitle_parser import SubtitleSegment

logger = logging.getLogger(__name__)

MAX_AUDIO_DURATION_SEC = 270  # ForcedAligner supports up to 5 min, use 4.5 min for safety

# Non-speech tags that confuse the aligner
_NON_SPEECH_PATTERN = re.compile(
    r"\[(?:음악|박수|웃음|환호|박수소리|음악 소리|잡음|침묵|한숨|노래)\]",
    re.IGNORECASE,
)


def _clean_text_for_alignment(text: str) -> str:
    """Remove non-speech tags and clean text before sending to ForcedAligner."""
    # Remove [음악], [박수], etc.
    text = _NON_SPEECH_PATTERN.sub("", text)
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


@dataclass
class AlignedWord:
    """A word with precise timestamps from forced alignment."""

    text: str
    start_time: float
    end_time: float


class ForcedAligner:
    """Wrapper for Qwen3-ForcedAligner-0.6B."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        device: str = "auto",
        compute_type: str = "auto",
        language: str = "ko",
    ):
        self.model_name = model_name
        self.language = language
        self._model = None

        # FA (0.6B) runs on CPU to avoid MPS memory pressure
        # on Apple Silicon with limited unified memory.
        # The ASR model (1.7B, Stage 4) needs MPS more.
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda:0"
            else:
                # Use CPU for FA even if MPS is available
                self.device = "cpu"
        else:
            self.device = device

        # CPU always uses float32
        if self.device == "cpu":
            self.dtype = torch.float32
        elif compute_type in ("auto", "bfloat16"):
            self.dtype = torch.bfloat16
        elif compute_type == "float16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        logger.info(f"ForcedAligner device: {self.device}, dtype: {self.dtype}")

    def _load_model(self):
        """Lazy load the aligner model."""
        if self._model is not None:
            return

        from qwen_asr import Qwen3ForcedAligner

        logger.info(f"Loading ForcedAligner model: {self.model_name}")
        self._model = Qwen3ForcedAligner.from_pretrained(
            self.model_name,
            dtype=self.dtype,
            device_map=self.device,
        )
        logger.info("ForcedAligner model loaded")

    def unload(self):
        """Explicitly unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("ForcedAligner model unloaded")

    def align(
        self,
        audio_path: str | Path,
        text: str,
    ) -> list[AlignedWord]:
        """Align text with audio and return word-level timestamps."""
        self._load_model()

        lang_map = {
            "ko": "Korean", "en": "English", "ja": "Japanese",
            "zh": "Chinese", "fr": "French", "de": "German",
        }
        language = lang_map.get(self.language, self.language)

        assert self._model is not None
        results = self._model.align(
            audio=str(audio_path),
            text=text,
            language=language,
        )

        words = []
        if results and results[0]:
            for token in results[0]:
                words.append(AlignedWord(
                    text=token.text,
                    start_time=token.start_time,
                    end_time=token.end_time,
                ))

        return words


def _strip_for_matching(text: str) -> str:
    """Remove whitespace and punctuation for character-level matching.

    The ForcedAligner output strips punctuation, so we must do the same
    for the original text to get exact matches.
    """
    # Remove whitespace and common punctuation
    return re.sub(r"[\s\W]", "", text)


def aligned_words_to_segments(
    words: list[AlignedWord],
    original_segments: list[SubtitleSegment],
) -> list[SubtitleSegment]:
    """Convert word-level alignment back into sentence-level segments.

    Uses character-position mapping to find precise start/end times
    for each original segment's text within the aligned word sequence.
    Falls back to original timestamps when no match is found.
    """
    if not words:
        return list(original_segments)

    # Build character-to-word index mapping (ignoring spaces)
    char_to_word: list[int] = []
    for word_idx, w in enumerate(words):
        for _ in w.text:
            char_to_word.append(word_idx)

    aligned_full_text = "".join(w.text for w in words)

    results: list[SubtitleSegment] = []
    search_pos = 0
    stats = {"exact": 0, "partial": 0, "none": 0}

    for orig_seg in original_segments:
        clean_text = _strip_for_matching(_clean_text_for_alignment(orig_seg.text))
        if not clean_text:
            continue

        # Try exact match first
        match_pos = aligned_full_text.find(clean_text, search_pos)

        if match_pos != -1:
            stats["exact"] += 1
        else:
            # Try partial matching with first 8 chars
            partial = clean_text[:min(8, len(clean_text))]
            match_pos = aligned_full_text.find(partial, search_pos)
            if match_pos != -1:
                stats["partial"] += 1

        if match_pos == -1:
            logger.debug(f"FA: no match for '{orig_seg.text[:30]}...', using original")
            results.append(orig_seg)
            stats["none"] += 1
            continue

        match_end = match_pos + len(clean_text)
        search_pos = match_end

        # Map character positions to word indices
        if match_pos < len(char_to_word) and (match_end - 1) < len(char_to_word):
            start_word_idx = char_to_word[match_pos]
            end_word_idx = char_to_word[match_end - 1]

            aligned_seg = SubtitleSegment(
                start_sec=words[start_word_idx].start_time,
                end_sec=words[end_word_idx].end_time,
                text=orig_seg.text,
            )
            results.append(aligned_seg)
        else:
            results.append(orig_seg)
            stats["none"] += 1

    logger.info(
        f"[FA] Match stats: exact={stats['exact']}, "
        f"partial={stats['partial']}, none={stats['none']}"
    )

    return results


def _fix_overlapping_segments(segments: list[SubtitleSegment]) -> None:
    """Fix overlapping segment boundaries in-place.

    When two segments overlap, snap the boundary to the midpoint
    of the overlapping region. This prevents audio from one segment
    bleeding into another.
    """
    for i in range(len(segments) - 1):
        curr = segments[i]
        next_seg = segments[i + 1]

        if curr.end_sec > next_seg.start_sec:
            # Overlap detected: snap to midpoint
            midpoint = (curr.end_sec + next_seg.start_sec) / 2
            segments[i] = SubtitleSegment(
                start_sec=curr.start_sec,
                end_sec=midpoint,
                text=curr.text,
            )
            segments[i + 1] = SubtitleSegment(
                start_sec=midpoint,
                end_sec=next_seg.end_sec,
                text=next_seg.text,
            )


def align_subtitles_with_audio(
    audio_path: str | Path,
    subtitle_segments: list[SubtitleSegment],
    aligner: ForcedAligner,
    max_chunk_sec: float = MAX_AUDIO_DURATION_SEC,
) -> list[SubtitleSegment]:
    """Align subtitle segments with audio using forced alignment.

    Uses VAD-aware chunking to split long audio at silence boundaries,
    then bulk-aligns each chunk. Falls back to original timestamps
    for segments that can't be matched.
    """
    if not subtitle_segments:
        return []

    audio_path = Path(audio_path)

    from echoharvester.utils.audio_utils import extract_segment_sync, get_audio_duration

    total_duration = get_audio_duration(audio_path)
    logger.info(
        f"[FA] Aligning {len(subtitle_segments)} segments with "
        f"{total_duration:.1f}s audio"
    )

    # Short audio: align all at once
    if total_duration <= max_chunk_sec:
        full_text = " ".join(
            _clean_text_for_alignment(seg.text) for seg in subtitle_segments
        )
        words = aligner.align(audio_path, full_text)
        if words:
            segments = aligned_words_to_segments(words, subtitle_segments)
        else:
            segments = list(subtitle_segments)
        _fix_overlapping_segments(segments)
        return segments

    # Long audio: chunk using VAD boundaries
    chunks = _chunk_segments_by_vad(
        subtitle_segments, audio_path, max_chunk_sec,
    )
    logger.info(f"[FA] VAD-based chunking: {len(chunks)} chunks")

    all_segments: list[SubtitleSegment] = []
    total_chunks = len(chunks)

    for chunk_idx, chunk in enumerate(chunks):
        chunk_start = max(0, chunk[0].start_sec - 1.0)
        chunk_end = min(total_duration, chunk[-1].end_sec + 1.0)

        logger.info(
            f"[FA] Chunk {chunk_idx + 1}/{total_chunks}: "
            f"{chunk_start:.1f}-{chunk_end:.1f}s "
            f"({chunk_end - chunk_start:.1f}s), {len(chunk)} segments"
        )

        chunk_audio = audio_path.parent / f"_align_chunk_{chunk_idx}.wav"
        try:
            extract_segment_sync(
                audio_path, chunk_audio,
                chunk_start, chunk_end,
                padding_sec=0, sample_rate=16000,
            )

            chunk_cleaned = " ".join(
                _clean_text_for_alignment(seg.text) for seg in chunk
            )
            words = aligner.align(chunk_audio, chunk_cleaned)

            if words:
                for w in words:
                    w.start_time += chunk_start
                    w.end_time += chunk_start

                chunk_results = aligned_words_to_segments(words, chunk)
                all_segments.extend(chunk_results)
            else:
                all_segments.extend(chunk)

        except Exception as e:
            logger.error(f"[FA] Chunk {chunk_idx + 1}/{total_chunks} error: {e}")
            all_segments.extend(chunk)
        finally:
            if chunk_audio.exists():
                chunk_audio.unlink()

    # Final overlap fix
    _fix_overlapping_segments(all_segments)

    return all_segments


def _chunk_segments_by_vad(
    segments: list[SubtitleSegment],
    audio_path: Path,
    max_duration_sec: float,
) -> list[list[SubtitleSegment]]:
    """Group subtitle segments into chunks using VAD silence boundaries.

    Instead of cutting at arbitrary duration limits, finds actual silence
    gaps in the audio and uses those as chunk boundaries. This prevents
    cutting mid-speech which causes alignment drift.
    """
    from echoharvester.utils.audio_utils import load_audio
    from echoharvester.utils.vad import get_vad

    # Get speech segments from VAD
    audio, sr = load_audio(audio_path)
    vad = get_vad()
    speech_segs = vad.get_speech_timestamps(
        audio, sr, min_silence_duration_ms=300,
    )

    if not speech_segs:
        # VAD found no speech, fall back to duration-based chunking
        return _chunk_segments_by_duration(segments, max_duration_sec)

    # Build list of silence gaps (>300ms) as potential split points
    silence_gaps: list[float] = []  # midpoints of silence gaps
    for i in range(len(speech_segs) - 1):
        gap_start = speech_segs[i].end_sec
        gap_end = speech_segs[i + 1].start_sec
        if gap_end - gap_start >= 0.3:
            silence_gaps.append((gap_start + gap_end) / 2)

    logger.info(
        f"[FA] VAD: {len(speech_segs)} speech segments, "
        f"{len(silence_gaps)} silence gaps (≥300ms)"
    )

    if not silence_gaps:
        return _chunk_segments_by_duration(segments, max_duration_sec)

    # Group subtitle segments into chunks, splitting at silence gaps
    chunks: list[list[SubtitleSegment]] = []
    current_chunk: list[SubtitleSegment] = []
    chunk_start_time = segments[0].start_sec if segments else 0
    gap_idx = 0

    for seg in segments:
        elapsed = seg.end_sec - chunk_start_time

        # Check if we should split at a silence gap
        if elapsed > max_duration_sec and current_chunk:
            # Find the best silence gap near this segment boundary
            best_gap = _find_nearest_silence_gap(
                silence_gaps, seg.start_sec, gap_idx,
            )

            if best_gap is not None:
                # Split at the silence gap
                # Move segments after the gap to a new chunk
                split_chunk: list[SubtitleSegment] = []
                keep_chunk: list[SubtitleSegment] = []
                for s in current_chunk:
                    if s.start_sec < best_gap:
                        keep_chunk.append(s)
                    else:
                        split_chunk.append(s)

                if keep_chunk:
                    chunks.append(keep_chunk)
                current_chunk = split_chunk + [seg]
                chunk_start_time = current_chunk[0].start_sec if current_chunk else seg.start_sec
                # Advance gap index
                while gap_idx < len(silence_gaps) and silence_gaps[gap_idx] <= best_gap:
                    gap_idx += 1
            else:
                # No nearby gap, just split at duration boundary
                chunks.append(current_chunk)
                current_chunk = [seg]
                chunk_start_time = seg.start_sec
        else:
            current_chunk.append(seg)

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _find_nearest_silence_gap(
    silence_gaps: list[float],
    target_time: float,
    start_idx: int,
) -> float | None:
    """Find the silence gap nearest to target_time."""
    best_gap = None
    best_dist = float("inf")

    for i in range(start_idx, len(silence_gaps)):
        gap = silence_gaps[i]
        dist = abs(gap - target_time)

        if dist < best_dist:
            best_dist = dist
            best_gap = gap

        # Stop searching if we're past the target by a lot
        if gap > target_time + 60:
            break

    # Only use if within reasonable distance (30 seconds)
    if best_gap is not None and best_dist < 30:
        return best_gap
    return None


def _chunk_segments_by_duration(
    segments: list[SubtitleSegment],
    max_duration_sec: float,
) -> list[list[SubtitleSegment]]:
    """Fallback: group subtitle segments by duration (original method)."""
    chunks: list[list[SubtitleSegment]] = []
    current_chunk: list[SubtitleSegment] = []
    chunk_start = segments[0].start_sec if segments else 0

    for seg in segments:
        elapsed = seg.end_sec - chunk_start
        if elapsed > max_duration_sec and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [seg]
            chunk_start = seg.start_sec
        else:
            current_chunk.append(seg)

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
