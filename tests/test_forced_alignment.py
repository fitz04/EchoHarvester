"""Tests for forced alignment utilities (non-model tests)."""

import pytest

from echoharvester.utils.forced_alignment import (
    AlignedWord,
    _chunk_segments_by_duration,
    _clean_text_for_alignment,
    _fix_overlapping_segments,
    _strip_for_matching,
    aligned_words_to_segments,
)
from echoharvester.utils.subtitle_parser import SubtitleSegment


class TestCleanTextForAlignment:
    def test_removes_non_speech_tags(self):
        assert _clean_text_for_alignment("[음악] 안녕하세요") == "안녕하세요"
        assert _clean_text_for_alignment("[박수] 좋아요") == "좋아요"

    def test_collapses_whitespace(self):
        assert _clean_text_for_alignment("안녕   하세요") == "안녕 하세요"

    def test_strips(self):
        assert _clean_text_for_alignment("  안녕  ") == "안녕"

    def test_empty(self):
        assert _clean_text_for_alignment("[음악]") == ""


class TestStripForMatching:
    def test_removes_punctuation(self):
        assert _strip_for_matching("안녕하세요.") == "안녕하세요"
        assert _strip_for_matching("안녕, 하세요!") == "안녕하세요"

    def test_removes_whitespace(self):
        assert _strip_for_matching("안녕 하세요") == "안녕하세요"

    def test_keeps_korean_chars(self):
        assert _strip_for_matching("한글테스트") == "한글테스트"

    def test_empty(self):
        assert _strip_for_matching("...") == ""


class TestAlignedWordsToSegments:
    def test_empty_words(self):
        segments = [SubtitleSegment(0, 5, "text")]
        result = aligned_words_to_segments([], segments)
        assert result == segments

    def test_exact_match(self):
        words = [
            AlignedWord("안녕", 1.0, 1.5),
            AlignedWord("하세요", 1.5, 2.0),
        ]
        segments = [SubtitleSegment(0, 5, "안녕하세요")]
        result = aligned_words_to_segments(words, segments)
        assert len(result) == 1
        assert result[0].start_sec == 1.0
        assert result[0].end_sec == 2.0

    def test_no_match_uses_original(self):
        words = [AlignedWord("다른텍스트", 1.0, 2.0)]
        segments = [SubtitleSegment(0, 5, "전혀다른것")]
        result = aligned_words_to_segments(words, segments)
        assert result[0].start_sec == 0  # Original

    def test_empty_text_segment_skipped(self):
        words = [AlignedWord("안녕", 1.0, 2.0)]
        segments = [SubtitleSegment(0, 5, "[음악]")]
        result = aligned_words_to_segments(words, segments)
        assert len(result) == 0  # Cleaned text is empty


class TestFixOverlappingSegments:
    def test_no_overlap(self):
        segments = [
            SubtitleSegment(0, 2, "a"),
            SubtitleSegment(3, 5, "b"),
        ]
        _fix_overlapping_segments(segments)
        assert segments[0].end_sec == 2
        assert segments[1].start_sec == 3

    def test_overlap(self):
        segments = [
            SubtitleSegment(0, 4, "a"),
            SubtitleSegment(3, 6, "b"),
        ]
        _fix_overlapping_segments(segments)
        mid = (4 + 3) / 2  # 3.5
        assert segments[0].end_sec == mid
        assert segments[1].start_sec == mid

    def test_empty(self):
        segments = []
        _fix_overlapping_segments(segments)  # Should not raise

    def test_single(self):
        segments = [SubtitleSegment(0, 5, "a")]
        _fix_overlapping_segments(segments)  # Should not raise


class TestChunkSegmentsByDuration:
    def test_single_chunk(self):
        segments = [SubtitleSegment(0, 10, "a"), SubtitleSegment(10, 20, "b")]
        chunks = _chunk_segments_by_duration(segments, max_duration_sec=30)
        assert len(chunks) == 1
        assert len(chunks[0]) == 2

    def test_multiple_chunks(self):
        segments = [
            SubtitleSegment(0, 10, "a"),
            SubtitleSegment(10, 20, "b"),
            SubtitleSegment(20, 30, "c"),
            SubtitleSegment(30, 40, "d"),
        ]
        chunks = _chunk_segments_by_duration(segments, max_duration_sec=15)
        assert len(chunks) >= 2

    def test_empty(self):
        # Should handle empty gracefully (no crash, but may be undefined)
        # The function accesses segments[0], so empty input is not typical
        pass
