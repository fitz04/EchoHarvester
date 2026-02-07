"""Tests for subtitle parsing utilities."""

import pytest

from echoharvester.utils.subtitle_parser import (
    SubtitleSegment,
    clean_vtt_text,
    detect_subtitle_type,
    merge_overlapping_segments,
    parse_srt,
    parse_subtitle,
    parse_timestamp,
    parse_vtt,
    split_long_segments,
)


class TestParseTimestamp:
    def test_hhmmss_dot(self):
        assert parse_timestamp("01:02:03.456") == pytest.approx(3723.456, abs=0.001)

    def test_hhmmss_comma(self):
        assert parse_timestamp("01:02:03,456") == pytest.approx(3723.456, abs=0.001)

    def test_mmss(self):
        assert parse_timestamp("02:30.000") == pytest.approx(150.0, abs=0.001)

    def test_seconds_only(self):
        assert parse_timestamp("45.5") == pytest.approx(45.5, abs=0.001)

    def test_zero(self):
        assert parse_timestamp("00:00:00.000") == 0.0

    def test_with_whitespace(self):
        assert parse_timestamp("  01:00:00.000  ") == pytest.approx(3600.0, abs=0.001)


class TestCleanVttText:
    def test_timestamp_tags(self):
        assert clean_vtt_text("<00:00:01.000>hello") == "hello"

    def test_voice_tags(self):
        assert clean_vtt_text("<v Speaker>hello</v>") == "hello"

    def test_style_tags(self):
        assert clean_vtt_text("<i>italic</i>") == "italic"
        assert clean_vtt_text("<b>bold</b>") == "bold"

    def test_ruby_annotations(self):
        assert clean_vtt_text("<ruby>한<rt>han</rt></ruby>") == "한"

    def test_multiple_tags(self):
        text = "<00:00:01.000><v Speaker><i>hello</i></v>"
        assert clean_vtt_text(text) == "hello"

    def test_whitespace_cleanup(self):
        assert clean_vtt_text("  hello   world  ") == "hello world"

    def test_empty(self):
        assert clean_vtt_text("") == ""


class TestParseVtt:
    def test_basic_vtt(self, sample_vtt_file):
        segments = parse_vtt(sample_vtt_file)
        assert len(segments) == 3
        assert segments[0].text == "안녕하세요 여러분"
        assert segments[0].start_sec == pytest.approx(1.0, abs=0.001)
        assert segments[0].end_sec == pytest.approx(4.0, abs=0.001)

    def test_empty_vtt(self, tmp_path):
        path = tmp_path / "empty.vtt"
        path.write_text("WEBVTT\n\n", encoding="utf-8")
        segments = parse_vtt(path)
        assert segments == []

    def test_invalid_vtt(self, tmp_path):
        path = tmp_path / "invalid.vtt"
        path.write_text("not a vtt file", encoding="utf-8")
        with pytest.raises(ValueError):
            parse_vtt(path)


class TestParseSrt:
    def test_basic_srt(self, sample_srt_file):
        segments = parse_srt(sample_srt_file)
        assert len(segments) == 3
        assert segments[0].text == "안녕하세요 여러분"
        assert segments[0].start_sec == pytest.approx(1.0, abs=0.001)

    def test_empty_srt(self, tmp_path):
        path = tmp_path / "empty.srt"
        path.write_text("", encoding="utf-8")
        segments = parse_srt(path)
        assert segments == []


class TestParseSubtitle:
    def test_auto_detect_vtt(self, sample_vtt_file):
        segments = parse_subtitle(sample_vtt_file)
        assert len(segments) == 3

    def test_auto_detect_srt(self, sample_srt_file):
        segments = parse_subtitle(sample_srt_file)
        assert len(segments) == 3


class TestSubtitleSegmentDuration:
    def test_duration(self):
        seg = SubtitleSegment(start_sec=1.0, end_sec=4.0, text="test")
        assert seg.duration_sec == pytest.approx(3.0)


class TestMergeOverlappingSegments:
    def test_no_overlap(self):
        segments = [
            SubtitleSegment(0, 2, "a"),
            SubtitleSegment(3, 5, "b"),
        ]
        merged = merge_overlapping_segments(segments)
        assert len(merged) == 2

    def test_overlapping(self):
        segments = [
            SubtitleSegment(0, 3, "a"),
            SubtitleSegment(2, 5, "b"),
        ]
        merged = merge_overlapping_segments(segments)
        assert len(merged) == 1
        assert merged[0].text == "a b"
        assert merged[0].end_sec == 5

    def test_adjacent_within_gap(self):
        segments = [
            SubtitleSegment(0, 2, "a"),
            SubtitleSegment(2.05, 4, "b"),
        ]
        merged = merge_overlapping_segments(segments, max_gap_sec=0.1)
        assert len(merged) == 1

    def test_empty(self):
        assert merge_overlapping_segments([]) == []

    def test_single(self):
        segments = [SubtitleSegment(0, 2, "a")]
        assert len(merge_overlapping_segments(segments)) == 1

    def test_unsorted_input(self):
        segments = [
            SubtitleSegment(3, 5, "b"),
            SubtitleSegment(0, 2, "a"),
        ]
        merged = merge_overlapping_segments(segments)
        assert merged[0].start_sec == 0


class TestSplitLongSegments:
    def test_short_segments_unchanged(self):
        segments = [SubtitleSegment(0, 5, "hello")]
        result = split_long_segments(segments, max_duration_sec=30)
        assert len(result) == 1

    def test_long_segment_split(self):
        text = "첫번째 문장입니다. 두번째 문장입니다."
        segments = [SubtitleSegment(0, 40, text)]
        result = split_long_segments(segments, max_duration_sec=30)
        assert len(result) == 2

    def test_no_split_point(self):
        # Single sentence without punctuation
        segments = [SubtitleSegment(0, 40, "단일문장")]
        result = split_long_segments(segments, max_duration_sec=30)
        assert len(result) == 1  # Can't split

    def test_empty(self):
        assert split_long_segments([]) == []


class TestDetectSubtitleType:
    def test_auto_patterns(self, tmp_path):
        for name in ["video.auto.vtt", "video_auto.vtt", "video.a.vtt"]:
            path = tmp_path / name
            path.touch()
            assert detect_subtitle_type(path) == "auto"

    def test_manual_pattern(self, tmp_path):
        path = tmp_path / "video.ko.vtt"
        path.touch()
        assert detect_subtitle_type(path) == "manual"

    def test_unknown(self, tmp_path):
        path = tmp_path / "video.vtt"
        path.touch()
        assert detect_subtitle_type(path) == "unknown"
