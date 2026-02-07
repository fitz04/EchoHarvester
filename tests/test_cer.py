"""Tests for CER (Character Error Rate) calculation."""

import pytest

from echoharvester.utils.cer import (
    calculate_cer,
    calculate_wer,
    get_alignment_details,
    is_acceptable_cer,
    levenshtein_distance,
    normalize_for_cer,
)


class TestNormalizeForCer:
    def test_removes_spaces(self):
        assert " " not in normalize_for_cer("안녕 하세요")

    def test_lowercase(self):
        assert normalize_for_cer("Hello") == "hello"

    def test_nfc_normalization(self):
        # Composed vs decomposed Korean
        import unicodedata
        composed = "한"
        decomposed = unicodedata.normalize("NFD", composed)
        assert normalize_for_cer(composed) == normalize_for_cer(decomposed)


class TestLevenshteinDistance:
    def test_identical(self):
        assert levenshtein_distance("abc", "abc") == 0

    def test_empty_strings(self):
        assert levenshtein_distance("", "") == 0
        assert levenshtein_distance("abc", "") == 3
        assert levenshtein_distance("", "abc") == 3

    def test_single_substitution(self):
        assert levenshtein_distance("abc", "axc") == 1

    def test_single_insertion(self):
        assert levenshtein_distance("ac", "abc") == 1

    def test_single_deletion(self):
        assert levenshtein_distance("abc", "ac") == 1

    def test_korean(self):
        assert levenshtein_distance("안녕하세요", "안녕하세요") == 0
        assert levenshtein_distance("안녕하세요", "안녕하세") == 1

    def test_completely_different(self):
        assert levenshtein_distance("abc", "xyz") == 3


class TestCalculateCer:
    def test_identical(self):
        assert calculate_cer("안녕하세요", "안녕하세요") == 0.0

    def test_completely_different(self):
        cer = calculate_cer("가나다", "라마바")
        assert cer == 1.0  # 3 substitutions / 3 chars

    def test_empty_reference(self):
        assert calculate_cer("", "something") == 1.0
        assert calculate_cer("", "") == 0.0

    def test_longer_hypothesis(self):
        # CER can exceed 1.0 if hypothesis is much longer
        cer = calculate_cer("가", "가나다라마")
        assert cer > 1.0

    def test_normalization(self):
        # Spaces should be removed during normalization
        cer = calculate_cer("안녕 하세요", "안녕하세요")
        assert cer == 0.0

    def test_no_normalization(self):
        cer = calculate_cer("안녕 하세요", "안녕하세요", normalize=False)
        assert cer > 0.0  # Space makes a difference


class TestCalculateWer:
    def test_identical(self):
        wer = calculate_wer("안녕 하세요", "안녕 하세요")
        assert wer == 0.0

    def test_empty_reference(self):
        assert calculate_wer("", "something") == 1.0
        assert calculate_wer("", "") == 0.0


class TestGetAlignmentDetails:
    def test_identical(self):
        details = get_alignment_details("안녕하세요", "안녕하세요")
        assert details["cer"] == 0.0
        assert details["substitutions"] == 0
        assert details["deletions"] == 0
        assert details["insertions"] == 0

    def test_substitution(self):
        details = get_alignment_details("가나다", "가마다")
        assert details["substitutions"] == 1
        assert details["cer"] == pytest.approx(1 / 3, abs=0.01)

    def test_empty_reference(self):
        details = get_alignment_details("", "abc")
        assert details["cer"] == 1.0
        assert details["insertions"] == 3

    def test_empty_both(self):
        details = get_alignment_details("", "")
        assert details["cer"] == 0.0

    def test_reference_length(self):
        details = get_alignment_details("가나다", "가나")
        assert details["reference_length"] == 3
        assert details["hypothesis_length"] == 2


class TestIsAcceptableCer:
    def test_manual_within_threshold(self):
        assert is_acceptable_cer(0.10, "manual") is True

    def test_manual_exceeds_threshold(self):
        assert is_acceptable_cer(0.20, "manual") is False

    def test_auto_within_threshold(self):
        assert is_acceptable_cer(0.05, "auto") is True

    def test_auto_exceeds_threshold(self):
        assert is_acceptable_cer(0.12, "auto") is False

    def test_custom_thresholds(self):
        assert is_acceptable_cer(0.25, "manual", manual_threshold=0.30) is True
        assert is_acceptable_cer(0.25, "auto", auto_threshold=0.20) is False

    def test_zero_cer(self):
        assert is_acceptable_cer(0.0, "manual") is True
        assert is_acceptable_cer(0.0, "auto") is True

    def test_boundary(self):
        assert is_acceptable_cer(0.15, "manual") is True
        assert is_acceptable_cer(0.10, "auto") is True
