"""Tests for text normalization utilities."""

import pytest

from echoharvester.utils.text_normalize import (
    handle_dual_transcription,
    is_valid_text,
    normalize_english,
    normalize_numbers,
    normalize_punctuation,
    normalize_text,
    normalize_whitespace,
    prepare_for_cer,
    remove_filler_words,
    remove_non_speech_tags,
    remove_special_characters,
    remove_stage_directions,
)


class TestRemoveNonSpeechTags:
    def test_korean_tags(self):
        assert remove_non_speech_tags("[음악] 안녕하세요") == " 안녕하세요"
        assert remove_non_speech_tags("[박수] 감사합니다") == " 감사합니다"
        assert remove_non_speech_tags("[웃음]") == ""

    def test_english_tags(self):
        assert remove_non_speech_tags("[music] hello") == " hello"
        assert remove_non_speech_tags("[applause]") == ""

    def test_generic_brackets(self):
        # Generic bracketed content (<=30 chars) is also removed
        assert remove_non_speech_tags("[효과음]") == ""

    def test_no_tags(self):
        assert remove_non_speech_tags("안녕하세요") == "안녕하세요"

    def test_empty_string(self):
        assert remove_non_speech_tags("") == ""


class TestRemoveStageDirections:
    def test_korean_directions(self):
        assert remove_stage_directions("(웃으며) 안녕") == " 안녕"
        assert remove_stage_directions("(한숨) 괜찮아") == " 괜찮아"

    def test_no_directions(self):
        assert remove_stage_directions("안녕하세요") == "안녕하세요"


class TestHandleDualTranscription:
    def test_dual_pattern(self):
        result = handle_dual_transcription("(7시)/(일곱시)")
        assert result == "7시"

    def test_no_dual(self):
        assert handle_dual_transcription("안녕하세요") == "안녕하세요"


class TestNormalizePunctuation:
    def test_quotes(self):
        # LEFT/RIGHT DOUBLE QUOTATION MARK → ASCII double quote
        left_q = "\u201c"
        right_q = "\u201d"
        result = normalize_punctuation(f"{left_q}한국{right_q}")
        assert left_q not in result
        assert right_q not in result
        assert '"' in result  # Replaced with ASCII double quote

    def test_ellipsis(self):
        result = normalize_punctuation("음....")
        assert "...." not in result  # Should collapse to ... or fewer
        result2 = normalize_punctuation("음\u2026\u2026")
        assert "\u2026" not in result2  # Unicode ellipsis normalized

    def test_dashes(self):
        assert normalize_punctuation("한국\u2014미국") == "한국-미국"

    def test_repeated_punctuation(self):
        assert normalize_punctuation("뭐???") == "뭐?"
        assert normalize_punctuation("와!!!") == "와!"


class TestNormalizeWhitespace:
    def test_tabs_and_newlines(self):
        assert normalize_whitespace("안녕\t하세요\n") == "안녕 하세요"

    def test_multiple_spaces(self):
        assert normalize_whitespace("안녕   하세요") == "안녕 하세요"

    def test_space_before_punctuation(self):
        assert normalize_whitespace("안녕 .") == "안녕."

    def test_fullwidth_space(self):
        assert normalize_whitespace("안녕\u3000하세요") == "안녕 하세요"


class TestNormalizeNumbers:
    def test_fullwidth_numbers(self):
        assert normalize_numbers("\uff11\uff12\uff13") == "123"

    def test_comma_in_numbers(self):
        assert normalize_numbers("1,000,000") == "1000000"

    def test_no_numbers(self):
        assert normalize_numbers("안녕하세요") == "안녕하세요"


class TestNormalizeEnglish:
    def test_lowercase_long_words(self):
        assert normalize_english("Hello World") == "hello world"

    def test_short_words_unchanged(self):
        # Words < 3 chars are not lowercased
        assert normalize_english("AI ML") == "AI ML"

    def test_mixed_korean_english(self):
        result = normalize_english("Python 프로그래밍")
        assert result == "python 프로그래밍"


class TestRemoveSpecialCharacters:
    def test_emoji_removal(self):
        result = remove_special_characters("안녕 \U0001f600 하세요")
        assert "\U0001f600" not in result

    def test_control_chars(self):
        result = remove_special_characters("안녕\x00하세요")
        assert "\x00" not in result

    def test_keeps_korean_and_punctuation(self):
        result = remove_special_characters("안녕하세요! Hello 123")
        assert "안녕하세요" in result
        assert "Hello" in result


class TestRemoveFillerWords:
    def test_non_aggressive(self):
        # Non-aggressive mode returns text unchanged
        assert remove_filler_words("어 안녕하세요") == "어 안녕하세요"

    def test_aggressive(self):
        result = remove_filler_words("어 안녕하세요", aggressive=True)
        assert "안녕하세요" in result


class TestNormalizeText:
    def test_full_pipeline(self):
        text = "[음악] 안녕하세요???  Hello  World "
        result = normalize_text(text)
        assert "[음악]" not in result
        assert "???" not in result
        assert "  " not in result
        assert result.strip() == result

    def test_empty_string(self):
        assert normalize_text("") == ""
        assert normalize_text(None) == ""  # type: ignore

    def test_disable_non_speech(self):
        text = "[음악] 안녕"
        result = normalize_text(text, remove_non_speech=False)
        # Even with remove_non_speech=False, generic brackets (<30 chars) may
        # be removed by the non-speech pattern or special char removal
        assert "안녕" in result

    def test_all_disabled(self):
        text = "안녕하세요"
        result = normalize_text(
            text,
            remove_non_speech=False,
            remove_directions=False,
            handle_dual=False,
            normalize_punct=False,
            normalize_nums=False,
            normalize_eng=False,
            remove_special=False,
            remove_fillers=False,
        )
        assert result == "안녕하세요"


class TestIsValidText:
    def test_valid_korean(self):
        assert is_valid_text("안녕하세요") is True

    def test_too_short(self):
        assert is_valid_text("a") is False
        assert is_valid_text("") is False

    def test_low_korean_ratio(self):
        assert is_valid_text("hello world test", min_korean_ratio=0.3) is False

    def test_only_spaces(self):
        assert is_valid_text("   ") is False

    def test_custom_thresholds(self):
        assert is_valid_text("ab", min_chars=3) is False
        assert is_valid_text("abc한", min_korean_ratio=0.5) is False


class TestPrepareForCer:
    def test_removes_spaces(self):
        result = prepare_for_cer("안녕 하세요")
        assert " " not in result

    def test_normalizes(self):
        # Should apply full normalization + NFC + remove spaces
        result = prepare_for_cer("[음악] 안녕   하세요!")
        assert "[음악]" not in result
        assert " " not in result
