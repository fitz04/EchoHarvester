"""Korean text normalization for ASR training data."""

import re
import unicodedata


# Non-speech tags to remove
NON_SPEECH_TAGS = [
    # Korean
    "음악",
    "박수",
    "웃음",
    "박수소리",
    "웃음소리",
    "환호",
    "탄식",
    "한숨",
    "노래",
    "효과음",
    "배경음악",
    "배경음",
    "인터뷰",
    "광고",
    "자막",
    "수정",
    "정정",
    # English
    "music",
    "applause",
    "laughter",
    "cheering",
    "sighing",
    "singing",
    "sound effect",
    "background music",
    "bgm",
    "inaudible",
    "indistinct",
    "crosstalk",
]

# Regex pattern for bracketed non-speech content
NON_SPEECH_PATTERN = re.compile(
    r"\[([^\]]*(?:" + "|".join(re.escape(tag) for tag in NON_SPEECH_TAGS) + r")[^\]]*)\]",
    re.IGNORECASE,
)

# Parenthetical stage directions
STAGE_DIRECTION_PATTERN = re.compile(
    r"\(([^)]*(?:웃으며|울며|화내며|놀라며|한숨|침묵|말을 더듬|속삭|외치)[^)]*)\)",
    re.IGNORECASE,
)


def remove_non_speech_tags(text: str) -> str:
    """Remove non-speech annotations like [음악], [박수], etc."""
    # Remove bracketed tags
    text = NON_SPEECH_PATTERN.sub("", text)

    # Remove generic bracketed content that looks like annotations
    text = re.sub(r"\[[^\]]{0,30}\]", "", text)

    return text


def remove_stage_directions(text: str) -> str:
    """Remove parenthetical stage directions like (웃으며), (한숨)."""
    text = STAGE_DIRECTION_PATTERN.sub("", text)
    return text


def handle_dual_transcription(text: str) -> str:
    """Handle dual transcription patterns like (7시)/(일곱시).

    Keeps the written form (first option) for consistency with Whisper.
    """
    # Pattern: (written)/(spoken) or (spoken)/(written)
    # Keep the first one (typically the written/display form)
    text = re.sub(r"\(([^)]+)\)/\([^)]+\)", r"\1", text)
    text = re.sub(r"\([^)]+\)/\(([^)]+)\)", r"\1", text)

    return text


def normalize_punctuation(text: str) -> str:
    """Normalize punctuation marks."""
    # Normalize quotes (Unicode fancy quotes → ASCII)
    text = re.sub(r"[\u201c\u201d\u201e\u300c\u300d\u300e\u300f]", '"', text)
    text = re.sub(r"[\u2018\u2019\u201a`]", "'", text)

    # Normalize ellipsis
    text = re.sub(r"\.{2,}", "...", text)
    text = re.sub(r"…+", "...", text)

    # Normalize dashes
    text = re.sub(r"[—–]", "-", text)

    # Remove repeated punctuation (keep one)
    text = re.sub(r"([.!?])\1+", r"\1", text)
    text = re.sub(r"([,;:])\1+", r"\1", text)

    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace characters."""
    # Replace various whitespace with regular space
    text = re.sub(r"[\t\r\n\xa0\u3000]+", " ", text)

    # Remove multiple spaces
    text = re.sub(r" +", " ", text)

    # Remove spaces around punctuation
    text = re.sub(r"\s+([.!?,;:])", r"\1", text)

    return text.strip()


def normalize_numbers(text: str) -> str:
    """Normalize number representations.

    Keep Arabic numerals as-is (Whisper typically outputs Arabic numerals).
    """
    # Full-width to half-width numbers
    for i, char in enumerate("0123456789"):
        text = text.replace(chr(0xFF10 + i), char)

    # Remove commas in numbers (1,000 -> 1000)
    text = re.sub(r"(\d),(\d)", r"\1\2", text)

    return text


def normalize_english(text: str) -> str:
    """Normalize English text within Korean text."""
    # Lowercase English for consistency
    def lowercase_english(match):
        return match.group(0).lower()

    # Only lowercase pure English words, not abbreviations
    text = re.sub(r"\b[A-Za-z]{3,}\b", lowercase_english, text)

    return text


def remove_special_characters(text: str) -> str:
    """Remove unnecessary special characters."""
    # Keep: Korean, English, numbers, basic punctuation
    # Remove: emojis, special symbols, etc.

    # Remove emojis and symbols
    text = re.sub(
        r"[\U0001F600-\U0001F64F"  # emoticons
        r"\U0001F300-\U0001F5FF"  # symbols & pictographs
        r"\U0001F680-\U0001F6FF"  # transport & map
        r"\U0001F1E0-\U0001F1FF"  # flags
        r"\U00002702-\U000027B0"  # dingbats
        r"\U0001F900-\U0001F9FF"  # supplemental symbols
        r"]+",
        "",
        text,
    )

    # Remove control characters
    text = "".join(char for char in text if unicodedata.category(char)[0] != "C")

    return text


def remove_filler_words(text: str, aggressive: bool = False) -> str:
    """Optionally remove filler words.

    Args:
        text: Input text
        aggressive: If True, remove more filler words (might affect meaning)
    """
    if not aggressive:
        return text

    # Korean filler words
    fillers = [
        r"\b어\b",
        r"\b음\b",
        r"\b그\b",
        r"\b아\b",
        r"\b뭐\b",
        r"\b저\b",
        r"\b이제\b",
    ]

    for filler in fillers:
        text = re.sub(filler, "", text)

    return normalize_whitespace(text)


def normalize_text(
    text: str,
    remove_non_speech: bool = True,
    remove_directions: bool = True,
    handle_dual: bool = True,
    normalize_punct: bool = True,
    normalize_nums: bool = True,
    normalize_eng: bool = True,
    remove_special: bool = True,
    remove_fillers: bool = False,
) -> str:
    """Apply full text normalization pipeline.

    Args:
        text: Input text to normalize
        remove_non_speech: Remove [음악], [박수] etc.
        remove_directions: Remove (웃으며), (한숨) etc.
        handle_dual: Handle dual transcription (7시)/(일곱시)
        normalize_punct: Normalize punctuation marks
        normalize_nums: Normalize number representations
        normalize_eng: Normalize English text (lowercase)
        remove_special: Remove emojis and special symbols
        remove_fillers: Remove filler words (aggressive)

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Apply normalizations in order
    if remove_non_speech:
        text = remove_non_speech_tags(text)

    if remove_directions:
        text = remove_stage_directions(text)

    if handle_dual:
        text = handle_dual_transcription(text)

    if normalize_punct:
        text = normalize_punctuation(text)

    if normalize_nums:
        text = normalize_numbers(text)

    if normalize_eng:
        text = normalize_english(text)

    if remove_special:
        text = remove_special_characters(text)

    if remove_fillers:
        text = remove_filler_words(text, aggressive=True)

    # Always normalize whitespace at the end
    text = normalize_whitespace(text)

    return text


def is_valid_text(text: str, min_chars: int = 2, min_korean_ratio: float = 0.3) -> bool:
    """Check if text is valid for ASR training.

    Args:
        text: Text to check
        min_chars: Minimum number of characters
        min_korean_ratio: Minimum ratio of Korean characters

    Returns:
        True if text is valid
    """
    if not text or len(text) < min_chars:
        return False

    # Count Korean characters
    korean_chars = sum(1 for c in text if "\uac00" <= c <= "\ud7a3")
    total_chars = sum(1 for c in text if not c.isspace())

    if total_chars == 0:
        return False

    korean_ratio = korean_chars / total_chars

    return korean_ratio >= min_korean_ratio


def prepare_for_cer(text: str) -> str:
    """Prepare text for CER calculation.

    - Remove all spaces (character-level comparison for Korean)
    - Normalize to NFC form
    """
    text = normalize_text(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", "")
    return text
