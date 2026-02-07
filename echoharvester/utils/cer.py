"""Character Error Rate (CER) calculation for Korean ASR evaluation."""

import unicodedata


def normalize_for_cer(text: str) -> str:
    """Normalize text for CER calculation.

    - Unicode NFC normalization
    - Remove spaces (Korean character-level comparison)
    - Lowercase (for consistency)
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", "")
    text = text.lower()
    return text


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost: 0 if same, 1 if different
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_cer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """Calculate Character Error Rate.

    CER = (S + D + I) / N

    Where:
        S = number of substitutions
        D = number of deletions
        I = number of insertions
        N = number of characters in reference

    Args:
        reference: Ground truth text (original subtitle)
        hypothesis: Predicted text (ASR transcription)
        normalize: Apply normalization before comparison

    Returns:
        CER as a float between 0.0 and potentially > 1.0
        (CER can exceed 1.0 if hypothesis is much longer than reference)
    """
    if normalize:
        reference = normalize_for_cer(reference)
        hypothesis = normalize_for_cer(hypothesis)

    # Handle edge cases
    if len(reference) == 0:
        return 1.0 if len(hypothesis) > 0 else 0.0

    distance = levenshtein_distance(reference, hypothesis)
    cer = distance / len(reference)

    return cer


def calculate_cer_jiwer(reference: str, hypothesis: str) -> float:
    """Calculate CER using jiwer library (if available).

    Falls back to custom implementation if jiwer not installed.
    """
    try:
        import jiwer

        # Normalize
        reference = normalize_for_cer(reference)
        hypothesis = normalize_for_cer(hypothesis)

        if len(reference) == 0:
            return 1.0 if len(hypothesis) > 0 else 0.0

        # jiwer expects word-like units, so we split into characters
        ref_chars = " ".join(list(reference))
        hyp_chars = " ".join(list(hypothesis))

        return jiwer.cer(ref_chars, hyp_chars)
    except ImportError:
        return calculate_cer(reference, hypothesis, normalize=True)


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate.

    Less common for Korean but useful for comparison.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0

    distance = levenshtein_distance(
        " ".join(ref_words),
        " ".join(hyp_words),
    )

    # Count words for WER (need word-level distance)
    # Simple approximation: character distance / avg word length
    avg_word_len = len(reference) / max(len(ref_words), 1)
    word_distance = distance / max(avg_word_len, 1)

    return word_distance / len(ref_words)


def get_alignment_details(reference: str, hypothesis: str) -> dict:
    """Get detailed alignment information for debugging.

    Returns:
        Dictionary with:
        - cer: Character Error Rate
        - substitutions: Number of substitutions
        - deletions: Number of deletions
        - insertions: Number of insertions
        - reference_length: Length of reference
        - hypothesis_length: Length of hypothesis
    """
    reference = normalize_for_cer(reference)
    hypothesis = normalize_for_cer(hypothesis)

    if len(reference) == 0:
        return {
            "cer": 1.0 if len(hypothesis) > 0 else 0.0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": len(hypothesis),
            "reference_length": 0,
            "hypothesis_length": len(hypothesis),
        }

    # Calculate distance with traceback
    m, n = len(reference), len(hypothesis)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # deletion
                    dp[i][j - 1] + 1,  # insertion
                    dp[i - 1][j - 1] + 1,  # substitution
                )

    # Traceback to count operations
    i, j = m, n
    subs = dels = ins = 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i - 1] == hypothesis[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1

    return {
        "cer": dp[m][n] / m,
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "reference_length": m,
        "hypothesis_length": n,
    }


def is_acceptable_cer(
    cer: float,
    subtitle_type: str,
    manual_threshold: float = 0.15,
    auto_threshold: float = 0.10,
) -> bool:
    """Check if CER is acceptable based on subtitle type.

    Args:
        cer: Calculated CER
        subtitle_type: 'manual' or 'auto'
        manual_threshold: CER threshold for manual subtitles
        auto_threshold: CER threshold for auto-generated subtitles

    Returns:
        True if CER is acceptable
    """
    if subtitle_type == "manual":
        return cer <= manual_threshold
    else:
        # Auto-generated subtitles: stricter threshold
        # If both original auto subtitle and whisper disagree significantly,
        # both might be wrong
        return cer <= auto_threshold
