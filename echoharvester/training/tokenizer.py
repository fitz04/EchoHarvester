"""Character-level tokenizer for Korean ASR."""

import logging
import unicodedata
from pathlib import Path

from echoharvester.training.config import TrainingConfig

logger = logging.getLogger(__name__)

# Special token IDs
BLANK_ID = 0
SOS_EOS_ID = 1
UNK_ID = 2
SPECIAL_TOKENS = {
    "<blk>": BLANK_ID,
    "<sos/eos>": SOS_EOS_ID,
    "<unk>": UNK_ID,
}


class CharTokenizer:
    """Character-level tokenizer for Korean ASR.

    Extracts unique characters from CutSet supervision texts,
    assigns IDs ordered by Unicode codepoint (Korean syllables naturally sorted),
    and saves to tokens.txt.
    """

    def __init__(self, tokens_path: Path | None = None):
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        if tokens_path and tokens_path.exists():
            self.load(tokens_path)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def blank_id(self) -> int:
        return BLANK_ID

    def build_from_cutset(self, cutset, output_dir: Path) -> Path:
        """Build tokenizer from CutSet supervision texts.

        Args:
            cutset: Lhotse CutSet with supervisions containing text.
            output_dir: Directory to save tokens.txt.

        Returns:
            Path to saved tokens.txt.
        """
        # Collect all unique characters
        char_set = set()
        text_count = 0

        for cut in cutset:
            for sup in cut.supervisions:
                text = sup.text
                if not text:
                    continue
                text = unicodedata.normalize("NFC", text)
                for ch in text:
                    if not ch.isspace():
                        char_set.add(ch)
                text_count += 1

        logger.info(f"Scanned {text_count} supervision texts, found {len(char_set)} unique characters")

        # Sort characters: Korean syllables (AC00-D7A3), then others by codepoint
        sorted_chars = sorted(char_set, key=lambda c: ord(c))

        # Build token mapping: special tokens first, then sorted characters
        self.token_to_id = dict(SPECIAL_TOKENS)
        next_id = len(SPECIAL_TOKENS)

        for ch in sorted_chars:
            self.token_to_id[ch] = next_id
            next_id += 1

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        # Count categories
        korean_count = sum(1 for c in sorted_chars if "\uac00" <= c <= "\ud7a3")
        logger.info(
            f"Vocabulary: {self.vocab_size} tokens "
            f"({len(SPECIAL_TOKENS)} special + {korean_count} Korean syllables + "
            f"{len(sorted_chars) - korean_count} other)"
        )

        # Save
        lang_dir = output_dir / "lang_char"
        lang_dir.mkdir(parents=True, exist_ok=True)
        tokens_path = lang_dir / "tokens.txt"
        self.save(tokens_path)

        return tokens_path

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        text = unicodedata.normalize("NFC", text)
        ids = []
        for ch in text:
            if ch.isspace():
                continue
            ids.append(self.token_to_id.get(ch, UNK_ID))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        chars = []
        for i in ids:
            if i in (BLANK_ID, SOS_EOS_ID):
                continue
            token = self.id_to_token.get(i, "")
            if token and token != "<unk>":
                chars.append(token)
        return "".join(chars)

    def save(self, path: Path) -> None:
        """Save tokens.txt in icefall format: 'token id' per line."""
        with open(path, "w", encoding="utf-8") as f:
            for token, idx in sorted(self.token_to_id.items(), key=lambda x: x[1]):
                f.write(f"{token} {idx}\n")
        logger.info(f"Saved {self.vocab_size} tokens to {path}")

    def load(self, path: Path) -> None:
        """Load tokens.txt."""
        self.token_to_id = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Last space-separated token is the ID
                parts = line.rsplit(" ", 1)
                if len(parts) == 2:
                    token, idx = parts[0], int(parts[1])
                    self.token_to_id[token] = idx

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        logger.info(f"Loaded {self.vocab_size} tokens from {path}")


def build_tokenizer(config: TrainingConfig) -> CharTokenizer:
    """Build tokenizer from prepared training data.

    Loads the train split and builds character vocabulary.
    """
    from lhotse import CutSet

    data_dir = Path(config.data_dir)
    train_path = data_dir / "train_cuts.jsonl.gz"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. Run 'train prepare' first."
        )

    logger.info(f"Building tokenizer from {train_path}")
    train_cs = CutSet.from_jsonl(train_path)

    tokenizer = CharTokenizer()
    tokens_path = tokenizer.build_from_cutset(train_cs, data_dir)
    logger.info(f"Tokenizer built: {tokenizer.vocab_size} tokens -> {tokens_path}")

    return tokenizer
