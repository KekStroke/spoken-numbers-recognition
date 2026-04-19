from __future__ import annotations

from src.asr.tokenizers.compact import CompactNumberTokenizer
from src.asr.tokenizers.words import RussianNumberTokenizer

NumberTokenizer = RussianNumberTokenizer | CompactNumberTokenizer


def build_tokenizer(config: dict[str, object] | None = None) -> NumberTokenizer:
    tokenizer_type = str((config or {}).get("type", "russian_number_words"))
    blank_id = int((config or {}).get("blank_id", 0))

    if tokenizer_type == "russian_number_words":
        return RussianNumberTokenizer(blank_id=blank_id)
    if tokenizer_type == "russian_number_compact":
        return CompactNumberTokenizer(blank_id=blank_id)
    raise ValueError(f"Unsupported tokenizer type in checkpoint: {tokenizer_type!r}")

