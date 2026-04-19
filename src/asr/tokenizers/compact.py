from __future__ import annotations

from dataclasses import dataclass

COMPACT_VOCAB = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "дцать",
    "20",
    "30",
    "40",
    "50",
    "60",
    "70",
    "80",
    "90",
    "100",
    "200",
    "300",
    "400",
    "500",
    "600",
    "700",
    "800",
    "900",
    "тысяча",
]
COMPACT_UNITS = {str(i): i for i in range(1, 10)}
COMPACT_TENS = {str(i): i for i in range(20, 100, 10)}
COMPACT_HUNDREDS = {str(i): i for i in range(100, 1000, 100)}


def _collapse_ctc_ids(frame_ids: list[int], blank_id: int) -> list[int]:
    collapsed: list[int] = []
    prev = blank_id
    for idx in frame_ids:
        if idx != blank_id and idx != prev:
            collapsed.append(idx)
        prev = idx
    return collapsed


@dataclass(frozen=True)
class CompactNumberTokenizer:
    blank_id: int = 0

    def __post_init__(self) -> None:
        token_to_id = {token: idx + 1 for idx, token in enumerate(COMPACT_VOCAB)}
        object.__setattr__(self, "_token_to_id", token_to_id)
        object.__setattr__(
            self, "_id_to_token", {idx: token for token, idx in token_to_id.items()}
        )

    @property
    def tokenizer_type(self) -> str:
        return "russian_number_compact"

    @property
    def config(self) -> dict[str, object]:
        return {"type": self.tokenizer_type, "blank_id": self.blank_id}

    @property
    def vocab_size(self) -> int:
        return len(COMPACT_VOCAB) + 1

    def normalize_text(self, value: str | int) -> str:
        text = str(value).strip()
        if not text or text.startswith("-") or not text.isdigit():
            raise ValueError(f"Expected positive integer transcription, got: {value!r}")
        return text

    def encode_number(self, value: str | int) -> list[str]:
        number = int(self.normalize_text(value))
        if number == 0:
            raise ValueError("Zero is not supported for this dataset.")

        thousands, remainder = divmod(number, 1000)
        tokens: list[str] = []
        if thousands:
            tokens.extend(self._triplet_to_tokens(thousands))
            tokens.append("тысяча")
        if remainder:
            tokens.extend(self._triplet_to_tokens(remainder))
        return tokens

    def encode(self, value: str | int) -> list[int]:
        return [self._token_to_id[token] for token in self.encode_number(value)]

    def encode_as_text(self, value: str | int) -> str:
        return " ".join(self.encode_number(value))

    def ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [self._id_to_token[idx] for idx in token_ids if idx != self.blank_id]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(self.ids_to_tokens(token_ids))

    def decode_tokens(self, tokens: list[str]) -> str:
        return str(self.tokens_to_number(tokens))

    def decode_to_number(self, token_ids: list[int]) -> str:
        return str(self.tokens_to_number(self.ids_to_tokens(token_ids)))

    def ctc_collapse_tokens(self, frame_ids: list[int]) -> list[str]:
        return self.ids_to_tokens(_collapse_ctc_ids(frame_ids, self.blank_id))

    def ctc_collapse_words_raw(self, frame_ids: list[int]) -> str:
        return " ".join(self.ctc_collapse_tokens(frame_ids))

    def ctc_collapse_words(self, frame_ids: list[int]) -> str:
        return " ".join(self.repair_tokens(self.ctc_collapse_tokens(frame_ids)))

    def ctc_collapse(self, frame_ids: list[int]) -> str:
        tokens = self.repair_tokens(self.ctc_collapse_tokens(frame_ids))
        if not tokens:
            return ""
        return str(self.tokens_to_number(tokens))

    def repair_tokens(self, tokens: list[str]) -> list[str]:
        try:
            number = self.tokens_to_number_relaxed(tokens)
        except ValueError:
            return []
        return self.encode_number(number)

    def tokens_to_number(self, tokens: list[str]) -> int:
        if not tokens:
            raise ValueError("Empty token sequence.")

        thousand_positions = [idx for idx, token in enumerate(tokens) if token == "тысяча"]
        if len(thousand_positions) > 1:
            raise ValueError("Multiple thousand markers are not allowed.")

        if not thousand_positions:
            return self._parse_triplet(tokens)

        split_idx = thousand_positions[0]
        thousands_value = self._parse_triplet(tokens[:split_idx])
        if thousands_value <= 0:
            raise ValueError("Thousands part must be positive.")
        remainder_value = self._parse_triplet(tokens[split_idx + 1 :])
        return thousands_value * 1000 + remainder_value

    def tokens_to_number_relaxed(self, tokens: list[str]) -> int:
        return self.tokens_to_number(tokens)

    def _triplet_to_tokens(self, value: int) -> list[str]:
        hundreds, remainder = divmod(value, 100)
        tokens: list[str] = []

        if hundreds:
            tokens.append(str(hundreds * 100))

        if remainder == 0:
            return tokens
        if remainder <= 9:
            tokens.append(str(remainder))
            return tokens
        if remainder == 10:
            tokens.append("10")
            return tokens
        if 11 <= remainder <= 19:
            tokens.append(str(remainder - 10))
            tokens.append("дцать")
            return tokens

        tens = (remainder // 10) * 10
        units = remainder % 10
        tokens.append(str(tens))
        if units:
            tokens.append(str(units))
        return tokens

    def _parse_triplet(self, tokens: list[str]) -> int:
        if not tokens:
            return 0

        idx = 0
        value = 0
        n_tokens = len(tokens)

        if idx < n_tokens and tokens[idx] in COMPACT_HUNDREDS:
            value += COMPACT_HUNDREDS[tokens[idx]]
            idx += 1

        remaining = n_tokens - idx
        if remaining == 0:
            return value

        if remaining == 1:
            token = tokens[idx]
            if token == "10":
                return value + 10
            if token in COMPACT_TENS:
                return value + COMPACT_TENS[token]
            if token in COMPACT_UNITS:
                return value + COMPACT_UNITS[token]
            raise ValueError(f"Invalid token in triplet tail: {token!r}")

        if remaining == 2:
            first = tokens[idx]
            second = tokens[idx + 1]
            if first in COMPACT_UNITS and second == "дцать":
                return value + 10 + COMPACT_UNITS[first]
            if first in COMPACT_TENS and second in COMPACT_UNITS:
                return value + COMPACT_TENS[first] + COMPACT_UNITS[second]
            raise ValueError(f"Invalid 2-token tail: {first!r} {second!r}")

        raise ValueError("Too many tokens for one triplet.")
