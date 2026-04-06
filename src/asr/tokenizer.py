from __future__ import annotations

from dataclasses import dataclass


VOCAB = [
    "один",
    "два",
    "три",
    "четыре",
    "пять",
    "шесть",
    "семь",
    "восемь",
    "девять",
    "одна",
    "две",
    "десять",
    "одиннадцать",
    "двенадцать",
    "тринадцать",
    "четырнадцать",
    "пятнадцать",
    "шестнадцать",
    "семнадцать",
    "восемнадцать",
    "девятнадцать",
    "двадцать",
    "тридцать",
    "сорок",
    "пятьдесят",
    "шестьдесят",
    "семьдесят",
    "восемьдесят",
    "девяносто",
    "сто",
    "двести",
    "триста",
    "четыреста",
    "пятьсот",
    "шестьсот",
    "семьсот",
    "восемьсот",
    "девятьсот",
    "тысяча",
    "тысячи",
    "тысяч",
]

UNITS = {
    "один": 1,
    "два": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
}
THOUSAND_UNITS = {1: "одна", 2: "две"}
PARSE_UNITS = {
    "один": 1,
    "одна": 1,
    "два": 2,
    "две": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
}
TEENS = {
    "десять": 10,
    "одиннадцать": 11,
    "двенадцать": 12,
    "тринадцать": 13,
    "четырнадцать": 14,
    "пятнадцать": 15,
    "шестнадцать": 16,
    "семнадцать": 17,
    "восемнадцать": 18,
    "девятнадцать": 19,
}
TENS = {
    "двадцать": 20,
    "тридцать": 30,
    "сорок": 40,
    "пятьдесят": 50,
    "шестьдесят": 60,
    "семьдесят": 70,
    "восемьдесят": 80,
    "девяносто": 90,
}
HUNDREDS = {
    "сто": 100,
    "двести": 200,
    "триста": 300,
    "четыреста": 400,
    "пятьсот": 500,
    "шестьсот": 600,
    "семьсот": 700,
    "восемьсот": 800,
    "девятьсот": 900,
}
THOUSAND_FORMS = {"тысяча", "тысячи", "тысяч"}

VALUE_TO_UNIT = {value: word for word, value in UNITS.items()}
VALUE_TO_TEEN = {value: word for word, value in TEENS.items()}
VALUE_TO_TEN = {value // 10: word for word, value in TENS.items()}
VALUE_TO_HUNDRED = {value // 100: word for word, value in HUNDREDS.items()}


@dataclass(frozen=True)
class RussianNumberTokenizer:
    blank_id: int = 0

    def __post_init__(self) -> None:
        token_to_id = {token: idx + 1 for idx, token in enumerate(VOCAB)}
        object.__setattr__(self, "_token_to_id", token_to_id)
        object.__setattr__(
            self, "_id_to_token", {idx: token for token, idx in token_to_id.items()}
        )

    @property
    def tokenizer_type(self) -> str:
        return "russian_number_words"

    @property
    def config(self) -> dict[str, object]:
        return {"type": self.tokenizer_type, "blank_id": self.blank_id}

    @property
    def vocab_size(self) -> int:
        return len(VOCAB) + 1

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
            tokens.extend(self._triplet_to_tokens(thousands, thousands_part=True))
            tokens.append(self._thousand_word(thousands))
        if remainder:
            tokens.extend(self._triplet_to_tokens(remainder, thousands_part=False))
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

        split_idx = self._find_thousand_split(tokens)
        if split_idx is None:
            return self._parse_triplet(tokens)

        thousands_tokens = tokens[:split_idx]
        thousands_word = tokens[split_idx]
        remainder_tokens = tokens[split_idx + 1 :]

        thousands_value = self._parse_triplet(thousands_tokens)
        if thousands_value <= 0:
            raise ValueError("Thousands part must be positive.")
        if thousands_word != self._thousand_word(thousands_value):
            raise ValueError("Wrong form of 'тысяча'.")

        return thousands_value * 1000 + self._parse_triplet(remainder_tokens)

    def tokens_to_number_relaxed(self, tokens: list[str]) -> int:
        if not tokens:
            raise ValueError("Empty token sequence.")

        best_value, best_score = self._parse_triplet_relaxed(tokens)
        if best_value <= 0:
            best_value, best_score = None, -1  # type: ignore[assignment]

        for idx, token in enumerate(tokens):
            if token not in THOUSAND_FORMS:
                continue

            thousands_value, thousands_score = self._parse_triplet_relaxed(tokens[:idx])
            remainder_value, remainder_score = self._parse_triplet_relaxed(
                tokens[idx + 1 :]
            )

            if thousands_value == 0 and idx == 0 and token == "тысяча":
                thousands_value = 1

            if thousands_value <= 0:
                continue

            candidate_value = thousands_value * 1000 + remainder_value
            candidate_score = thousands_score + remainder_score + 1
            if candidate_score > best_score:
                best_value = candidate_value
                best_score = candidate_score

        if best_value is None or best_value <= 0:
            raise ValueError("Could not parse token sequence.")
        return best_value

    def _find_thousand_split(self, tokens: list[str]) -> int | None:
        positions = [idx for idx, token in enumerate(tokens) if token in THOUSAND_FORMS]
        if len(positions) > 1:
            raise ValueError("Multiple thousand markers are not allowed.")
        return positions[0] if positions else None

    def _triplet_to_tokens(self, value: int, *, thousands_part: bool) -> list[str]:
        hundreds, remainder = divmod(value, 100)
        tens, units = divmod(remainder, 10)
        tokens: list[str] = []

        if hundreds:
            tokens.append(VALUE_TO_HUNDRED[hundreds])
        if 10 <= remainder <= 19:
            tokens.append(VALUE_TO_TEEN[remainder])
            return tokens
        if tens >= 2:
            tokens.append(VALUE_TO_TEN[tens])
        if units:
            if thousands_part and units in THOUSAND_UNITS:
                tokens.append(THOUSAND_UNITS[units])
            else:
                tokens.append(VALUE_TO_UNIT[units])
        return tokens

    def _parse_triplet(self, tokens: list[str]) -> int:
        if not tokens:
            return 0

        value = 0
        seen_hundreds = False
        seen_tens = False
        seen_units = False

        for token in tokens:
            if token in HUNDREDS:
                if seen_hundreds or seen_tens or seen_units:
                    raise ValueError("Hundreds must go first.")
                value += HUNDREDS[token]
                seen_hundreds = True
                continue

            if token in TEENS:
                if seen_tens or seen_units:
                    raise ValueError("Teen token must replace tens and units.")
                value += TEENS[token]
                seen_tens = True
                seen_units = True
                continue

            if token in TENS:
                if seen_tens or seen_units:
                    raise ValueError("Tens can appear only once before units.")
                value += TENS[token]
                seen_tens = True
                continue

            if token in PARSE_UNITS:
                if seen_units:
                    raise ValueError("Units can appear only once.")
                value += PARSE_UNITS[token]
                seen_units = True
                continue

            raise ValueError(f"Unknown token: {token}")

        return value

    def _parse_triplet_relaxed(self, tokens: list[str]) -> tuple[int, int]:
        if not tokens:
            return 0, 0

        value = 0
        consumed = 0
        stage = "start"

        for token in tokens:
            if token in HUNDREDS:
                if stage == "start":
                    value += HUNDREDS[token]
                    consumed += 1
                    stage = "after_hundreds"
                continue

            if token in TEENS:
                if stage in {"start", "after_hundreds"}:
                    value += TEENS[token]
                    consumed += 1
                    stage = "done"
                continue

            if token in TENS:
                if stage in {"start", "after_hundreds"}:
                    value += TENS[token]
                    consumed += 1
                    stage = "after_tens"
                continue

            if token in PARSE_UNITS:
                if stage in {"start", "after_hundreds", "after_tens"}:
                    value += PARSE_UNITS[token]
                    consumed += 1
                    stage = "done"

        return value, consumed

    def _thousand_word(self, value: int) -> str:
        last_two = value % 100
        last_one = value % 10
        if 11 <= last_two <= 14:
            return "тысяч"
        if last_one == 1:
            return "тысяча"
        if last_one in (2, 3, 4):
            return "тысячи"
        return "тысяч"


def build_tokenizer(config: dict[str, object] | None = None) -> RussianNumberTokenizer:
    tokenizer_type = str((config or {}).get("type", "russian_number_words"))
    blank_id = int((config or {}).get("blank_id", 0))
    if tokenizer_type != "russian_number_words":
        raise ValueError(
            f"Unsupported tokenizer type in checkpoint: {tokenizer_type!r}"
        )
    return RussianNumberTokenizer(blank_id=blank_id)


def _collapse_ctc_ids(frame_ids: list[int], blank_id: int) -> list[int]:
    collapsed: list[int] = []
    prev = blank_id
    for idx in frame_ids:
        if idx != blank_id and idx != prev:
            collapsed.append(idx)
        prev = idx
    return collapsed
