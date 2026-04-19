from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Mapping, Sequence


def edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    prev = list(range(len(right) + 1))
    for i, ch_left in enumerate(left, start=1):
        curr = [i]
        for j, ch_right in enumerate(right, start=1):
            cost = 0 if ch_left == ch_right else 1
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = curr
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(reference, hypothesis) / len(reference)


@dataclass(frozen=True)
class DomainCerSummary:
    in_domain_cer: float | None
    out_of_domain_cer: float | None
    harmonic_mean_cer: float | None
    in_domain_count: int
    out_of_domain_count: int


def harmonic_mean(left: float, right: float) -> float:
    if left < 0.0 or right < 0.0:
        raise ValueError("CER values must be non-negative")
    if left == 0.0 and right == 0.0:
        return 0.0
    return (2.0 * left * right) / (left + right)


def compute_domain_cer_summary(
    speaker_cers: Mapping[str, Sequence[float]],
    in_domain_speakers: Collection[str],
) -> DomainCerSummary:
    in_domain = {str(speaker) for speaker in in_domain_speakers}
    in_sum = 0.0
    in_count = 0
    out_sum = 0.0
    out_count = 0

    for speaker, values in speaker_cers.items():
        if not values:
            continue
        if str(speaker) in in_domain:
            in_sum += sum(values)
            in_count += len(values)
        else:
            out_sum += sum(values)
            out_count += len(values)

    in_cer = (in_sum / in_count) if in_count else None
    out_cer = (out_sum / out_count) if out_count else None
    hmean = None
    if in_cer is not None and out_cer is not None:
        hmean = harmonic_mean(in_cer, out_cer)

    return DomainCerSummary(
        in_domain_cer=in_cer,
        out_of_domain_cer=out_cer,
        harmonic_mean_cer=hmean,
        in_domain_count=in_count,
        out_of_domain_count=out_count,
    )
