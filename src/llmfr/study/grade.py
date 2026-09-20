"""Honest last-whole-number grading. No LLM judge."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

Grade = Literal["correct", "wrong", "no_answer"]
PairKind = Literal["sampling", "decoding_config"]
PairOutcome = Literal["agree_correct", "agree_wrong", "disagree", "ungraded"]

# Last number in the completion, GSM8K-style. A match is a whole number iff
# float(match).is_integer(). No LLM judge. Prompt numerals can leak into a
# truncated completion; that is a documented limit, not a hidden rewrite.
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

GRADING_RULE = (
    "Extract the last whole number in output_text. A token matching "
    r"-?\d+(?:\.\d+)? is a whole number iff float(token).is_integer(). "
    "Grade correct if that integer equals gold, wrong if it differs, "
    "no_answer if none is found. No LLM judge."
)

# Descriptive, not a p-value. Positive only when both classes have at least
# 4 gradeable diverged pairs and one disagree rate is at least twice the other
# (or the lower rate is 0 and the higher rate is > 0). Otherwise null.
MIN_GRADEABLE_FOR_VERDICT = 4
RATE_RATIO_FOR_POSITIVE = 2.0


def extract_last_whole_number(text: str) -> int | None:
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    value = float(matches[-1])
    if not value.is_integer():
        return None
    return int(value)


def grade_output(text: str, gold: int) -> Grade:
    extracted = extract_last_whole_number(text)
    if extracted is None:
        return "no_answer"
    return "correct" if extracted == gold else "wrong"


def pair_outcome(grade_a: Grade, grade_b: Grade) -> PairOutcome:
    if grade_a == "no_answer" or grade_b == "no_answer":
        return "ungraded"
    if grade_a == "correct" and grade_b == "correct":
        return "agree_correct"
    if grade_a == "wrong" and grade_b == "wrong":
        return "agree_wrong"
    return "disagree"


def _empty_class_row() -> dict[str, Any]:
    return {
        "n_pairs": 0,
        "n_gradeable_diverged": 0,
        "agree_correct": 0,
        "agree_wrong": 0,
        "disagree": 0,
        "ungraded": 0,
        "n_gradeable_traces": 0,
        "n_correct_traces": 0,
        "n_wrong_traces": 0,
        "disagree_rate": None,
        "wrong_answer_rate": None,
    }


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def summarize_pairs(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_class = {
        "sampling": _empty_class_row(),
        "decoding config": _empty_class_row(),
    }
    other_classes: dict[str, int] = {}
    n_identical = 0
    n_pairs = len(pairs)
    for pair in pairs:
        observed = pair.get("observed_class")
        if observed is None:
            n_identical += 1
            continue
        row = by_class.get(str(observed))
        if row is None:
            other_classes[str(observed)] = other_classes.get(str(observed), 0) + 1
            continue
        row["n_pairs"] += 1
        outcome = str(pair["outcome"])
        if outcome in ("agree_correct", "agree_wrong", "disagree", "ungraded"):
            row[outcome] += 1
        if outcome in ("agree_correct", "agree_wrong", "disagree"):
            row["n_gradeable_diverged"] += 1
        for grade in (pair["grade_a"], pair["grade_b"]):
            if grade == "correct":
                row["n_gradeable_traces"] += 1
                row["n_correct_traces"] += 1
            elif grade == "wrong":
                row["n_gradeable_traces"] += 1
                row["n_wrong_traces"] += 1
    for row in by_class.values():
        row["disagree_rate"] = _rate(row["disagree"], row["n_gradeable_diverged"])
        row["wrong_answer_rate"] = _rate(row["n_wrong_traces"], row["n_gradeable_traces"])
    return {
        "n_pairs": n_pairs,
        "n_identical": n_identical,
        "n_other_class": sum(other_classes.values()),
        "other_classes": other_classes,
        "by_class": by_class,
        "verdict": _verdict(by_class),
    }


def _verdict(by_class: Mapping[str, Mapping[str, Any]]) -> str:
    sampling = by_class["sampling"]
    decoding = by_class["decoding config"]
    rate_s = sampling["disagree_rate"]
    rate_d = decoding["disagree_rate"]
    n_s = int(sampling["n_gradeable_diverged"])
    n_d = int(decoding["n_gradeable_diverged"])
    if rate_s is None or rate_d is None:
        return "null"
    if n_s < MIN_GRADEABLE_FOR_VERDICT or n_d < MIN_GRADEABLE_FOR_VERDICT:
        return "null"
    hi, lo = (rate_s, rate_d) if rate_s >= rate_d else (rate_d, rate_s)
    if lo == 0.0:
        return "positive" if hi > 0.0 else "null"
    if hi / lo >= RATE_RATIO_FOR_POSITIVE:
        return "positive"
    return "null"
