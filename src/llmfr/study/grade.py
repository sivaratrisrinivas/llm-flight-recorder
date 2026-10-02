"""Answer grading for ``llmfr study``. Code only, no LLM judge.

The default rule (``FINAL_ANSWER_RULE``) reads the answer the model committed
to, not the last number in the text. It was built from error analysis of the
120 real Qwen2.5-0.5B traces under ``examples/findings/gs-t22q`` (see
``evals/README.md``). The old last-number rule is kept as
``extract_last_whole_number`` for reproducing GS-T22n and GS-T22q.
"""

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

LAST_NUMBER_RULE = (
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


def grade_output_last_number(text: str, gold: int) -> Grade:
    """Legacy GS-T22n/GS-T22q rule. Over-counts wrong answers on truncated text."""
    extracted = extract_last_whole_number(text)
    if extracted is None:
        return "no_answer"
    return "correct" if extracted == gold else "wrong"


FINAL_ANSWER_RULE = (
    "Cut output_text at the first end-of-sequence marker (<|endoftext|>, <|im_end|>, "
    "</s>, <|eot_id|>) or a new 'Human:'/'User:' turn. Take the last explicit answer "
    "statement ('answer is N', 'final answer: N', 'final number is N', \\boxed{N}). "
    "If there is none, take the last completed equation result ('A op B = N', "
    "'equals N', 'A op B is N') where N is not followed by another operator. "
    "When the run hit max_new_tokens, a number that ends exactly at the end of the "
    "text is ignored because it may be cut mid-number. A reply that is only a number "
    "counts when the run ended on its own. No candidate means no_answer. "
    "No LLM judge."
)

_EOS_MARKERS = ("<|endoftext|>", "<|im_end|>", "</s>", "<|eot_id|>", "<|end|>")
_TURN_RE = re.compile(r"\n\s*(?:Human|User)\s*:")
_NUM = r"-?\d[\d,]*(?:\.\d+)?"
_EXPLICIT_RE = re.compile(
    r"(?:final\s+answer|the\s+answer|answer|final\s+(?:number|result))"
    r"\s*(?:is|:|=|would\s+be|will\s+be)?\s*[:=]?\s*(?:\\\(|\\\[|\$|\*\*)?\s*"
    r"(?:\\boxed\{)?\s*(" + _NUM + r")"
    r"|\\boxed\{\s*(" + _NUM + r")",
    re.IGNORECASE,
)
_OP = r"(?:\+|-|−|×|x|\*|/|÷|\\times|\\div|\\cdot|plus|minus|times|divided\s+by|multiplied\s+by)"
_EQUATION_RE = re.compile(
    r"\d+(?:\.\d+)?\s*" + _OP + r"\s*\d+(?:\.\d+)?(?:\s*" + _OP + r"\s*\d+(?:\.\d+)?)*"
    r"\s*(?:\\?\)?\s*)(?:=|equals|is|gives)\s*(?:\\\(|\$)?\s*(" + _NUM + r")",
    re.IGNORECASE,
)
_EQUALS_RE = re.compile(
    r"\b(?:equals|is\s+equal\s+to)\s*(?:\\\(|\$)?\s*(" + _NUM + r")",
    re.IGNORECASE,
)
_BARE_RE = re.compile(r"^\s*(" + _NUM + r")\s*[.!]?\s*$")
_TRAILING_OP_RE = re.compile(r"\s*(?:\+|-|−|×|\*|/|÷|\\times|\\div|\\cdot)\s*\d")


def strip_after_eos(text: str) -> str:
    """Drop anything the model wrote after its own end-of-sequence or a new turn."""
    cut = len(text)
    for marker in _EOS_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    match = _TURN_RE.search(text)
    if match is not None:
        cut = min(cut, match.start())
    return text[:cut]


def _to_int(token: str) -> int | None:
    try:
        value = float(token.replace(",", ""))
    except ValueError:
        return None
    if not value.is_integer():
        return None
    return int(value)


def _candidates(regex: re.Pattern[str], text: str, *, truncated: bool) -> list[tuple[int, int]]:
    found: list[tuple[int, int]] = []
    for match in regex.finditer(text):
        group = next((g for g in range(1, (regex.groups or 0) + 1) if match.group(g)), None)
        if group is None:
            continue
        end = match.end(group)
        if truncated and text[end:].strip() == "":
            continue
        if _TRAILING_OP_RE.match(text, end):
            continue
        value = _to_int(match.group(group))
        if value is not None:
            found.append((match.start(group), value))
    return found


def extract_final_answer(text: str, *, truncated: bool = False) -> int | None:
    """Return the integer the model committed to, or None. See FINAL_ANSWER_RULE."""
    cut = strip_after_eos(text)
    hit_eos = len(cut) < len(text)
    effective_truncated = truncated and not hit_eos
    explicit = _candidates(_EXPLICIT_RE, cut, truncated=effective_truncated)
    if explicit:
        return explicit[-1][1]
    implicit = _candidates(_EQUATION_RE, cut, truncated=effective_truncated) + _candidates(
        _EQUALS_RE, cut, truncated=effective_truncated
    )
    if implicit:
        return max(implicit)[1]
    bare = _BARE_RE.match(cut)
    if bare is not None and not effective_truncated:
        return _to_int(bare.group(1))
    return None


def grade_output(text: str, gold: int, *, truncated: bool = False) -> Grade:
    """Grade with FINAL_ANSWER_RULE. ``truncated`` means the run hit max_new_tokens."""
    extracted = extract_final_answer(text, truncated=truncated)
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
    """Count sampling vs decoding-config pairs. No verdict field.

    ``observed_class is None`` means no first event divergence. That is not
    ``compare.identical``: seed or temperature can still differ. Ungraded
    pairs (any ``no_answer`` side) are excluded from disagree_rate and
    wrong_answer_rate.
    """
    by_class = {
        "sampling": _empty_class_row(),
        "decoding config": _empty_class_row(),
    }
    other_classes: dict[str, int] = {}
    n_no_first_divergence = 0
    n_pairs = len(pairs)
    for pair in pairs:
        observed = pair.get("observed_class")
        if observed is None:
            n_no_first_divergence += 1
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
        "n_no_first_divergence": n_no_first_divergence,
        "n_other_class": sum(other_classes.values()),
        "other_classes": other_classes,
        "by_class": by_class,
    }


def descriptive_verdict(by_class: Mapping[str, Mapping[str, Any]]) -> str:
    """GS-T22n descriptive label only. Not a p-value. Not part of ``llmfr study``."""
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


# Rule used by ``llmfr study`` today.
GRADING_RULE = FINAL_ANSWER_RULE
