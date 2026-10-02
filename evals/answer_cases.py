"""Synthetic regression cases for the study answer grader.

Each case is a completion built from dimensions seen in the real GS-T22q
traces (see evals/README.md, "Failure modes"). The expected answer comes from
how the case was built, not from running the grader.

Dimensions:
  preamble   : none | to_solve | steps | numbered | latex
  commit     : answer_is | final_answer | final_number | boxed | equation |
               equals | is_phrase | bare | none
  ending     : natural | eos_then_bleed | truncated_after | truncated_mid_number |
               truncated_before
  distractor : none | prompt_numerals | step_labels | decimal_form | bled_problem
  problem    : add | sub | mul | div (operands vary)

Run: python evals/answer_cases.py --write evals/data/answer_cases.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from typing import Any

OPS = {
    "add": ("+", "plus", lambda a, b: a + b),
    "sub": ("-", "minus", lambda a, b: a - b),
    "mul": ("\\times", "times", lambda a, b: a * b),
    "div": ("\\div", "divided by", lambda a, b: a // b),
}


def _operands(op: str, rng: random.Random) -> tuple[int, int]:
    if op == "div":
        b = rng.randint(2, 9)
        return b * rng.randint(2, 15), b
    if op == "sub":
        a = rng.randint(12, 90)
        return a, rng.randint(2, a - 3)
    return rng.randint(3, 40), rng.randint(3, 12)


def _preamble(kind: str, a: int, b: int, word: str) -> str:
    return {
        "none": "",
        "to_solve": f"To solve {a} {word} {b}, we work it out. ",
        "steps": f"Step 1: Start with {a}.\nStep 2: Use {b}.\nStep 3: Compute.\n",
        "numbered": f"1. Start with the first number: {a}\n2. Take the second number: {b}\n",
        "latex": f"We need \\( {a} {word} {b} \\).\n\n",
    }[kind]


def _commit(kind: str, a: int, sym: str, word: str, b: int, n: str) -> str:
    return {
        "answer_is": f"The answer is {n}.",
        "final_answer": f"Final answer: {n}",
        "final_number": f"So, the final number is {n}.",
        "boxed": f"So the result is \\(\\boxed{{{n}}}\\)",
        "equation": f"\\[ {a} {sym} {b} = {n} \\]",
        "equals": f"So {a} {word} {b} equals {n}.",
        "is_phrase": f"Therefore {a} {word} {b} is {n}.",
        "bare": n,
        "none": "",
    }[kind]


def build_cases(seed: int = 20261002) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    preambles = ("none", "to_solve", "steps", "numbered", "latex")
    commits = (
        "answer_is",
        "final_answer",
        "final_number",
        "boxed",
        "equation",
        "equals",
        "is_phrase",
        "bare",
        "none",
    )
    endings = (
        "natural",
        "eos_then_bleed",
        "truncated_after",
        "truncated_mid_number",
        "truncated_before",
    )
    distractors = ("none", "prompt_numerals", "step_labels", "decimal_form", "bled_problem")
    cases: list[dict[str, Any]] = []
    for pre, commit, ending, distractor in itertools.product(
        preambles, commits, endings, distractors
    ):
        # Keep combos that make sense and keep the suite near 400 cases.
        if commit == "bare" and pre != "none":
            continue
        if commit == "none" and ending in ("truncated_mid_number", "truncated_after"):
            continue
        if distractor == "decimal_form" and commit in ("bare", "none", "boxed"):
            continue
        if rng.random() > 0.55:
            continue
        op = rng.choice(list(OPS))
        sym, word, fn = OPS[op]
        a, b = _operands(op, rng)
        gold = fn(a, b)
        n = f"{gold}.0" if distractor == "decimal_form" else str(gold)
        text = _preamble(pre, a, b, word)
        if distractor == "step_labels":
            text += "Step 4: Check the work.\n"
        if distractor == "prompt_numerals":
            text += f"We were given {a} and {b}. "
        expected: int | None = gold
        truncated = False
        body = _commit(commit, a, sym, word, b, n)
        if commit == "none":
            expected = None
        if ending == "natural":
            text += body
            if distractor == "bled_problem" and commit not in ("none",):
                # Model keeps going into a made-up problem without an EOS token.
                text += f" There are {b} x 3 = {b * 3} more items in the next box."
                if commit in ("equation", "equals", "is_phrase", "bare"):
                    # No explicit marker, so a later equation legitimately wins.
                    expected = b * 3
        elif ending == "eos_then_bleed":
            text += body + f"<|endoftext|>Human: What is {a + 7} times {b + 2}?"
            truncated = True
        elif ending == "truncated_after":
            text += body + "\n\nStep 5: Verify by checking"
            truncated = True
        elif ending == "truncated_mid_number":
            # The answer is cut before its last digit or right at the end.
            text += body.rstrip(".\\] )}")
            if len(str(gold)) > 1:
                cut = text.rfind(str(gold))
                text = text[: cut + 1]
            truncated = True
            expected = None
        elif ending == "truncated_before":
            text += f"\\[ {a} {sym} {b} ="
            truncated = True
            expected = None
        if commit == "bare" and (ending == "truncated_after" or distractor == "prompt_numerals"):
            # A lone number after or before other text is not a reply that is
            # only a number, so it is not a commitment.
            expected = None
        cases.append(
            {
                "id": f"{pre}|{commit}|{ending}|{distractor}|{op}",
                "dims": {
                    "preamble": pre,
                    "commit": commit,
                    "ending": ending,
                    "distractor": distractor,
                    "op": op,
                },
                "text": text,
                "truncated": truncated,
                "expected": expected,
            }
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", required=True)
    args = parser.parse_args()
    cases = build_cases()
    with open(args.write, "w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case) + "\n")
    print(f"wrote {len(cases)} cases to {args.write}")


if __name__ == "__main__":
    main()
