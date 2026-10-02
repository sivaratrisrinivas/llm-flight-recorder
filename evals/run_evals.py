"""Run every llmfr eval that needs no model download and gate on a baseline.

1. Answer grader vs hand labels on 120 real GS-T22q traces (dev set; the rule
   was built while reading these, so this number is optimistic).
2. Answer grader vs hand labels on held-out real traces recorded after the rule
   was frozen (test set), when evals/data/heldout_labels.jsonl exists.
3. Answer grader on the synthetic regression suite (evals/data/answer_cases.jsonl).
4. Divergence classification on generated record+compare pairs.

Old last-number grader numbers are printed next to the new ones.

python evals/run_evals.py            # print and gate
python evals/run_evals.py --write    # also refresh evals/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from llmfr.study.grade import extract_final_answer, extract_last_whole_number

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
sys.path.insert(0, str(HERE))

import divergence_eval  # noqa: E402


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _grade(answer: int | None, gold: int) -> str:
    if answer is None:
        return "no_answer"
    return "correct" if answer == gold else "wrong"


def score_labeled(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare grader output with the hand label for each trace.

    Positive class = "the trace contains a committed answer".
    TPR: of traces with a labeled answer, share where the grader found that exact answer.
    TNR: of traces with no labeled answer, share where the grader returned no answer.
    """
    out: dict[str, Any] = {"n": len(rows)}
    for name, fn in (
        ("final_answer", lambda r: extract_final_answer(r["output"], truncated=r["truncated"])),
        ("last_number", lambda r: extract_last_whole_number(r["output"])),
    ):
        pos = [r for r in rows if r["label_answer"] is not None]
        neg = [r for r in rows if r["label_answer"] is None]
        tp = sum(fn(r) == r["label_answer"] for r in pos)
        tn = sum(fn(r) is None for r in neg)
        grade_match = sum(
            _grade(fn(r), r["gold"]) == _grade(r["label_answer"], r["gold"]) for r in rows
        )
        labeled_wrong = sum(_grade(r["label_answer"], r["gold"]) == "wrong" for r in rows)
        grader_wrong = sum(_grade(fn(r), r["gold"]) == "wrong" for r in rows)
        out[name] = {
            "tpr": round(tp / len(pos), 4) if pos else None,
            "tnr": round(tn / len(neg), 4) if neg else None,
            "n_pos": len(pos),
            "n_neg": len(neg),
            "grade_agreement": round(grade_match / len(rows), 4),
            "wrong_by_label": labeled_wrong,
            "wrong_by_grader": grader_wrong,
        }
    return out


def score_synthetic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    new = [extract_final_answer(r["text"], truncated=r["truncated"]) == r["expected"] for r in rows]
    old = [extract_last_whole_number(r["text"]) == r["expected"] for r in rows]
    failures = [r["id"] for r, ok in zip(rows, new, strict=True) if not ok]
    return {
        "n": len(rows),
        "final_answer_pass": sum(new),
        "last_number_pass": sum(old),
        "failures": failures,
    }


def score_divergence() -> dict[str, Any]:
    rows = [divergence_eval.run_case(c) for c in divergence_eval.build_cases()]
    return {
        "n": len(rows),
        "n_pass": sum(r["pass"] for r in rows),
        "failures": [r["id"] for r in rows if not r["pass"]],
    }


def _print_labeled(title: str, res: dict[str, Any]) -> None:
    print(f"{title} (n={res['n']})")
    for name in ("final_answer", "last_number"):
        r = res[name]
        print(
            f"  {name:12s} TPR {r['tpr']}  TNR {r['tnr']}  grade agreement {r['grade_agreement']}"
            f"  wrong answers: grader {r['wrong_by_grader']} vs label {r['wrong_by_label']}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    results: dict[str, Any] = {}
    results["dev_real"] = score_labeled(_load(DATA / "dev_labels.jsonl"))
    _print_labeled("answer grader, real dev traces", results["dev_real"])
    heldout = DATA / "heldout_labels.jsonl"
    if heldout.exists():
        results["test_real"] = score_labeled(_load(heldout))
        _print_labeled("answer grader, real held-out traces", results["test_real"])
    results["synthetic"] = score_synthetic(_load(DATA / "answer_cases.jsonl"))
    s = results["synthetic"]
    print(
        f"answer grader, synthetic regression suite: {s['final_answer_pass']}/{s['n']} "
        f"(old last-number rule {s['last_number_pass']}/{s['n']})"
    )
    results["divergence"] = score_divergence()
    d = results["divergence"]
    print(f"divergence classification: {d['n_pass']}/{d['n']} name the injected cause")
    for fid in d["failures"]:
        print(f"  known miss: {fid}")

    baseline = json.loads((HERE / "baseline.json").read_text())
    problems = []
    if s["final_answer_pass"] < baseline["synthetic_min_pass"]:
        problems.append(f"synthetic {s['final_answer_pass']} < {baseline['synthetic_min_pass']}")
    if d["n_pass"] < baseline["divergence_min_pass"]:
        problems.append(f"divergence {d['n_pass']} < {baseline['divergence_min_pass']}")
    for split in ("dev_real", "test_real"):
        if split in results and split in baseline:
            got = results[split]["final_answer"]
            for key in ("tpr", "tnr"):
                if got[key] is not None and got[key] < baseline[split][key]:
                    problems.append(f"{split} {key} {got[key]} < {baseline[split][key]}")
    if args.write:
        (HERE / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    if problems:
        print("REGRESSION: " + "; ".join(problems))
        return 1
    print("all eval gates pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
