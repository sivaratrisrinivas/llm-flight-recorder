"""GS-T22n: does first-divergence class predict wrong answers?

Records paired traces on the portfolio Qwen2.5-0.5B-Instruct checkpoint
(sampling seed split vs decoding-config temperature split), grades each
completion with a last-whole-number rule, and counts whether the two
answers agree on correctness.

Finding numbers come from a `--backend hf` run you actually perform.
`--backend fake` is CI smoke only and must not overwrite `docs/findings`.
Recompute the table from checked-in JSON with `--from-results`. Do not
hand-edit rates into the finding.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from llmfr.adapters.huggingface import (
    PORTFOLIO_DEMO_MODEL_ID,
    PORTFOLIO_DEMO_MODEL_REVISION,
)
from llmfr.compare import compare_traces
from llmfr.core.schema import GenerationConfig, Trace, dumps_jsonl
from llmfr.record import record_generation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.fakes import FakeCausalLMAdapter  # noqa: E402

FINDING_DIR = ROOT / "docs" / "findings"
DEFAULT_RESULTS = FINDING_DIR / "gs-t22n-results.json"
DEFAULT_FINDING = FINDING_DIR / "gs-t22n.md"
DEFAULT_TRACES = ROOT / "examples" / "findings" / "gs-t22n"

MODEL_ID = PORTFOLIO_DEMO_MODEL_ID
MODEL_REVISION = PORTFOLIO_DEMO_MODEL_REVISION
MAX_NEW_TOKENS = 64
CAPTURE_K = 5
SAMPLING_SEEDS = (1, 2)
SAMPLING_TEMPERATURE = 1.0
DECODING_SEED = 1
DECODING_TEMPERATURES = (0.7, 1.2)
PROMPT_SUFFIX = " Think step by step, then give the final number."

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


@dataclass(frozen=True)
class Item:
    item_id: str
    question: str
    gold: int

    @property
    def prompt(self) -> str:
        return self.question + PROMPT_SUFFIX


ITEMS: tuple[Item, ...] = (
    Item(
        "sheep_trick",
        "A farmer has 17 sheep. All but 9 run away. How many sheep are left?",
        9,
    ),
    Item("seven_plus_five", "What is 7 plus 5?", 12),
    Item("twelve_minus_four", "What is 12 minus 4?", 8),
    Item("three_times_six", "What is 3 times 6?", 18),
    Item(
        "cows_sold",
        "A farmer has 10 cows. He sells 3. How many cows remain?",
        7,
    ),
    Item("hundred_div_four", "What is 100 divided by 4?", 25),
    Item(
        "apples_left",
        "There are 8 apples. You eat 2. How many apples are left?",
        6,
    ),
    Item("two_plus_two", "What is 2 plus 2?", 4),
)


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


def _docs_bound(path: Path) -> bool:
    resolved = path.resolve()
    return resolved == DEFAULT_RESULTS.resolve() or resolved == DEFAULT_FINDING.resolve()


def _flat_adapter() -> FakeCausalLMAdapter:
    return FakeCausalLMAdapter(
        prompt_ids=[1],
        logits=(1.0, 1.0, 1.0, 1.0),
        logits_for_prefix={(1,): (1.0, 1.0, 1.0, 1.0)},
        name="fake-lm",
    )


def _pair_kind_configs(kind: PairKind) -> tuple[GenerationConfig, GenerationConfig]:
    if kind == "sampling":
        return (
            GenerationConfig(
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=SAMPLING_TEMPERATURE,
                seed=SAMPLING_SEEDS[0],
            ),
            GenerationConfig(
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=SAMPLING_TEMPERATURE,
                seed=SAMPLING_SEEDS[1],
            ),
        )
    return (
        GenerationConfig(
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=DECODING_TEMPERATURES[0],
            seed=DECODING_SEED,
        ),
        GenerationConfig(
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=DECODING_TEMPERATURES[1],
            seed=DECODING_SEED,
        ),
    )


def _grade_pair(
    *,
    item: Item,
    kind: PairKind,
    trace_a: Trace,
    trace_b: Trace,
    path_a: str | None,
    path_b: str | None,
) -> dict[str, Any]:
    compared = compare_traces(trace_a, trace_b)
    first = compared.first_divergence
    grade_a = grade_output(trace_a.run_metadata.output_text, item.gold)
    grade_b = grade_output(trace_b.run_metadata.output_text, item.gold)
    extracted_a = extract_last_whole_number(trace_a.run_metadata.output_text)
    extracted_b = extract_last_whole_number(trace_b.run_metadata.output_text)
    return {
        "item_id": item.item_id,
        "gold": item.gold,
        "intended_kind": kind,
        "observed_class": None if first is None else first.classification,
        "first_step": None if first is None else first.step,
        "identical": compared.identical,
        "grade_a": grade_a,
        "grade_b": grade_b,
        "extracted_a": extracted_a,
        "extracted_b": extracted_b,
        "outcome": pair_outcome(grade_a, grade_b),
        "output_a": trace_a.run_metadata.output_text,
        "output_b": trace_b.run_metadata.output_text,
        "trace_a": str(trace_a.run_metadata.trace_id),
        "trace_b": str(trace_b.run_metadata.trace_id),
        "path_a": path_a,
        "path_b": path_b,
        "seed_a": trace_a.generation_config.seed,
        "seed_b": trace_b.generation_config.seed,
        "temperature_a": trace_a.generation_config.temperature,
        "temperature_b": trace_b.generation_config.temperature,
    }


def _repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _save_trace(out: Path, item_id: str, kind: PairKind, side: str, trace: Trace) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{item_id}_{kind}_{side}.jsonl"
    dest.write_text(dumps_jsonl(trace), encoding="utf-8")
    return dest


def run_experiment(
    *,
    adapter: Any,
    items: Sequence[Item],
    traces_dir: Path | None,
    source: str,
    save_traces: bool,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    kinds: tuple[PairKind, ...] = ("sampling", "decoding_config")
    total = len(items) * len(kinds)
    done = 0
    for item in items:
        for kind in kinds:
            done += 1
            gen_a, gen_b = _pair_kind_configs(kind)
            sys.stderr.write(f"recording {done}/{total} {item.item_id} {kind}\n")
            trace_a = record_generation(
                adapter,
                item.prompt,
                generation=gen_a,
                persist=False,
                capture_k=CAPTURE_K,
                source=source,
                tags={"gs_t22n_item": item.item_id, "gs_t22n_kind": kind, "side": "a"},
            )
            trace_b = record_generation(
                adapter,
                item.prompt,
                generation=gen_b,
                persist=False,
                capture_k=CAPTURE_K,
                source=source,
                tags={"gs_t22n_item": item.item_id, "gs_t22n_kind": kind, "side": "b"},
            )
            path_a = path_b = None
            if save_traces:
                assert traces_dir is not None
                path_a = _repo_rel(_save_trace(traces_dir, item.item_id, kind, "a", trace_a))
                path_b = _repo_rel(_save_trace(traces_dir, item.item_id, kind, "b", trace_b))
            pairs.append(
                _grade_pair(
                    item=item,
                    kind=kind,
                    trace_a=trace_a,
                    trace_b=trace_b,
                    path_a=path_a,
                    path_b=path_b,
                )
            )
    return pairs


def _hardware() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
    }


def build_results(
    *,
    backend: str,
    pairs: Sequence[Mapping[str, Any]],
    model_name: str,
    model_revision: str | None,
    library_versions: Mapping[str, str] | None,
    recorded_at: str,
) -> dict[str, Any]:
    summary = summarize_pairs(pairs)
    return {
        "finding_id": "GS-T22n",
        "recorded_at": recorded_at,
        "backend": backend,
        "model": model_name,
        "revision": model_revision,
        "n_items": len({pair["item_id"] for pair in pairs}),
        "n_pairs": len(pairs),
        "max_new_tokens": MAX_NEW_TOKENS,
        "capture_k": CAPTURE_K,
        "sampling_seeds": list(SAMPLING_SEEDS),
        "sampling_temperature": SAMPLING_TEMPERATURE,
        "decoding_seed": DECODING_SEED,
        "decoding_temperatures": list(DECODING_TEMPERATURES),
        "grading_rule": GRADING_RULE,
        "verdict_rule": (
            f"positive if both sampling and decoding-config disagree_rate are defined, "
            f"each n_gradeable_diverged >= {MIN_GRADEABLE_FOR_VERDICT}, and the higher "
            f"rate is at least {RATE_RATIO_FOR_POSITIVE}x the lower (or lower is 0 and "
            f"higher > 0); else null. Not a p-value."
        ),
        "hardware": _hardware(),
        "library_versions": dict(library_versions or {}),
        "items": [
            {"item_id": item.item_id, "question": item.question, "gold": item.gold}
            for item in ITEMS
            if any(pair["item_id"] == item.item_id for pair in pairs)
        ],
        "pairs": list(pairs),
        "summary": summary,
    }


def _fmt_rate(value: float | None) -> str:
    if value is None:
        return "n/a (denominator 0; not invented)"
    count_hint = f"{value:.4f}".rstrip("0").rstrip(".")
    return count_hint


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    path.write_text(text, encoding="utf-8")


def render_finding(results: Mapping[str, Any]) -> str:
    summary = results["summary"]
    by_class = summary["by_class"]
    sampling = by_class["sampling"]
    decoding = by_class["decoding config"]
    verdict = summary["verdict"]
    recorded_at = str(results["recorded_at"])
    rows = []
    for name, row in (("sampling", sampling), ("decoding config", decoding)):
        rows.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(row["n_pairs"]),
                    str(row["n_gradeable_diverged"]),
                    str(row["agree_correct"]),
                    str(row["agree_wrong"]),
                    str(row["disagree"]),
                    str(row["ungraded"]),
                    _fmt_rate(row["disagree_rate"]),
                    _fmt_rate(row["wrong_answer_rate"]),
                    f"{row['n_wrong_traces']}/{row['n_gradeable_traces']}",
                ]
            )
            + " |"
        )
    pair_lines = []
    for pair in results["pairs"]:
        observed = pair["observed_class"] if pair["observed_class"] else "identical"
        pair_lines.append(
            f"- `{pair['item_id']}` intended `{pair['intended_kind']}`: "
            f"class `{observed}` step={pair['first_step']} "
            f"grades {pair['grade_a']}/{pair['grade_b']} "
            f"(extracted {pair['extracted_a']}/{pair['extracted_b']}, "
            f"gold {pair['gold']}) outcome `{pair['outcome']}`"
        )
    other = summary["other_classes"] if summary["other_classes"] else "none"
    n_items = results["n_items"]
    n_pairs = results["n_pairs"]
    seeds = results["sampling_seeds"]
    dec_temps = results["decoding_temperatures"]
    table_header = (
        "| class | n pairs | n gradeable diverged | agree correct | "
        "agree wrong | disagree | ungraded | disagree rate | "
        "wrong-answer rate | wrong/gradeable traces |"
    )
    lines = [
        "# GS-T22n: Does first-divergence class predict wrong answers?",
        "",
        f"Status: measured ({verdict})",
        f"Date: {recorded_at}",
        "",
        "## Question",
        "",
        "When two recorded generations split, does the M5 first-divergence class",
        "(`sampling` vs `decoding config`) predict whether the completions disagree",
        "on correctness? This is a small descriptive count, like SunkeLo's GS-T7",
        "absent-claim / grounding number: report the measured rates, including a",
        "null, and do not invent a metric the run did not produce.",
        "",
        "## Setup",
        "",
        f"- **N items:** {n_items} gradeable integer questions ({n_pairs} pairs;",
        "  2 pair kinds per item).",
        f"- **Model:** `{results['model']}` revision `{results['revision']}`.",
        "- **Library default / CI smoke:** `sshleifer/tiny-gpt2` is not used",
        "  for this finding.",
        f"- **Sampling pairs:** seed {seeds[0]} vs {seeds[1]},",
        f"  temperature {results['sampling_temperature']}.",
        f"- **Decoding-config pairs:** seed {results['decoding_seed']},",
        f"  temperature {dec_temps[0]} vs {dec_temps[1]}.",
        f"- **max_new_tokens:** {results['max_new_tokens']};",
        f"  **capture_k:** {results['capture_k']}.",
        f"- **Grading rule:** {results['grading_rule']}",
        f"- **Verdict rule:** {results['verdict_rule']}",
        f"- **Hardware:** {results['hardware']['platform']}; "
        f"Python {results['hardware']['python_version']}; "
        f"cpus={results['hardware']['cpu_count']}.",
        f"- **Library versions:** {results['library_versions']}",
        f"- **Backend:** `{results['backend']}` (live recorder + `compare_traces`",
        "  on stored events; compare does not call a model).",
        "",
        "Re-run:",
        "",
        "```bash",
        "python scripts/gs_t22n_divergence_accuracy.py --backend hf",
        "```",
        "",
        "CI smoke (fake adapter; must not overwrite this finding):",
        "",
        "```bash",
        "python scripts/gs_t22n_divergence_accuracy.py --backend fake "
        "--out /tmp/llmfr-gs-t22n-fake",
        "```",
        "",
        "Raw JSON: `docs/findings/gs-t22n-results.json`.",
        "",
        "## Result",
        "",
        f"**{verdict}.** Among gradeable diverged pairs, sampling disagree rate is",
        f"{_fmt_rate(sampling['disagree_rate'])} "
        f"({sampling['disagree']}/{sampling['n_gradeable_diverged']});",
        "decoding-config disagree rate is",
        f"{_fmt_rate(decoding['disagree_rate'])} "
        f"({decoding['disagree']}/{decoding['n_gradeable_diverged']}).",
        "Wrong-answer rate among gradeable traces is",
        f"{_fmt_rate(sampling['wrong_answer_rate'])} for sampling pairs and",
        f"{_fmt_rate(decoding['wrong_answer_rate'])} for decoding-config pairs.",
        "",
        "## Table",
        "",
        table_header,
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        rows[0],
        rows[1],
        "",
        f"Identical pairs (no first behavioral divergence): {summary['n_identical']}.",
        f"Other observed classes: {other}.",
        "",
        "## Pair detail",
        "",
        *pair_lines,
        "",
        "## Limits",
        "",
        "- N is small on purpose. This is not a leaderboard and not a p-value.",
        f"- Completions are truncated at {results['max_new_tokens']} new tokens.",
        "  A restated prompt numeral can become the last whole number; that",
        "  grades `wrong` or accidentally `correct` without a hidden fix-up.",
        "- `no_answer` pairs are ungraded, not filled in.",
        "- 0.5B-class instruct model on CPU, no chat template (same raw-prompt",
        "  style as the portfolio demo).",
        "- First-divergence class is the stored-event classifier (ADR 0006).",
        "  Later token diffs are downstream and are not a second root cause",
        "  (ADR 0007).",
        "- tiny-gpt2 answers are unreadable; they are not this finding.",
        "- `--backend fake` is smoke. It must not be quoted as the table above.",
        "",
    ]
    return "\n".join(lines)


def write_source(out: Path, *, backend: str, model: str, revision: str | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    revision_line = "" if revision is None else f"revision={revision}\n"
    (out / "SOURCE.txt").write_text(
        f"backend={backend}\n"
        f"model={model}\n"
        f"{revision_line}"
        f"n_items={len(ITEMS)}\n"
        f"max_new_tokens={MAX_NEW_TOKENS}\n"
        f"sampling_seeds={SAMPLING_SEEDS[0]} {SAMPLING_SEEDS[1]}\n"
        f"sampling_temperature={SAMPLING_TEMPERATURE}\n"
        f"decoding_seed={DECODING_SEED} temperatures={DECODING_TEMPERATURES[0]} "
        f"{DECODING_TEMPERATURES[1]}\n"
        f"grading_rule={GRADING_RULE}\n",
        encoding="utf-8",
    )


def _load_results(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    pairs = payload.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError(f"{path} missing pairs")
    payload["summary"] = summarize_pairs(pairs)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("hf", "fake"),
        default=None,
        help="hf: live portfolio Qwen. fake: in-repo adapter (CI smoke).",
    )
    parser.add_argument(
        "--from-results",
        default=None,
        help="Recompute summary/table from a checked-in results JSON. No model.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Trace directory. Fake backend cannot use examples/findings/gs-t22n.",
    )
    parser.add_argument(
        "--results",
        default=None,
        help="Results JSON path. Fake backend cannot use docs/findings.",
    )
    parser.add_argument(
        "--finding",
        default=None,
        help="Finding markdown path. Fake backend cannot use docs/findings.",
    )
    parser.add_argument(
        "--limit-items",
        type=int,
        default=None,
        help="Run only the first N items (debug). Finding docs need the full N.",
    )
    parser.add_argument(
        "--write-docs",
        action="store_true",
        help="Write results JSON and finding markdown (hf or --from-results only).",
    )
    parser.add_argument(
        "--no-save-traces",
        action="store_true",
        help="Skip JSONL trace writes (still grade in memory).",
    )
    args = parser.parse_args(None if argv is None else list(argv))

    results_path = DEFAULT_RESULTS if args.results is None else Path(args.results)
    finding_path = DEFAULT_FINDING if args.finding is None else Path(args.finding)
    traces_dir = DEFAULT_TRACES if args.out is None else Path(args.out)

    if args.from_results:
        payload = _load_results(Path(args.from_results))
        text = render_finding(payload)
        sys.stdout.write(text)
        if args.write_docs:
            if args.backend == "fake":
                sys.stderr.write("error: refusing to write finding docs from fake backend\n")
                return 2
            _write_json(results_path, payload)
            finding_path.write_text(text, encoding="utf-8")
        return 0

    if args.backend is None:
        sys.stderr.write("error: pass --backend hf|fake or --from-results PATH\n")
        return 2

    items = ITEMS if args.limit_items is None else ITEMS[: args.limit_items]
    if not items:
        sys.stderr.write("error: no items to run\n")
        return 2

    if args.backend == "fake":
        if _docs_bound(results_path) or _docs_bound(finding_path):
            sys.stderr.write(
                "error: refusing to overwrite docs/findings with fake-adapter output; "
                "pass --results and --finding under a scratch directory\n"
            )
            return 2
        if traces_dir.resolve() == DEFAULT_TRACES.resolve():
            sys.stderr.write(
                "error: refusing to overwrite examples/findings/gs-t22n with fake "
                "adapter traces; pass --out DIR\n"
            )
            return 2
        adapter: Any = _flat_adapter()
        model_name = adapter.model_config.name
        model_revision = adapter.model_config.revision
        library_versions: dict[str, str] = {}
        source = "gs-t22n-fake"
        save_traces = not args.no_save_traces
    else:
        from llmfr.adapters.huggingface import HuggingFaceCausalLMAdapter

        adapter = HuggingFaceCausalLMAdapter(
            MODEL_ID,
            revision=MODEL_REVISION,
            device="cpu",
        )
        model_name = adapter.model_config.name
        model_revision = adapter.model_config.revision
        library_versions = dict(adapter.environment().library_versions)
        source = "gs-t22n"
        save_traces = not args.no_save_traces

    recorded_at = datetime.now(UTC).isoformat()
    pairs = run_experiment(
        adapter=adapter,
        items=items,
        traces_dir=traces_dir if save_traces else None,
        source=source,
        save_traces=save_traces,
    )
    payload = build_results(
        backend=args.backend,
        pairs=pairs,
        model_name=model_name,
        model_revision=model_revision,
        library_versions=library_versions,
        recorded_at=recorded_at,
    )
    if save_traces:
        write_source(traces_dir, backend=args.backend, model=model_name, revision=model_revision)

    text = render_finding(payload)
    sys.stdout.write(text)
    if args.write_docs:
        if args.backend == "fake":
            sys.stderr.write("error: refusing to write finding docs from fake backend\n")
            return 2
        if args.limit_items is not None and args.limit_items < len(ITEMS):
            sys.stderr.write("error: refusing to write finding docs from a truncated item list\n")
            return 2
        _write_json(results_path, payload)
        finding_path.write_text(text, encoding="utf-8")
    elif args.backend == "fake":
        _write_json(results_path, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
