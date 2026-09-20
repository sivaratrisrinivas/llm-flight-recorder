"""GS-T22q: pack `llmfr study` JSON + store traces into finding fixtures.

Rates come from `llmfr study --json` (recomputed with `summarize_pairs`).
This script does not record and must not invent metrics. Tiny-gpt2, Llama,
and fake-adapter output must not overwrite `docs/findings`.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llmfr.adapters.huggingface import PORTFOLIO_DEMO_MODEL_ID, PORTFOLIO_DEMO_MODEL_REVISION
from llmfr.core.schema import load_path
from llmfr.study import (
    GRADING_RULE,
    MIN_GRADEABLE_FOR_VERDICT,
    RATE_RATIO_FOR_POSITIVE,
    descriptive_verdict,
    summarize_pairs,
)

ROOT = Path(__file__).resolve().parents[1]
FINDING_DIR = ROOT / "docs" / "findings"
DEFAULT_RESULTS = FINDING_DIR / "gs-t22q-results.json"
DEFAULT_FINDING = FINDING_DIR / "gs-t22q.md"
DEFAULT_TRACES = ROOT / "examples" / "findings" / "gs-t22q"
DEFAULT_PROMPTS = DEFAULT_TRACES / "prompts.jsonl"

STUDY_COMMAND = (
    "llmfr study examples/findings/gs-t22q/prompts.jsonl "
    f"--model {PORTFOLIO_DEMO_MODEL_ID} "
    f"--revision {PORTFOLIO_DEMO_MODEL_REVISION} "
    "--store .llmfr-gs-t22q --max-new-tokens 64 --capture-k 5 --json"
)

VERDICT_RULE = (
    "GS-T22n descriptive label only (not emitted by llmfr study; not a p-value): "
    "positive if both sampling and decoding-config disagree_rate are defined, "
    f"each n_gradeable_diverged >= {MIN_GRADEABLE_FOR_VERDICT}, and the higher "
    f"rate is at least {RATE_RATIO_FOR_POSITIVE}x the lower (or lower is 0 and "
    "higher > 0); else null."
)


def _docs_bound(path: Path) -> bool:
    resolved = path.resolve()
    return resolved == DEFAULT_RESULTS.resolve() or resolved == DEFAULT_FINDING.resolve()


def _repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _fmt_rate(value: float | None) -> str:
    if value is None:
        return "n/a (denominator 0; not invented)"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def load_study_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"{path} missing pairs")
    return payload


def require_portfolio_qwen(payload: Mapping[str, Any]) -> None:
    model = str(payload.get("model") or "")
    revision = payload.get("revision")
    if model != PORTFOLIO_DEMO_MODEL_ID:
        raise ValueError(
            f"refusing model {model!r}; GS-T22q is {PORTFOLIO_DEMO_MODEL_ID} "
            "(no silent Llama, no tiny-gpt2)"
        )
    if revision != PORTFOLIO_DEMO_MODEL_REVISION:
        raise ValueError(f"refusing revision {revision!r}; pin {PORTFOLIO_DEMO_MODEL_REVISION}")


def _trace_filename(item_id: str, kind: str, side: str) -> str:
    return f"{item_id}_{kind}_{side}.jsonl"


def copy_named_traces(
    *,
    store_root: Path,
    pairs: Sequence[Mapping[str, Any]],
    dest: Path,
) -> list[dict[str, Any]]:
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    for pair in pairs:
        row = dict(pair)
        item_id = str(row["item_id"])
        kind = str(row["intended_kind"])
        for side, key in (("a", "trace_a"), ("b", "trace_b")):
            trace_id = str(row[key])
            src = store_root / "traces" / f"{trace_id}.jsonl"
            if not src.is_file():
                raise FileNotFoundError(f"store missing trace {trace_id}: {src}")
            named = dest / _trace_filename(item_id, kind, side)
            shutil.copyfile(src, named)
            row[f"path_{side}"] = _repo_rel(named)
        copied.append(row)
    return copied


def attach_finding_fields(
    payload: Mapping[str, Any],
    *,
    pairs: Sequence[Mapping[str, Any]],
    recorded_at: str,
    command: str,
    hardware: Mapping[str, Any],
    library_versions: Mapping[str, str],
    n_items_attempted: int | None = None,
    stop_reason: str | None = None,
) -> dict[str, Any]:
    summary = summarize_pairs(pairs)
    summary["descriptive_verdict"] = descriptive_verdict(summary["by_class"])
    out = dict(payload)
    out["finding_id"] = "GS-T22q"
    out["recorded_at"] = recorded_at
    out["backend"] = "hf"
    out["command"] = command
    out["grading_rule"] = GRADING_RULE
    out["verdict_rule"] = VERDICT_RULE
    out["hardware"] = dict(hardware)
    out["library_versions"] = dict(library_versions)
    out["pairs"] = list(pairs)
    out["summary"] = summary
    out["n_items"] = len({pair["item_id"] for pair in pairs})
    out["n_pairs"] = len(pairs)
    if n_items_attempted is not None:
        out["n_items_attempted"] = n_items_attempted
    if stop_reason:
        out["stop_reason"] = stop_reason
    else:
        out.pop("stop_reason", None)
    return out


def hardware_now() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
    }


def library_versions_from_trace(path: Path) -> dict[str, str]:
    trace = load_path(path)
    versions = dict(trace.environment.library_versions)
    if not versions:
        raise ValueError(f"{path} has no environment.library_versions")
    return versions


def render_finding(results: Mapping[str, Any]) -> str:
    summary = results["summary"]
    by_class = summary["by_class"]
    sampling = by_class["sampling"]
    decoding = by_class["decoding config"]
    verdict = summary["descriptive_verdict"]
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
        observed = pair["observed_class"] if pair["observed_class"] else "no first divergence"
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
    stop = results.get("stop_reason")
    n_attempted = results.get("n_items_attempted", n_items)
    stop_line = (
        f"- **Stop:** {stop}"
        if stop
        else f"- **N completed:** {n_items} of {n_attempted} listed prompts."
    )
    lines = [
        "# GS-T22q: Scaled sampling vs decoding-config correctness disagreement",
        "",
        f"Status: measured (GS-T22n descriptive label: {verdict})",
        f"Date: {recorded_at}",
        "Supersedes: N=8 GS-T22n (`docs/findings/gs-t22n.md`).",
        "",
        "## Question",
        "",
        "When two recorded generations split, does the M5 first-divergence class",
        "(`sampling` vs `decoding config`) predict whether the completions disagree",
        "on correctness, at a larger N than GS-T22n? Report the measured rates.",
        "Do not invent a metric the run did not produce. `llmfr study` does not",
        "emit a positive/null verdict; the GS-T22n descriptive label below uses",
        "the same documented 2x rule and is not a p-value.",
        "",
        "## Setup",
        "",
        f"- **N items:** {n_items} gradeable integer questions ({n_pairs} pairs;",
        "  2 pair kinds per item).",
        stop_line,
        f"- **Model:** `{results['model']}` revision `{results['revision']}`.",
        "- **Library default / CI smoke:** `sshleifer/tiny-gpt2` is not used",
        "  for this finding. Llama was not recorded (gated Hub repo; no silent",
        "  substitution).",
        f"- **Sampling pairs:** seed {seeds[0]} vs {seeds[1]},",
        f"  temperature {results['sampling_temperature']}.",
        f"- **Decoding-config pairs:** seed {results['decoding_seed']},",
        f"  temperature {dec_temps[0]} vs {dec_temps[1]}.",
        f"- **max_new_tokens:** {results['max_new_tokens']};",
        f"  **capture_k:** {results['capture_k']}.",
        f"- **Grading rule:** {results['grading_rule']}",
        f"- **Descriptive label rule:** {results['verdict_rule']}",
        f"- **Hardware:** {results['hardware']['platform']}; "
        f"Python {results['hardware']['python_version']}; "
        f"cpus={results['hardware']['cpu_count']}.",
        f"- **Library versions:** {results['library_versions']}",
        "- **Backend:** `hf` via `llmfr study` (live recorder + `compare_traces`",
        "  on stored events; compare does not call a model).",
        "",
        "Record (does not write this finding):",
        "",
        "```bash",
        STUDY_COMMAND,
        "```",
        "",
        "Pack named traces and regenerate this file from that JSON + store:",
        "",
        "```bash",
        "python scripts/gs_t22q_pack_study.py --from-json STUDY.json "
        "--from-store .llmfr-gs-t22q --write-docs",
        "```",
        "",
        "Recompute the table from checked-in JSON with no model:",
        "",
        "```bash",
        "python scripts/gs_t22q_pack_study.py --from-results docs/findings/gs-t22q-results.json",
        "```",
        "",
        "Raw JSON: `docs/findings/gs-t22q-results.json`.",
        "Prompts: `examples/findings/gs-t22q/prompts.jsonl`.",
        "",
        "## Result",
        "",
        f"**Rates (raw).** Sampling disagree rate is",
        f"{_fmt_rate(sampling['disagree_rate'])} "
        f"({sampling['disagree']}/{sampling['n_gradeable_diverged']});",
        "decoding-config disagree rate is",
        f"{_fmt_rate(decoding['disagree_rate'])} "
        f"({decoding['disagree']}/{decoding['n_gradeable_diverged']}).",
        "Wrong-answer rate among gradeable traces is",
        f"{_fmt_rate(sampling['wrong_answer_rate'])} for sampling pairs and",
        f"{_fmt_rate(decoding['wrong_answer_rate'])} for decoding-config pairs.",
        f"GS-T22n descriptive label: **{verdict}**.",
        "",
        "## So-what",
        "",
        _so_what(sampling, decoding, verdict, n_items),
        "",
        "## Table",
        "",
        table_header,
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        rows[0],
        rows[1],
        "",
        f"Pairs with no first event divergence: {summary['n_no_first_divergence']}.",
        f"Other observed classes: {other}.",
        "",
        "## Pair detail",
        "",
        *pair_lines,
        "",
        "## Limits",
        "",
        "- This is not a leaderboard and not a p-value.",
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
        "- Llama-3.2-1B-Instruct was not used (gated; no HF_TOKEN).",
        "",
    ]
    return "\n".join(lines)


def _so_what(
    sampling: Mapping[str, Any],
    decoding: Mapping[str, Any],
    verdict: str,
    n_items: int,
) -> str:
    rate_s = sampling["disagree_rate"]
    rate_d = decoding["disagree_rate"]
    if rate_s is None or rate_d is None:
        return (
            f"On N={n_items} items, at least one class had no gradeable diverged "
            "pairs, so there is no comparable disagree rate."
        )
    if verdict == "null":
        return (
            f"On N={n_items} `Qwen/Qwen2.5-0.5B-Instruct` items, sampling and "
            "decoding-config splits did not separate on this correctness-"
            "disagreement count (GS-T22n 2x descriptive label: null). First-"
            "divergence class is still a useful debug label; it did not predict "
            "wrong-answer disagreement here."
        )
    return (
        f"On N={n_items} `Qwen/Qwen2.5-0.5B-Instruct` items, the two split "
        "classes separated on correctness-disagreement by the GS-T22n 2x "
        "descriptive label. Read the table; this is not a p-value."
    )


def write_source(
    out: Path,
    *,
    model: str,
    revision: str | None,
    n_items: int,
    max_new_tokens: int,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    revision_line = "" if revision is None else f"revision={revision}\n"
    (out / "SOURCE.txt").write_text(
        "finding=GS-T22q\n"
        "recorded_via=llmfr study\n"
        f"model={model}\n"
        f"{revision_line}"
        f"n_items={n_items}\n"
        f"max_new_tokens={max_new_tokens}\n"
        "sampling_seeds=1 2\n"
        "sampling_temperature=1.0\n"
        "decoding_seed=1 temperatures=0.7 1.2\n"
        f"grading_rule={GRADING_RULE}\n"
        "llama=not recorded (gated Hub repo; no silent substitution)\n"
        f"command={STUDY_COMMAND}\n",
        encoding="utf-8",
    )


def pack_from_study(
    *,
    study_json: Path,
    store: Path,
    traces_dir: Path,
    command: str,
    n_items_attempted: int | None,
    stop_reason: str | None,
) -> dict[str, Any]:
    payload = load_study_json(study_json)
    require_portfolio_qwen(payload)
    pairs = copy_named_traces(store_root=store, pairs=payload["pairs"], dest=traces_dir)
    first_path = ROOT / pairs[0]["path_a"]
    first = load_path(first_path)
    if first.model.name != PORTFOLIO_DEMO_MODEL_ID:
        raise ValueError(f"packed trace model is {first.model.name}, not portfolio Qwen")
    if first.model.revision != PORTFOLIO_DEMO_MODEL_REVISION:
        raise ValueError(f"packed trace revision is {first.model.revision}")
    recorded_at = first.run_metadata.created_at
    if recorded_at.tzinfo is None:
        recorded_at = recorded_at.replace(tzinfo=UTC)
    return attach_finding_fields(
        payload,
        pairs=pairs,
        recorded_at=recorded_at.isoformat(),
        command=command,
        hardware=hardware_now(),
        library_versions=library_versions_from_trace(first_path),
        n_items_attempted=n_items_attempted,
        stop_reason=stop_reason,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-json", default=None, help="llmfr study --json output.")
    parser.add_argument("--from-store", default=None, help="TraceStore used by that study.")
    parser.add_argument(
        "--from-results",
        default=None,
        help="Recompute summary/table from a checked-in results JSON. No model.",
    )
    parser.add_argument("--out", default=None, help="Named-trace directory.")
    parser.add_argument("--results", default=None, help="Results JSON path.")
    parser.add_argument("--finding", default=None, help="Finding markdown path.")
    parser.add_argument(
        "--command",
        default=STUDY_COMMAND,
        help="Recorded llmfr study invocation (documentation only).",
    )
    parser.add_argument(
        "--n-items-attempted",
        type=int,
        default=None,
        help="Prompt-file size if the run stopped early.",
    )
    parser.add_argument(
        "--stop-reason",
        default=None,
        help="Honest reason if N is below the prompt-file size.",
    )
    parser.add_argument(
        "--write-docs",
        action="store_true",
        help="Write results JSON and finding markdown.",
    )
    args = parser.parse_args(None if argv is None else list(argv))

    results_path = DEFAULT_RESULTS if args.results is None else Path(args.results)
    finding_path = DEFAULT_FINDING if args.finding is None else Path(args.finding)
    traces_dir = DEFAULT_TRACES if args.out is None else Path(args.out)

    if args.from_results:
        payload = json.loads(Path(args.from_results).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            sys.stderr.write("error: results JSON is not an object\n")
            return 2
        payload["summary"] = summarize_pairs(payload["pairs"])
        payload["summary"]["descriptive_verdict"] = descriptive_verdict(
            payload["summary"]["by_class"]
        )
        text = render_finding(payload)
        sys.stdout.write(text)
        if args.write_docs:
            require_portfolio_qwen(payload)
            _write_json(results_path, payload)
            finding_path.write_text(text, encoding="utf-8")
        return 0

    if args.from_json is None or args.from_store is None:
        sys.stderr.write("error: pass --from-json PATH --from-store DIR, or --from-results PATH\n")
        return 2

    try:
        payload = pack_from_study(
            study_json=Path(args.from_json),
            store=Path(args.from_store),
            traces_dir=traces_dir,
            command=args.command,
            n_items_attempted=args.n_items_attempted,
            stop_reason=args.stop_reason,
        )
    except (OSError, TypeError, ValueError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1

    text = render_finding(payload)
    sys.stdout.write(text)
    write_source(
        traces_dir,
        model=str(payload["model"]),
        revision=None if payload["revision"] is None else str(payload["revision"]),
        n_items=int(payload["n_items"]),
        max_new_tokens=int(payload["max_new_tokens"]),
    )
    if args.write_docs:
        if _docs_bound(results_path) or _docs_bound(finding_path):
            try:
                require_portfolio_qwen(payload)
            except ValueError as exc:
                sys.stderr.write(f"error: {exc}\n")
                return 2
        _write_json(results_path, payload)
        finding_path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
