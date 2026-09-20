"""Human table for a graded sampling vs decoding-config study."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from llmfr.study.run import StudyReport


def format_study_report(report: StudyReport) -> str:
    summary = report.summary
    by_class = summary["by_class"]
    sampling = by_class["sampling"]
    decoding = by_class["decoding config"]
    revision = "none" if report.revision is None else report.revision
    header = (
        "| class | n pairs | n gradeable diverged | agree correct | "
        "agree wrong | disagree | ungraded | disagree rate | wrong-answer rate |"
    )
    lines = [
        f"study n_items={report.n_items} n_pairs={report.n_pairs}",
        f"model={report.model} revision={revision}",
        (
            f"sampling seeds: {report.sampling_seeds[0]} vs {report.sampling_seeds[1]} "
            f"(temperature {report.sampling_temperature})"
        ),
        (
            f"decoding-config temperatures: {report.decoding_temperatures[0]} vs "
            f"{report.decoding_temperatures[1]} (seed {report.decoding_seed})"
        ),
        f"max_new_tokens={report.max_new_tokens} capture_k={report.capture_k}",
        f"grading: {report.grading_rule}",
        "",
        header,
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        _class_row("sampling", sampling),
        _class_row("decoding config", decoding),
        "",
        f"pairs with no first event divergence: {summary['n_no_first_divergence']}",
        f"other observed classes: {_other(summary['other_classes'])}",
        "",
        "pairs",
    ]
    for pair in report.pairs:
        observed = _pair_class_label(pair)
        lines.append(
            f"  {pair['item_id']} intended {pair['intended_kind']}: "
            f"class {observed} step={pair['first_step']} "
            f"grades {pair['grade_a']}/{pair['grade_b']} "
            f"(extracted {pair['extracted_a']}/{pair['extracted_b']}, gold {pair['gold']}) "
            f"outcome {pair['outcome']}"
        )
    lines.append("")
    return "\n".join(lines)


def _pair_class_label(pair: Mapping[str, Any]) -> str:
    observed = pair.get("observed_class")
    if observed:
        return str(observed)
    if pair.get("identical"):
        return "identical"
    return "no first divergence"


def _class_row(name: str, row: Mapping[str, Any]) -> str:
    return (
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
            ]
        )
        + " |"
    )


def _fmt_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _other(other_classes: Mapping[str, int]) -> str:
    if not other_classes:
        return "none"
    return ", ".join(f"{name}={count}" for name, count in sorted(other_classes.items()))
