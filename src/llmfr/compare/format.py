"""Human-readable compare report. Root cause vs downstream effects (M6).

Not a polished M7 UX. Section order matches the compare Definition of Done.
"""

from __future__ import annotations

from llmfr.compare.result import CompareResult, FirstDivergence, StepDiff

REPORT_SECTIONS: tuple[str, ...] = (
    "CONFIG DIFFERENCE",
    "EXECUTION",
    "FIRST BEHAVIORAL DIVERGENCE",
    "LIKELY ENABLING CONFIG",
)

DOWNSTREAM_NOT_ROOT_CAUSE = "later context, logit, and token diffs are not a new root cause"


def format_compare_result(result: CompareResult) -> str:
    lines = [
        f"compare a={result.trace_a} b={result.trace_b}",
        "",
        "CONFIG DIFFERENCE",
    ]
    if not result.config_diffs:
        lines.append("  (none)")
    else:
        for diff in result.config_diffs:
            lines.append(f"  {diff.field}: {_show(diff.a)} vs {_show(diff.b)}")

    lines.extend(["", "EXECUTION"])
    lines.extend(_format_execution(result))

    lines.extend(["", "FIRST BEHAVIORAL DIVERGENCE"])
    first = result.first_divergence
    if first is None:
        lines.append("  (none; recorded behavior matches)")
    else:
        lines.extend(_format_first(first))

    lines.extend(["", "LIKELY ENABLING CONFIG"])
    lines.extend(_format_enabling(result))

    lines.extend(["", _downstream_heading(result)])
    lines.extend(_format_downstream(result))

    if result.notes:
        lines.extend(["", "notes"])
        for note in result.notes:
            lines.append(f"  {note}")

    lines.append("")
    lines.append("identical" if result.identical else "diverged")
    return "\n".join(lines) + "\n"


def _format_execution(result: CompareResult) -> list[str]:
    matched = result.matched_prefix_steps
    lines = [f"  recorded length: {result.event_count_a} vs {result.event_count_b} steps"]
    if matched <= 0:
        if result.first_divergence is None:
            lines.append("  same steps: (none recorded)")
        else:
            lines.append("  same steps: (none; paths split at step 0)")
        return lines
    last = matched - 1
    span = "0" if last == 0 else f"0-{last}"
    if result.first_divergence is None:
        lines.append(f"  same steps: {span} (all recorded steps still match)")
    else:
        lines.append(
            f"  same steps: {span} "
            "(sampled tokens, model-visible context, captured logits still match)"
        )
    return lines


def _format_first(first: FirstDivergence) -> list[str]:
    return [
        f"  step {first.step}",
        f"  class: {first.classification}",
        f"  differences: {', '.join(first.differences)}",
        (
            "  sampled token: "
            f"{_token(first.a_sampled_token, first.a_sampled_token_id)} vs "
            f"{_token(first.b_sampled_token, first.b_sampled_token_id)}"
        ),
        f"  reason: {first.reason}",
    ]


def _format_enabling(result: CompareResult) -> list[str]:
    if result.first_divergence is None:
        return ["  (none; no first behavioral split)"]
    lines: list[str] = []
    if not result.likely_enabling_config:
        lines.append("  (none recorded)")
    else:
        for diff in result.likely_enabling_config:
            lines.append(f"  {diff.field}: {_show(diff.a)} vs {_show(diff.b)}")
    if result.enabling_summary:
        lines.append(f"  {result.enabling_summary}")
    return lines


def _downstream_heading(result: CompareResult) -> str:
    first = result.first_divergence
    if first is None:
        return "DOWNSTREAM EFFECTS"
    return f"Steps {first.step + 1}+: downstream effects"


def _format_downstream(result: CompareResult) -> list[str]:
    if not result.downstream:
        if result.first_divergence is None:
            return ["  (none)"]
        return ["  (none; no later context, logit, or token diffs)"]
    lines = [f"  {DOWNSTREAM_NOT_ROOT_CAUSE}"]
    for row in result.downstream:
        lines.append(_format_step(row))
    return lines


def _format_step(row: StepDiff) -> str:
    sampled = (
        f"{_token(row.a_sampled_token, row.a_sampled_token_id)} vs "
        f"{_token(row.b_sampled_token, row.b_sampled_token_id)}"
    )
    return f"  step {row.step} (not root cause): {', '.join(row.differences)} ({sampled})"


def _token(token: str | None, token_id: int | None) -> str:
    if token is None and token_id is None:
        return "missing"
    shown = repr(token) if token is not None else "None"
    return f"{shown} (id={token_id})"


def _show(value: str | None) -> str:
    if value is None:
        return "None"
    return value
