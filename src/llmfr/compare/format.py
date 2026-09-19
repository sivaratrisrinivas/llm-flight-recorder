"""Human-readable compare report. Concise; not a polished M7 UX."""

from __future__ import annotations

from llmfr.compare.result import CompareResult, FirstDivergence, StepDiff


def format_compare_result(result: CompareResult) -> str:
    lines = [
        f"compare a={result.trace_a} b={result.trace_b}",
        "",
        "config differences",
    ]
    if not result.config_diffs:
        lines.append("  (none)")
    else:
        for diff in result.config_diffs:
            lines.append(f"  {diff.field}: {_show(diff.a)} vs {_show(diff.b)}")

    lines.extend(["", "first divergence"])
    first = result.first_divergence
    if first is None:
        lines.append("  (none; recorded behavior matches)")
    else:
        lines.extend(_format_first(first))

    lines.extend(["", "downstream (not root cause)"])
    if not result.downstream:
        lines.append("  (none)")
    else:
        for row in result.downstream:
            lines.append(_format_step(row))

    if result.notes:
        lines.extend(["", "notes"])
        for note in result.notes:
            lines.append(f"  {note}")

    lines.append("")
    lines.append("identical" if result.identical else "diverged")
    return "\n".join(lines) + "\n"


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


def _format_step(row: StepDiff) -> str:
    sampled = (
        f"{_token(row.a_sampled_token, row.a_sampled_token_id)} vs "
        f"{_token(row.b_sampled_token, row.b_sampled_token_id)}"
    )
    return f"  step {row.step}: {', '.join(row.differences)} ({sampled})"


def _token(token: str | None, token_id: int | None) -> str:
    if token is None and token_id is None:
        return "missing"
    shown = repr(token) if token is not None else "None"
    return f"{shown} (id={token_id})"


def _show(value: str | None) -> str:
    if value is None:
        return "None"
    return value
