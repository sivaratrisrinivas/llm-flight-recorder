"""M6 compare report framing: section order, enabling config, Case E language."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest import CaptureFixture

from llmfr.cli import run
from llmfr.compare import (
    DOWNSTREAM_NOT_ROOT_CAUSE,
    REPORT_SECTIONS,
    CompareResult,
    compare_traces,
    format_compare_result,
)
from llmfr.compare.compare import _enabling_summary
from llmfr.compare.result import DivergenceClass, FirstDivergence
from llmfr.core.schema import dumps_json
from tests.factories import TRACE_ID_B, make_trace
from tests.test_compare import _seeded_pair, _shift_event_logits


def _section_positions(report: str) -> list[int]:
    positions = [report.index(name) for name in REPORT_SECTIONS]
    downstream_at = report.lower().index("downstream effects")
    return [*positions, downstream_at]


def test_report_sections_constant_matches_dod() -> None:
    assert REPORT_SECTIONS == (
        "CONFIG DIFFERENCE",
        "EXECUTION",
        "FIRST BEHAVIORAL DIVERGENCE",
        "LIKELY ENABLING CONFIG",
    )


def test_report_section_order_on_sampling_split() -> None:
    result = compare_traces(*_seeded_pair())
    report = format_compare_result(result)
    for name in REPORT_SECTIONS:
        assert name in report
    positions = _section_positions(report)
    assert positions == sorted(positions)
    assert "Steps 1+: downstream effects" in report


def test_identical_report_still_has_section_order() -> None:
    result = compare_traces(make_trace(), make_trace())
    report = format_compare_result(result)
    positions = _section_positions(report)
    assert positions == sorted(positions)
    assert "DOWNSTREAM EFFECTS" in report
    assert result.matched_prefix_steps == 2
    assert result.likely_enabling_config == ()
    assert result.enabling_summary is None
    assert "class:" not in report


def test_case_e_report_does_not_label_later_diffs_root_cause() -> None:
    result = compare_traces(*_seeded_pair())
    first = result.first_divergence
    assert first is not None
    assert first.classification == "sampling"
    assert result.downstream
    report = format_compare_result(result)
    first_heading = report.index("FIRST BEHAVIORAL DIVERGENCE")
    enabling_heading = report.index("LIKELY ENABLING CONFIG")
    downstream_heading = report.index("Steps 1+: downstream effects")
    first_block = report[first_heading:enabling_heading]
    downstream_block = report[downstream_heading:]
    assert "class: sampling" in first_block
    assert "class: raw-logit" not in report
    assert DOWNSTREAM_NOT_ROOT_CAUSE in downstream_block
    assert "not a new root cause" in downstream_block
    assert "raw_logits" in downstream_block or "full_history" in downstream_block
    assert "class:" not in downstream_block
    for row in result.downstream:
        assert f"step {row.step} (not root cause):" in downstream_block
        assert row.role == "downstream"


def test_likely_enabling_config_is_seed_for_sampling() -> None:
    result = compare_traces(*_seeded_pair())
    fields = {diff.field for diff in result.likely_enabling_config}
    assert fields == {"generation_config.seed"}
    assert result.enabling_summary == "seed difference likely enabled this sampling split"
    report = format_compare_result(result)
    enabling_block = report[report.index("LIKELY ENABLING CONFIG") :]
    config_block = report[report.index("CONFIG DIFFERENCE") : report.index("EXECUTION")]
    assert "generation_config.seed" in config_block
    assert "generation_config.seed" in enabling_block
    assert "seed difference likely enabled this sampling split" in enabling_block


def test_likely_enabling_config_omits_unrelated_device_diff() -> None:
    trace_a, trace_b = _seeded_pair()
    trace_b = trace_b.model_copy(
        update={"environment": trace_b.environment.model_copy(update={"device": "cuda"})}
    )
    result = compare_traces(trace_a, trace_b)
    config_fields = {diff.field for diff in result.config_diffs}
    enabling_fields = {diff.field for diff in result.likely_enabling_config}
    assert "generation_config.seed" in config_fields
    assert "environment.device" in config_fields
    assert enabling_fields == {"generation_config.seed"}
    assert "environment.device" not in enabling_fields
    first = result.first_divergence
    assert first is not None
    assert first.classification == "sampling"


def test_prompt_history_enabling_omits_unrelated_tokenizer() -> None:
    trace_a = make_trace()
    trace_b = make_trace(
        trace_id=TRACE_ID_B,
        prompt="Say goodbye",
        model=make_trace().model.model_copy(update={"tokenizer": "other-tok"}),
    )
    result = compare_traces(trace_a, trace_b)
    first = result.first_divergence
    assert first is not None
    assert first.classification == "prompt/history"
    config_fields = {diff.field for diff in result.config_diffs}
    enabling_fields = {diff.field for diff in result.likely_enabling_config}
    assert "run_metadata.prompt" in config_fields
    assert "model.tokenizer" in config_fields
    assert "run_metadata.prompt" in enabling_fields
    assert "model.tokenizer" not in enabling_fields
    assert result.enabling_summary is not None
    assert "prompt or history" in result.enabling_summary


def test_model_visible_context_does_not_invent_enabling_config() -> None:
    trace_a = make_trace(visible_limit=None)
    trace_b = _shift_event_logits(make_trace(trace_id=TRACE_ID_B, visible_limit=2), delta=-3.0)
    result = compare_traces(trace_a, trace_b)
    first = result.first_divergence
    assert first is not None
    assert first.classification == "model-visible context"
    assert result.likely_enabling_config == ()
    assert result.enabling_summary is not None
    assert "no generation_config field records the visible limit" in result.enabling_summary
    report = format_compare_result(result)
    enabling_block = report[
        report.index("LIKELY ENABLING CONFIG") : report.lower().index("downstream effects")
    ]
    assert "(none recorded)" in enabling_block
    assert "class: raw-logit" not in report


def test_execution_describes_matching_prefix_before_length_split() -> None:
    trace_a = make_trace()
    trace_b = make_trace(
        trace_id=TRACE_ID_B,
        event_tokens=((101, "Hello"),),
        generation=make_trace().generation_config.model_copy(update={"max_new_tokens": 1}),
    )
    result = compare_traces(trace_a, trace_b)
    assert result.matched_prefix_steps == 1
    assert result.event_count_a == 2
    assert result.event_count_b == 1
    report = format_compare_result(result)
    exec_block = report[report.index("EXECUTION") : report.index("FIRST BEHAVIORAL DIVERGENCE")]
    assert "recorded length: 2 vs 1 steps" in exec_block
    assert "same steps: 0" in exec_block
    assert "Steps 2+: downstream effects" in report


def _assert_empty_enabling_is_honest(result: CompareResult) -> None:
    assert result.likely_enabling_config == ()
    assert result.enabling_summary is not None
    assert "likely enabled" not in result.enabling_summary
    assert "listed config diffs" not in result.enabling_summary
    assert "no recorded" in result.enabling_summary
    report = format_compare_result(result)
    enabling_block = report[report.index("LIKELY ENABLING CONFIG") :]
    assert "(none recorded)" in enabling_block
    assert result.enabling_summary in enabling_block


def test_sampling_without_seed_diff_empty_enabling_is_honest() -> None:
    trace_a = make_trace()
    event0 = trace_a.events[0].model_copy(
        update={"sampled_token_id": 202, "sampled_token": "alt2", "sampled_rank": 2}
    )
    trace_b = make_trace(trace_id=TRACE_ID_B).model_copy(
        update={"events": [event0, make_trace().events[1]]}
    )
    result = compare_traces(trace_a, trace_b)
    first = result.first_divergence
    assert first is not None
    assert first.classification == "sampling"
    _assert_empty_enabling_is_honest(result)
    assert "sampler draw" in (result.enabling_summary or "")


def test_tokenizer_decode_string_only_empty_enabling_is_honest() -> None:
    trace_a = make_trace()
    event0 = trace_a.events[0].model_copy(update={"sampled_token": "HELLO"})
    trace_b = make_trace(trace_id=TRACE_ID_B).model_copy(
        update={"events": [event0, make_trace().events[1]]}
    )
    result = compare_traces(trace_a, trace_b)
    first = result.first_divergence
    assert first is not None
    assert first.classification == "tokenizer"
    _assert_empty_enabling_is_honest(result)


def test_hand_edited_history_empty_enabling_is_honest() -> None:
    trace_a = make_trace()
    event0 = trace_a.events[0]
    new_ids = [9, 8, 7]
    rewritten = event0.model_copy(
        update={
            "full_history": event0.full_history.model_copy(update={"token_ids": new_ids}),
            "model_visible_context": event0.model_visible_context.model_copy(
                update={"token_ids": new_ids}
            ),
        }
    )
    trace_b = make_trace(trace_id=TRACE_ID_B).model_copy(
        update={"events": [rewritten, trace_a.events[1]]}
    )
    result = compare_traces(trace_a, trace_b)
    first = result.first_divergence
    assert first is not None
    assert first.classification == "prompt/history"
    assert result.config_diffs == ()
    _assert_empty_enabling_is_honest(result)


@pytest.mark.parametrize(
    "classification",
    (
        "prompt/history",
        "tokenizer",
        "model/version",
        "decoding config",
        "sampling",
        "raw-logit",
    ),
)
def test_empty_enabling_summary_does_not_claim_config_cause(
    classification: DivergenceClass,
) -> None:
    first = FirstDivergence(
        step=0,
        classification=classification,
        differences=("sampled_token",),
        reason="test",
    )
    summary = _enabling_summary(first, ())
    assert summary is not None
    assert "likely enabled" not in summary
    assert "listed config diffs" not in summary
    assert "no recorded" in summary


def test_json_compare_includes_m6_fields(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    path_a.write_text(dumps_json(make_trace()), encoding="utf-8")
    path_b.write_text(dumps_json(make_trace(trace_id=TRACE_ID_B, prompt="other")), encoding="utf-8")
    assert run(["compare", str(path_a), str(path_b), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["matched_prefix_steps"] == 0
    assert payload["event_count_a"] == 2
    assert payload["event_count_b"] == 2
    assert "likely_enabling_config" in payload
    assert payload["enabling_summary"]
    assert payload["first_divergence"]["classification"] == "prompt/history"
