"""Two-trace compare and first-divergence tests. Fake adapter; no live model."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from pytest import CaptureFixture, MonkeyPatch

from llmfr.cli import build_parser, run
from llmfr.compare import DIVERGENCE_CLASSES, compare_traces, format_compare_result
from llmfr.core.schema import GenerationConfig, Trace, dumps_json
from llmfr.record import record_generation
from llmfr.storage import TraceStore
from tests.factories import TRACE_ID_B, make_trace
from tests.fakes import FakeCausalLMAdapter


def _flat_adapter() -> FakeCausalLMAdapter:
    peaked_after = {
        (1, 0): (5.0, 0.0, 0.0, 0.0),
        (1, 1): (0.0, 5.0, 0.0, 0.0),
        (1, 2): (0.0, 0.0, 5.0, 0.0),
        (1, 3): (0.0, 0.0, 0.0, 5.0),
    }
    return FakeCausalLMAdapter(
        prompt_ids=[1],
        logits=(1.0, 1.0, 1.0, 1.0),
        logits_for_prefix={(1,): (1.0, 1.0, 1.0, 1.0), **peaked_after},
    )


def _record(*, seed: int, max_new_tokens: int = 2) -> Trace:
    return record_generation(
        _flat_adapter(),
        "x",
        generation=GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            seed=seed,
        ),
    )


def _seeded_pair() -> tuple[Trace, Trace]:
    """Two stored traces that split at step 0 under LocalRNG (Case B/E)."""
    for seed_a, seed_b in ((42, 43), (1, 2), (7, 8), (100, 101)):
        trace_a = _record(seed=seed_a)
        trace_b = _record(seed=seed_b)
        ids_a = [event.sampled_token_id for event in trace_a.events]
        ids_b = [event.sampled_token_id for event in trace_b.events]
        if ids_a[0] != ids_b[0]:
            return trace_a, trace_b
    raise AssertionError("LocalRNG did not split at step 0 for the tried seeds")


def _shift_event_logits(trace: Trace, *, delta: float) -> Trace:
    events = []
    for event in trace.events:
        top_k = [
            candidate.model_copy(update={"logit": (candidate.logit or 0.0) + delta})
            for candidate in event.top_k
        ]
        events.append(
            event.model_copy(
                update={
                    "top_k": top_k,
                    "sampled_logit": None
                    if event.sampled_logit is None
                    else event.sampled_logit + delta,
                }
            )
        )
    return trace.model_copy(update={"events": events})


def test_divergence_classes_match_acceptance() -> None:
    assert DIVERGENCE_CLASSES == (
        "prompt/history",
        "tokenizer",
        "model-visible context",
        "model/version",
        "raw-logit",
        "decoding config",
        "probability distribution",
        "sampling",
        "unknown/runtime",
    )


def test_identical_traces_have_no_first_divergence() -> None:
    result = compare_traces(make_trace(), make_trace())
    assert result.identical is True
    assert result.first_divergence is None
    assert result.downstream == ()
    assert result.config_diffs == ()


def test_case_b_seed_split_is_sampling() -> None:
    trace_a, trace_b = _seeded_pair()
    result = compare_traces(trace_a, trace_b)
    first = result.first_divergence
    assert first is not None
    assert first.classification == "sampling"
    assert "generation_config.seed" in {diff.field for diff in result.config_diffs}
    report = format_compare_result(result)
    assert "class: sampling" in report
    assert "downstream effects" in report
    assert "not a new root cause" in report


def test_case_e_later_logit_and_context_diffs_are_downstream() -> None:
    trace_a, trace_b = _seeded_pair()
    result = compare_traces(trace_a, trace_b)
    first = result.first_divergence
    assert first is not None
    assert first.classification == "sampling"
    later = [row for row in result.steps if row.step > first.step]
    assert later, "need a token after the first split so Case E can tag fallout"
    assert all(row.role == "downstream" for row in later)
    assert result.downstream == tuple(later)
    assert any(
        "full_history" in row.differences
        or "model_visible_context" in row.differences
        or "raw_logits" in row.differences
        for row in later
    )
    assert not any(row.role == "first" for row in result.downstream)
    report = format_compare_result(result)
    assert "not a new root cause" in report
    assert "class: sampling" in report
    assert "class: raw-logit" not in report
    assert "raw-logit" not in {row.role for row in later}


def test_case_f_window_split_is_model_visible_context() -> None:
    trace_a = make_trace(visible_limit=None)
    # Same full_history and sampled tokens; clipped window plus different logits.
    # First cause must be the window, not raw-logit (Case F).
    trace_b = _shift_event_logits(make_trace(trace_id=TRACE_ID_B, visible_limit=2), delta=-3.0)
    result = compare_traces(trace_a, trace_b)
    first = result.first_divergence
    assert first is not None
    assert first.step == 0
    assert first.classification == "model-visible context"
    assert "model_visible_context" in first.differences
    assert "raw_logits" in first.differences
    later = [row for row in result.steps if row.step > 0]
    assert later
    assert all(row.role == "downstream" for row in later)
    assert any("model_visible_context" in row.differences for row in later)


def test_prompt_history_classifies_before_later_token_diffs() -> None:
    trace_a = make_trace()
    trace_b = make_trace(trace_id=TRACE_ID_B, prompt="Say goodbye")
    result = compare_traces(trace_a, trace_b)
    first = result.first_divergence
    assert first is not None
    assert first.classification == "prompt/history"
    assert "run_metadata.prompt" in {diff.field for diff in result.config_diffs}


def test_tokenizer_same_prompt_different_ids() -> None:
    trace_a = make_trace()
    trace_b = make_trace(
        trace_id=TRACE_ID_B,
        prompt_token_ids=[9, 8, 7],
        model=make_trace().model.model_copy(update={"tokenizer": "other-tok"}),
    )
    result = compare_traces(trace_a, trace_b)
    first = result.first_divergence
    assert first is not None
    assert first.classification == "tokenizer"
    assert "run_metadata.prompt_token_ids" in {diff.field for diff in result.config_diffs}
    assert "model.tokenizer" in {diff.field for diff in result.config_diffs}


def test_model_version_beats_raw_logit() -> None:
    trace_a = make_trace()
    trace_b = _shift_event_logits(
        make_trace(
            trace_id=TRACE_ID_B,
            model=make_trace().model.model_copy(
                update={"revision": "def", "name": "tiny-model-v2"}
            ),
        ),
        delta=1.0,
    )
    result = compare_traces(trace_a, trace_b)
    first = result.first_divergence
    assert first is not None
    assert first.classification == "model/version"
    assert "model.revision" in {diff.field for diff in result.config_diffs}
    assert "model.name" in {diff.field for diff in result.config_diffs}


def test_raw_logit_when_model_and_context_match() -> None:
    trace_a = make_trace()
    trace_b = _shift_event_logits(make_trace(trace_id=TRACE_ID_B), delta=1.0)
    result = compare_traces(trace_a, trace_b)
    first = result.first_divergence
    assert first is not None
    assert first.classification == "raw-logit"
    assert first.step == 0
    assert all(row.role == "downstream" for row in result.downstream)


def test_decoding_config_same_logits_different_temperature() -> None:
    trace_a = make_trace()
    event0 = trace_a.events[0].model_copy(
        update={"sampled_token_id": 202, "sampled_token": "alt2", "sampled_rank": 2}
    )
    trace_b = make_trace(
        trace_id=TRACE_ID_B,
        generation=GenerationConfig(
            temperature=1.5,
            max_new_tokens=8,
            do_sample=True,
            seed=1,
        ),
    ).model_copy(update={"events": [event0, make_trace().events[1]]})
    result = compare_traces(trace_a, trace_b)
    first = result.first_divergence
    assert first is not None
    assert first.classification == "decoding config"
    assert "sampled_token" in first.differences
    assert "raw_logits" not in first.differences
    fields = {diff.field for diff in result.config_diffs}
    assert "generation_config.temperature" in fields
    assert "generation_config.do_sample" in fields


def test_probability_distribution_when_logits_match() -> None:
    trace_a = make_trace()
    events = []
    for event in make_trace().events:
        top_k = [
            candidate.model_copy(update={"prob": 0.1 if candidate.prob is None else 0.1})
            for candidate in event.top_k
        ]
        events.append(
            event.model_copy(update={"top_k": top_k, "sampled_prob": 0.1, "sampled_logprob": -2.3})
        )
    trace_b = make_trace(trace_id=TRACE_ID_B).model_copy(update={"events": events})
    result = compare_traces(trace_a, trace_b)
    first = result.first_divergence
    assert first is not None
    assert first.classification == "probability distribution"
    assert "probabilities" in first.differences
    assert "raw_logits" not in first.differences


def test_hosted_api_without_logits_is_unknown_runtime() -> None:
    trace_a = make_trace(logits_mode="none")
    trace_b = make_trace(
        logits_mode="none",
        trace_id=TRACE_ID_B,
        event_tokens=((109, "Nope"), (102, " world")),
    )
    result = compare_traces(trace_a, trace_b)
    first = result.first_divergence
    assert first is not None
    assert first.classification == "unknown/runtime"
    assert first.reason is not None
    assert "refusing to invent logits" in first.reason
    assert any("logits.mode=none" in note for note in result.notes)


def test_hosted_api_seed_split_is_sampling_without_inventing_logits() -> None:
    trace_a = make_trace(logits_mode="none")
    trace_b = make_trace(
        logits_mode="none",
        trace_id=TRACE_ID_B,
        event_tokens=((109, "Nope"), (102, " world")),
        generation=GenerationConfig(temperature=0.0, max_new_tokens=8, do_sample=False, seed=99),
    )
    result = compare_traces(trace_a, trace_b)
    first = result.first_divergence
    assert first is not None
    assert first.classification == "sampling"
    assert first.classification != "raw-logit"


def test_length_mismatch_uses_decoding_config() -> None:
    trace_a = make_trace()
    trace_b = make_trace(
        trace_id=TRACE_ID_B,
        event_tokens=((101, "Hello"),),
        generation=GenerationConfig(temperature=0.0, max_new_tokens=1, do_sample=False, seed=1),
    )
    result = compare_traces(trace_a, trace_b)
    first = result.first_divergence
    assert first is not None
    assert first.step == 1
    assert first.classification == "decoding config"
    assert first.differences == ("length",)
    fields = {diff.field for diff in result.config_diffs}
    assert "generation_config.max_new_tokens" in fields
    assert "event.top_k.length" not in fields
    assert not any("event top-k depths differ" in note for note in result.notes)


def test_prompt_metadata_without_history_split_is_not_prompt_history() -> None:
    trace_a = make_trace()
    shifted = _shift_event_logits(make_trace(trace_id=TRACE_ID_B), delta=1.0)
    trace_b = shifted.model_copy(
        update={
            "run_metadata": shifted.run_metadata.model_copy(update={"prompt": "other prompt"}),
        }
    )
    assert trace_a.events[0].full_history.token_ids == trace_b.events[0].full_history.token_ids
    assert trace_a.events[0].full_history.text == trace_b.events[0].full_history.text
    result = compare_traces(trace_a, trace_b)
    first = result.first_divergence
    assert first is not None
    assert first.classification == "raw-logit"
    assert first.classification != "prompt/history"
    assert "run_metadata.prompt" in {diff.field for diff in result.config_diffs}


def test_unequal_capture_k_matching_prefix_is_config_only() -> None:
    trace_a = make_trace(k=5)
    trace_b = make_trace(k=3, trace_id=TRACE_ID_B)
    result = compare_traces(trace_a, trace_b)
    assert result.first_divergence is None
    assert result.identical is False
    assert "run_metadata.logits.k" in {diff.field for diff in result.config_diffs}
    assert any("capture k" in note for note in result.notes)


def test_unequal_capture_k_token_split_is_not_raw_logit() -> None:
    trace_a = make_trace(k=5)
    trace_b = make_trace(
        k=3,
        trace_id=TRACE_ID_B,
        event_tokens=((109, "Nope"), (102, " world")),
    )
    result = compare_traces(trace_a, trace_b)
    first = result.first_divergence
    assert first is not None
    assert first.classification != "raw-logit"
    assert first.classification == "unknown/runtime"


def test_case_a_identical_greedy_recordings_match() -> None:
    generation = GenerationConfig(max_new_tokens=2, do_sample=False, seed=3)
    trace_a = record_generation(
        FakeCausalLMAdapter(prompt_ids=[1, 2]),
        "hello",
        generation=generation,
    )
    trace_b = record_generation(
        FakeCausalLMAdapter(prompt_ids=[1, 2]),
        "hello",
        generation=generation,
    )
    result = compare_traces(trace_a, trace_b)
    assert result.identical is True
    assert result.first_divergence is None
    assert result.config_diffs == ()
    assert [event.sampled_token_id for event in trace_a.events] == [
        event.sampled_token_id for event in trace_b.events
    ]


def test_case_c_seed_diff_without_token_split_is_config_only() -> None:
    trace_a = make_trace()
    trace_b = make_trace(
        trace_id=TRACE_ID_B,
        generation=trace_a.generation_config.model_copy(update={"seed": 99}),
    )
    result = compare_traces(trace_a, trace_b)
    assert result.first_divergence is None
    assert result.identical is False
    assert {diff.field for diff in result.config_diffs} == {"generation_config.seed"}
    report = format_compare_result(result)
    assert "generation_config.seed" in report
    assert "class:" not in report
    assert report.strip().endswith("diverged")


def test_case_d_recorded_window_split_is_model_visible_context() -> None:
    prompt_ids = [0, 1, 2, 3, 4]
    generation = GenerationConfig(max_new_tokens=2, do_sample=False)
    wide = record_generation(
        FakeCausalLMAdapter(prompt_ids=prompt_ids, max_visible_tokens=16),
        "long",
        generation=generation,
    )
    narrow = record_generation(
        FakeCausalLMAdapter(prompt_ids=prompt_ids, max_visible_tokens=2),
        "long",
        generation=generation,
    )
    result = compare_traces(wide, narrow)
    first = result.first_divergence
    assert first is not None
    assert first.classification == "model-visible context"
    assert "model_visible_context" in first.differences
    assert first.classification != "raw-logit"


def _with_topk_depth(trace: Trace, depth: int) -> Trace:
    events = []
    for event in trace.events:
        events.append(event.model_copy(update={"top_k": event.top_k[:depth]}))
    return trace.model_copy(update={"events": events})


def test_case_g_event_topk_depth_mismatch_same_k_is_not_raw_logit() -> None:
    trace_a = make_trace(k=5)
    trace_b = _with_topk_depth(make_trace(k=5, trace_id=TRACE_ID_B), 3)
    result = compare_traces(trace_a, trace_b)
    assert result.first_divergence is None
    assert result.identical is False
    assert "event.top_k.length" in {diff.field for diff in result.config_diffs}
    assert "run_metadata.logits.k" not in {diff.field for diff in result.config_diffs}
    assert any("event top-k depths differ" in note for note in result.notes)


def test_empty_topk_vs_captured_is_not_raw_logit() -> None:
    trace_a = make_trace(k=5)
    trace_b = _with_topk_depth(make_trace(k=5, trace_id=TRACE_ID_B), 0)
    result = compare_traces(trace_a, trace_b)
    first = result.first_divergence
    assert result.identical is False
    assert "event.top_k.length" in {diff.field for diff in result.config_diffs}
    if first is not None:
        assert first.classification != "raw-logit"
        assert first.classification == "unknown/runtime"
    assert any("empty top-k" in note for note in result.notes)


def test_compare_cli_files_does_not_create_store(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    path_a.write_text(dumps_json(make_trace()), encoding="utf-8")
    path_b.write_text(dumps_json(make_trace(trace_id=TRACE_ID_B, prompt="other")), encoding="utf-8")
    assert run(["compare", str(path_a), str(path_b)]) == 1
    capsys.readouterr()
    assert not (tmp_path / ".llmfr").exists()
    assert not (tmp_path / "index.sqlite").exists()


def test_compare_cli_missing_file_is_not_unknown_trace_id(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    missing = tmp_path / "a.jsonl"
    present = tmp_path / "b.json"
    present.write_text(dumps_json(make_trace()), encoding="utf-8")
    code = run(["compare", str(missing), str(present)])
    assert code == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "trace file not found" in err
    assert "unknown trace_id" not in err
    assert "Traceback" not in err


def test_compare_cli_files(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    path_a.write_text(dumps_json(make_trace()), encoding="utf-8")
    path_b.write_text(dumps_json(make_trace(trace_id=TRACE_ID_B, prompt="other")), encoding="utf-8")
    assert run(["compare", str(path_a), str(path_b)]) == 1
    out = capsys.readouterr().out
    assert "class: prompt/history" in out
    assert "downstream effects" in out
    assert "CONFIG DIFFERENCE" in out
    assert "Traceback" not in out


def test_compare_cli_store_ids_json(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    store = TraceStore(tmp_path)
    trace_a = make_trace()
    trace_b = make_trace(trace_id=TRACE_ID_B, prompt="other")
    store.put(trace_a)
    store.put(trace_b)
    code = run(
        [
            "compare",
            str(trace_a.run_metadata.trace_id),
            str(trace_b.run_metadata.trace_id),
            "--store",
            str(tmp_path),
            "--json",
        ]
    )
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["first_divergence"]["classification"] == "prompt/history"
    assert payload["identical"] is False


def test_compare_cli_identical_files(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    path = tmp_path / "same.json"
    path.write_text(dumps_json(make_trace()), encoding="utf-8")
    assert run(["compare", str(path), str(path)]) == 0
    out = capsys.readouterr().out
    assert "identical" in out
    assert "class:" not in out


def test_compare_cli_unknown_id(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    TraceStore(tmp_path)
    missing = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    code = run(["compare", str(missing), str(missing), "--store", str(tmp_path)])
    assert code == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "unknown trace_id" in err
    assert "Traceback" not in err


def test_compare_is_listed() -> None:
    help_text = build_parser().format_help()
    assert "compare" in help_text
    assert "Compare commands are not available yet." not in help_text
