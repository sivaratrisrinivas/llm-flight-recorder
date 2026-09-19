"""OpenAI traces fail closed on replay. Compare/inspect still read stored scores."""

from __future__ import annotations

import json
from pathlib import Path

from pytest import CaptureFixture

from llmfr.cli import run
from llmfr.compare import compare_traces
from llmfr.core.format import format_inspect_step
from llmfr.core.schema import GenerationConfig, dumps_json
from llmfr.replay import replay_trace
from llmfr.storage import TraceStore
from tests.factories import TRACE_ID_B
from tests.openai_fakes import alt_openai_trace, make_openai_trace


def test_openai_trace_is_not_replayable() -> None:
    trace = make_openai_trace()
    result = replay_trace(trace)
    assert result.status == "not_replayable"
    assert result.token_ids_matched is False
    assert result.bit_identical is False
    assert result.logits_bit_identical is False
    assert result.reason is not None
    assert "top_logprobs" in result.reason
    assert "bit-identical" in result.reason


def test_openai_mode_none_trace_is_not_replayable() -> None:
    trace = make_openai_trace(logits_mode="none")
    result = replay_trace(trace)
    assert result.status == "not_replayable"
    assert result.reason is not None
    assert "not_replayable" in result.reason or "top_logprobs" in result.reason


def test_compare_and_inspect_openai_logprob_traces() -> None:
    trace_a = make_openai_trace()
    trace_b = alt_openai_trace()
    result = compare_traces(trace_a, trace_b)
    first = result.first_divergence
    assert first is not None
    assert first.classification != "raw-logit"
    assert "sampled_token" in first.differences
    assert "raw_logits" not in first.differences
    assert first.reason is not None
    assert "captured scores" in first.reason
    assert "captured logits" not in first.reason
    text = format_inspect_step(trace_a, 0)
    assert "He" in text
    assert "top-k" in text
    assert "logprob" in text
    assert "invented" not in text.lower()


def test_openai_sampling_reason_says_scores_not_logits() -> None:
    trace_a = make_openai_trace()
    event0 = trace_a.events[0].model_copy(
        update={"sampled_token_id": 109, "sampled_token": "No", "sampled_rank": 2}
    )
    trace_b = make_openai_trace(trace_id=TRACE_ID_B).model_copy(
        update={"events": [event0, trace_a.events[1]]}
    )
    result = compare_traces(trace_a, trace_b)
    first = result.first_divergence
    assert first is not None
    assert first.classification == "sampling"
    assert first.reason is not None
    assert "captured scores" in first.reason
    assert "captured logits" not in first.reason


def test_openai_decoding_config_reason_says_scores_not_logits() -> None:
    trace_a = make_openai_trace()
    event0 = trace_a.events[0].model_copy(
        update={"sampled_token_id": 109, "sampled_token": "No", "sampled_rank": 2}
    )
    trace_b = make_openai_trace(
        trace_id=TRACE_ID_B,
        generation=GenerationConfig(temperature=1.5, max_new_tokens=8, do_sample=True),
    ).model_copy(update={"events": [event0, trace_a.events[1]]})
    result = compare_traces(trace_a, trace_b)
    first = result.first_divergence
    assert first is not None
    assert first.classification == "decoding config"
    assert first.reason is not None
    assert first.reason.startswith("captured scores match")
    assert "captured logits" not in first.reason


def test_logprob_only_score_split_is_not_raw_logit() -> None:
    trace_a = make_openai_trace()
    events = []
    for event in make_openai_trace().events:
        top_k = [
            candidate.model_copy(update={"logprob": (candidate.logprob or 0.0) - 1.0})
            for candidate in event.top_k
        ]
        events.append(
            event.model_copy(
                update={
                    "top_k": top_k,
                    "sampled_logprob": (event.sampled_logprob or 0.0) - 1.0,
                }
            )
        )
    trace_b = make_openai_trace().model_copy(update={"events": events})
    result = compare_traces(trace_a, trace_b)
    first = result.first_divergence
    assert first is not None
    assert first.classification == "probability distribution"
    assert "raw_logits" not in first.differences
    assert "probabilities" in first.differences
    assert first.reason is not None
    assert "captured scores" in first.reason
    assert "captured logits" not in first.reason
    assert result.enabling_summary is not None
    assert "captured scores" in result.enabling_summary
    assert "captured logits" not in result.enabling_summary


def test_cli_replay_openai_trace_fail_closed(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    store = TraceStore(tmp_path)
    trace = make_openai_trace()
    store.put(trace)
    code = run(["replay", str(trace.run_metadata.trace_id), "--store", str(tmp_path)])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "not_replayable"
    assert payload["bit_identical"] is False
    assert "top_logprobs" in payload["reason"]


def test_cli_inspect_openai_trace(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    path = tmp_path / "openai.json"
    path.write_text(dumps_json(make_openai_trace()), encoding="utf-8")
    assert run(["inspect", str(path), "--step", "0"]) == 0
    out = capsys.readouterr().out
    assert "sampled token" in out
    assert "top-k" in out
    assert "logprob" in out
    assert run(["topk", str(path)]) == 0
    topk_out = capsys.readouterr().out
    assert "openai top_logprobs are not full-vocab raw logits" in topk_out
    assert "replay is not bit-identical" in topk_out
    assert run(["compare", str(path), str(path)]) == 0
    compare_out = capsys.readouterr().out
    assert "identical" in compare_out
