"""Live replay tests. Need CPU torch, the `hf` extra, and Hub access."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmfr.adapters.huggingface import DEFAULT_HF_MODEL_ID, HuggingFaceCausalLMAdapter
from llmfr.cli import run
from llmfr.core.schema import GenerationConfig
from llmfr.record import record_generation
from llmfr.replay import BIT_IDENTICAL_CAVEAT, replay_trace
from llmfr.storage import TraceStore

pytest.importorskip("torch")
pytest.importorskip("transformers")

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def adapter() -> HuggingFaceCausalLMAdapter:
    return HuggingFaceCausalLMAdapter(
        DEFAULT_HF_MODEL_ID,
        device="cpu",
        max_visible_tokens=8,
    )


def test_case_a_greedy_tiny_gpt2_reproduces_tokens(
    adapter: HuggingFaceCausalLMAdapter, tmp_path: Path
) -> None:
    store = TraceStore(tmp_path)
    trace = record_generation(
        adapter,
        "Hello",
        generation=GenerationConfig(
            max_new_tokens=4,
            do_sample=False,
            temperature=0.0,
            seed=11,
        ),
        store=store,
        capture_k=5,
        source="test",
    )
    assert trace.model.revision is not None
    assert len(trace.model.revision) == 40

    result = replay_trace(trace, adapter=adapter)
    assert result.status == "reproduced"
    assert result.token_ids_matched is True
    assert result.matched_steps == 4
    assert result.total_steps == 4
    assert result.first_unmatched_step is None
    assert result.recorded_revision == trace.model.revision
    assert BIT_IDENTICAL_CAVEAT in result.notes
    # Same CPU adapter instance: stored sampled/top-k logits should compare equal.
    assert result.logits_bit_identical is True
    assert result.bit_identical is True


def test_case_a_seeded_sampling_tiny_gpt2_reproduces_tokens(
    adapter: HuggingFaceCausalLMAdapter,
) -> None:
    generation = GenerationConfig(
        max_new_tokens=4,
        do_sample=True,
        temperature=1.0,
        seed=42,
    )
    trace = record_generation(adapter, "Hello", generation=generation)
    result = replay_trace(trace, adapter=adapter)
    assert result.status == "reproduced"
    assert result.token_ids_matched is True
    assert result.logits_bit_identical is True


def test_case_a_replay_loads_hub_revision_pin(
    adapter: HuggingFaceCausalLMAdapter, tmp_path: Path
) -> None:
    trace = record_generation(
        adapter,
        "Hello",
        generation=GenerationConfig(max_new_tokens=2, do_sample=False, seed=0),
        store=TraceStore(tmp_path),
    )
    result = replay_trace(trace)
    assert result.status == "reproduced"
    assert result.token_ids_matched is True
    assert result.replayed_revision == trace.model.revision
    assert result.recorded_revision == trace.model.revision


def test_cli_replay_tiny_gpt2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    record_code = run(
        [
            "record",
            "Hello",
            "--store",
            str(tmp_path),
            "--max-new-tokens",
            "2",
            "--greedy",
            "--seed",
            "0",
            "--max-visible-tokens",
            "8",
        ]
    )
    assert record_code == 0
    trace_id = capsys.readouterr().out.strip()
    replay_code = run(["replay", trace_id, "--store", str(tmp_path)])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert replay_code == 0
    assert payload["status"] == "reproduced"
    assert payload["token_ids_matched"] is True
    assert payload["trace_id"] == trace_id
    assert BIT_IDENTICAL_CAVEAT in payload["notes"]
    assert payload["logits_bit_identical"] is True
