"""Live recorder tests. Need CPU torch, the `hf` extra, and Hub access."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from llmfr.adapters.huggingface import DEFAULT_HF_MODEL_ID, HuggingFaceCausalLMAdapter
from llmfr.cli import run
from llmfr.core.schema import GenerationConfig
from llmfr.record import RecordableAdapter, record_generation
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


def test_hf_recorder_pipeline_and_store(
    adapter: HuggingFaceCausalLMAdapter, tmp_path: Path
) -> None:
    assert isinstance(adapter, RecordableAdapter)
    assert adapter.capabilities.supports_seed is True
    store = TraceStore(tmp_path)
    prompt = "Hello"
    trace = record_generation(
        adapter,
        prompt,
        generation=GenerationConfig(
            max_new_tokens=3,
            do_sample=False,
            temperature=0.0,
            seed=11,
        ),
        store=store,
        capture_k=5,
        source="test",
    )
    prompt_ids = adapter.encode(prompt)
    assert trace.run_metadata.prompt_token_ids == prompt_ids
    assert len(trace.events) == 3
    sampled: list[int] = []
    for index, event in enumerate(trace.events):
        assert event.step == index
        assert event.full_history.token_ids == prompt_ids + sampled
        assert event.sampled_logit is not None
        assert event.top_k
        assert event.top_k[0].logit is not None
        if not event.model_visible_context.truncated:
            assert event.model_visible_context.token_ids == event.full_history.token_ids
        sampled.append(event.sampled_token_id)

    loaded = store.get(str(trace.run_metadata.trace_id))
    assert loaded == trace
    assert loaded.generation_config.seed == 11
    assert loaded.model.name == DEFAULT_HF_MODEL_ID
    assert loaded.model.revision is not None
    assert len(loaded.model.revision) == 40


def test_hf_history_not_equal_visible_when_windowed(tmp_path: Path) -> None:
    adapter = HuggingFaceCausalLMAdapter(
        DEFAULT_HF_MODEL_ID,
        device="cpu",
        max_visible_tokens=3,
    )
    prompt = "Hello world this is a longer windowing prompt"
    prompt_ids = adapter.encode(prompt)
    assert len(prompt_ids) > 3
    trace = record_generation(
        adapter,
        prompt,
        generation=GenerationConfig(max_new_tokens=2, do_sample=False),
        store=TraceStore(tmp_path),
    )
    for event in trace.events:
        assert event.full_history.token_ids != event.model_visible_context.token_ids
        assert event.model_visible_context.truncated is True
        assert event.model_visible_context.token_ids == event.full_history.token_ids[-3:]
        assert len(event.model_visible_context.token_ids) < len(event.full_history.token_ids)


def test_hf_seed_reproduces_sampled_tokens(adapter: HuggingFaceCausalLMAdapter) -> None:
    generation = GenerationConfig(
        max_new_tokens=4,
        do_sample=True,
        temperature=1.0,
        seed=42,
    )
    first = record_generation(adapter, "Hello", generation=generation)
    second = record_generation(adapter, "Hello", generation=generation)
    other = record_generation(
        adapter,
        "Hello",
        generation=generation.model_copy(update={"seed": 43}),
    )
    ids_first = [event.sampled_token_id for event in first.events]
    ids_second = [event.sampled_token_id for event in second.events]
    ids_other = [event.sampled_token_id for event in other.events]
    assert ids_first == ids_second
    assert first.generation_config.seed == 42
    # Flat-ish tiny-gpt2 plus temperature=1 usually diverges; if it does not,
    # the fake-adapter tests still prove seed changes the categorical draw.
    if ids_first == ids_other:
        pytest.skip("tiny-gpt2 sampled the same tokens for two seeds")


def test_cli_record_prints_trace_id(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = run(
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
        ]
    )
    assert code == 0
    out = capsys.readouterr().out.strip()
    UUID(out)
    loaded = TraceStore(tmp_path).get(out)
    assert len(loaded.events) == 2
    assert loaded.run_metadata.source == "cli"
    assert loaded.generation_config.do_sample is False
