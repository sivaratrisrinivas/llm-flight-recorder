"""Batch prompt files and ``llmfr record --prompts``. Fake adapters only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest import CaptureFixture

from llmfr.adapters.openai import OpenAIChatAdapter
from llmfr.cli import run
from llmfr.core.schema import GenerationConfig
from llmfr.privacy import REDACTED
from llmfr.record.batch import BatchPromptError, load_prompt_file, record_prompt_batch
from llmfr.storage import TraceStore
from tests.fakes import FakeCausalLMAdapter
from tests.openai_fakes import (
    FakeEncoding,
    FakeOpenAIClient,
    hello_logprob_tokens,
    make_chat_response,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_load_text_file_skips_blank_lines(tmp_path: Path) -> None:
    path = _write(tmp_path / "prompts.txt", "Hello\n\n  How many sheep?  \n\n")
    jobs = load_prompt_file(path)
    assert [job.prompt for job in jobs] == ["Hello", "How many sheep?"]
    assert [job.line for job in jobs] == [1, 3]


def test_load_jsonl_objects_and_strings(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "prompts.jsonl",
        "\n".join(
            [
                json.dumps({"prompt": "Hello", "id": "a", "seed": 1}),
                '"How many sheep?"',
                "",
                json.dumps(
                    {
                        "prompt": "Greedy",
                        "greedy": True,
                        "temperature": 0.7,
                        "max_new_tokens": 4,
                        "tags": {"case": "demo"},
                    }
                ),
            ]
        )
        + "\n",
    )
    jobs = load_prompt_file(path)
    assert len(jobs) == 3
    assert jobs[0].prompt == "Hello"
    assert jobs[0].id == "a"
    assert jobs[0].seed == 1
    assert jobs[0].line == 1
    assert jobs[1].prompt == "How many sheep?"
    assert jobs[1].id is None
    assert jobs[1].line == 2
    assert jobs[2].greedy is True
    assert jobs[2].temperature == 0.7
    assert jobs[2].max_new_tokens == 4
    assert jobs[2].tags == {"case": "demo"}


def test_load_jsonl_gold_integer(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "prompts.jsonl",
        json.dumps({"prompt": "What is 2 plus 2?", "id": "two_plus_two", "gold": 4}) + "\n",
    )
    jobs = load_prompt_file(path)
    assert jobs[0].gold == 4
    assert jobs[0].id == "two_plus_two"


def test_load_jsonl_rejects_non_integer_gold(tmp_path: Path) -> None:
    path = _write(tmp_path / "prompts.jsonl", json.dumps({"prompt": "Hello", "gold": 4.5}) + "\n")
    with pytest.raises(BatchPromptError, match="gold must be an integer"):
        load_prompt_file(path)


def test_load_json_array(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "prompts.json",
        json.dumps(["Hello", {"prompt": "How many sheep?", "id": "sheep"}]),
    )
    jobs = load_prompt_file(path)
    assert [job.prompt for job in jobs] == ["Hello", "How many sheep?"]
    assert jobs[1].id == "sheep"


def test_load_rejects_empty_file(tmp_path: Path) -> None:
    path = _write(tmp_path / "empty.txt", "\n\n")
    with pytest.raises(BatchPromptError, match="prompt list is empty"):
        load_prompt_file(path)


def test_load_rejects_unknown_jsonl_keys(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "prompts.jsonl",
        json.dumps({"prompt": "Hello", "provider": "openai", "api_key": "sk-secret"}),
    )
    with pytest.raises(BatchPromptError, match="Extra inputs are not permitted") as caught:
        load_prompt_file(path)
    assert "sk-secret" not in str(caught.value)


def test_load_rejects_corrupt_jsonl(tmp_path: Path) -> None:
    path = _write(tmp_path / "prompts.jsonl", '{"prompt":"Hello"}\n{not-json\n')
    with pytest.raises(BatchPromptError, match=r":2: corrupt JSON"):
        load_prompt_file(path)


def test_load_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.jsonl"
    with pytest.raises(FileNotFoundError, match="prompt file not found"):
        load_prompt_file(missing)


def test_record_prompt_batch_persists_and_tags(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "prompts.jsonl",
        json.dumps({"prompt": "Hello", "id": "a"})
        + "\n"
        + json.dumps({"prompt": "How many sheep?", "seed": 9})
        + "\n",
    )
    store = TraceStore(tmp_path / "store")
    streamed: list[str] = []
    traces = record_prompt_batch(
        FakeCausalLMAdapter(prompt_ids=[1, 2]),
        load_prompt_file(path),
        generation=GenerationConfig(max_new_tokens=1, do_sample=False, seed=1),
        store=store,
        source="test",
        on_recorded=lambda trace: streamed.append(str(trace.run_metadata.trace_id)),
    )
    assert len(traces) == 2
    assert streamed == [str(trace.run_metadata.trace_id) for trace in traces]
    assert [trace.run_metadata.prompt for trace in traces] == ["Hello", "How many sheep?"]
    assert traces[0].run_metadata.tags["id"] == "a"
    assert traces[0].run_metadata.tags["batch_index"] == "0"
    assert traces[1].run_metadata.tags["batch_index"] == "1"
    assert traces[0].generation_config.seed == 1
    assert traces[1].generation_config.seed == 9
    assert {entry.trace_id for entry in store.list()} == {
        str(traces[0].run_metadata.trace_id),
        str(traces[1].run_metadata.trace_id),
    }


def test_record_prompt_batch_stores_gold_tag(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "prompts.jsonl",
        json.dumps({"prompt": "What is 2 plus 2?", "gold": 4}) + "\n",
    )
    traces = record_prompt_batch(
        FakeCausalLMAdapter(prompt_ids=[1, 2]),
        load_prompt_file(path),
        generation=GenerationConfig(max_new_tokens=1, do_sample=False),
        persist=False,
    )
    assert traces[0].run_metadata.tags["gold"] == "4"


def test_record_prompt_batch_openai_fail_closed_keeps_prior(tmp_path: Path) -> None:
    path = _write(tmp_path / "prompts.jsonl", '{"prompt":"Hi"}\n{"prompt":"Yo"}\n')
    client = FakeOpenAIClient(
        [
            make_chat_response(hello_logprob_tokens()),
            make_chat_response(hello_logprob_tokens(), output_text="Hello", include_logprobs=False),
        ]
    )
    store = TraceStore(tmp_path / "store")
    streamed: list[str] = []
    with pytest.raises(BatchPromptError, match="per-token logprob content"):
        record_prompt_batch(
            OpenAIChatAdapter("gpt-4o-mini", client=client, encoding=FakeEncoding()),
            load_prompt_file(path),
            generation=GenerationConfig(max_new_tokens=2, do_sample=False),
            store=store,
            on_recorded=lambda trace: streamed.append(str(trace.run_metadata.trace_id)),
        )
    listed = store.list()
    assert len(listed) == 1
    assert streamed == [listed[0].trace_id]
    loaded = store.get(str(listed[0].trace_id))
    assert loaded.run_metadata.prompt == "Hi"
    assert loaded.run_metadata.logits.mode == "topk"


def test_cli_record_prompts_prints_trace_ids(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "llmfr.cli._build_hf_adapter",
        lambda **_kwargs: FakeCausalLMAdapter(prompt_ids=[1, 2]),
    )
    path = _write(tmp_path / "prompts.jsonl", '{"prompt":"Hello"}\n{"prompt":"Sheep"}\n')
    store = tmp_path / "store"
    code = run(
        [
            "record",
            "--prompts",
            str(path),
            "--store",
            str(store),
            "--max-new-tokens",
            "1",
            "--greedy",
        ]
    )
    assert code == 0
    ids = capsys.readouterr().out.strip().splitlines()
    assert len(ids) == 2
    loaded = [TraceStore(store).get(trace_id) for trace_id in ids]
    assert [trace.run_metadata.prompt for trace in loaded] == ["Hello", "Sheep"]
    assert loaded[0].run_metadata.source == "cli"
    assert loaded[0].run_metadata.tags["batch_index"] == "0"


def test_cli_record_prompts_redact_and_no_persist(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "llmfr.cli._build_hf_adapter",
        lambda **_kwargs: FakeCausalLMAdapter(prompt_ids=[1, 2]),
    )
    path = _write(tmp_path / "secret.txt", "secret prompt\nsecond secret\n")
    store = tmp_path / "store"
    assert (
        run(
            [
                "record",
                "--prompts",
                str(path),
                "--store",
                str(store),
                "--max-new-tokens",
                "1",
                "--greedy",
                "--no-persist",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert not (store / "index.sqlite").exists()

    assert (
        run(
            [
                "record",
                "--prompts",
                str(path),
                "--store",
                str(store),
                "--max-new-tokens",
                "1",
                "--greedy",
                "--redact",
            ]
        )
        == 0
    )
    ids = capsys.readouterr().out.strip().splitlines()
    loaded = TraceStore(store).get(ids[0])
    assert loaded.run_metadata.prompt == REDACTED
    assert loaded.events[0].sampled_token == REDACTED
    assert "secret" not in loaded.run_metadata.prompt


def test_cli_record_prompts_openai_mocked(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeOpenAIClient(
        [
            make_chat_response(hello_logprob_tokens()),
            make_chat_response(hello_logprob_tokens()),
        ]
    )
    monkeypatch.setattr(
        "llmfr.cli._build_openai_adapter",
        lambda **_kwargs: OpenAIChatAdapter("gpt-4o-mini", client=client, encoding=FakeEncoding()),
    )
    path = _write(tmp_path / "prompts.jsonl", '{"prompt":"Hi"}\n{"prompt":"Yo"}\n')
    code = run(
        [
            "record",
            "--prompts",
            str(path),
            "--provider",
            "openai",
            "--store",
            str(tmp_path / "store"),
            "--max-new-tokens",
            "2",
            "--greedy",
        ]
    )
    assert code == 0
    ids = capsys.readouterr().out.strip().splitlines()
    assert len(ids) == 2
    loaded = TraceStore(tmp_path / "store").get(ids[0])
    assert loaded.model.provider == "openai"
    assert loaded.run_metadata.logits.mode == "topk"
    assert all(event.sampled_logit is None for event in loaded.events)


def test_cli_record_prompts_openai_prefix_and_bare_gpt(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    openai_built: list[str] = []
    hf_built: list[str] = []
    path = _write(tmp_path / "prompts.txt", "Hi\n")

    def _openai(*, model_id: str | None, max_visible_tokens: int | None, capture_k: int) -> object:
        openai_built.append(str(model_id))
        return OpenAIChatAdapter(
            "gpt-4o-mini",
            client=FakeOpenAIClient([make_chat_response(hello_logprob_tokens())]),
            encoding=FakeEncoding(),
        )

    def _hf(
        *, model_id: str | None, max_visible_tokens: int | None, revision: str | None = None
    ) -> FakeCausalLMAdapter:
        hf_built.append(str(model_id))
        return FakeCausalLMAdapter(prompt_ids=[1, 2])

    monkeypatch.setattr("llmfr.cli._build_openai_adapter", _openai)
    monkeypatch.setattr("llmfr.cli._build_hf_adapter", _hf)
    assert (
        run(
            [
                "record",
                "--prompts",
                str(path),
                "--model",
                "openai:gpt-4o-mini",
                "--store",
                str(tmp_path / "oa"),
                "--max-new-tokens",
                "1",
                "--greedy",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert openai_built == ["openai:gpt-4o-mini"]
    assert hf_built == []

    openai_built.clear()
    hf_built.clear()
    assert (
        run(
            [
                "record",
                "--prompts",
                str(path),
                "--model",
                "gpt-4o-mini",
                "--store",
                str(tmp_path / "hf"),
                "--max-new-tokens",
                "1",
                "--greedy",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert openai_built == []
    assert hf_built == ["gpt-4o-mini"]


def test_cli_record_prompts_openai_omitted_logprobs_fail_closed(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeOpenAIClient(
        [make_chat_response(hello_logprob_tokens(), output_text="Hello", include_logprobs=False)]
    )
    monkeypatch.setattr(
        "llmfr.cli._build_openai_adapter",
        lambda **_kwargs: OpenAIChatAdapter("gpt-4o-mini", client=client, encoding=FakeEncoding()),
    )
    path = _write(tmp_path / "prompts.jsonl", '{"prompt":"Hi"}\n')
    store = tmp_path / "store"
    code = run(
        [
            "record",
            "--prompts",
            str(path),
            "--provider",
            "openai",
            "--store",
            str(store),
            "--max-new-tokens",
            "2",
            "--greedy",
        ]
    )
    assert code == 1
    captured = capsys.readouterr()
    assert captured.out.strip() == ""
    assert "error:" in captured.err
    assert "per-token logprob content" in captured.err
    assert "Traceback" not in captured.err
    assert TraceStore(store).list() == []


def test_cli_record_prompts_streams_ids_before_mid_batch_failure(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeOpenAIClient(
        [
            make_chat_response(hello_logprob_tokens()),
            make_chat_response(hello_logprob_tokens(), output_text="Hello", include_logprobs=False),
        ]
    )
    monkeypatch.setattr(
        "llmfr.cli._build_openai_adapter",
        lambda **_kwargs: OpenAIChatAdapter("gpt-4o-mini", client=client, encoding=FakeEncoding()),
    )
    path = _write(tmp_path / "prompts.jsonl", '{"prompt":"Hi"}\n{"prompt":"Yo"}\n')
    store = tmp_path / "store"
    code = run(
        [
            "record",
            "--prompts",
            str(path),
            "--provider",
            "openai",
            "--store",
            str(store),
            "--max-new-tokens",
            "2",
            "--greedy",
        ]
    )
    assert code == 1
    captured = capsys.readouterr()
    ids = captured.out.strip().splitlines()
    assert len(ids) == 1
    loaded = TraceStore(store).get(ids[0])
    assert loaded.run_metadata.prompt == "Hi"
    assert loaded.run_metadata.logits.mode == "topk"
    assert "error:" in captured.err
    assert "per-token logprob content" in captured.err
    assert "Traceback" not in captured.err
    assert [entry.trace_id for entry in TraceStore(store).list()] == ids


def test_cli_record_prompts_mutually_exclusive(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    path = _write(tmp_path / "prompts.txt", "Hello\n")
    code = run(["record", "Hello", "--prompts", str(path)])
    assert code == 2
    err = capsys.readouterr().err
    assert "error:" in err
    assert "mutually exclusive" in err
    assert "Traceback" not in err


def test_cli_record_prompts_missing_file(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    missing = tmp_path / "nope.jsonl"
    code = run(["record", "--prompts", str(missing), "--store", str(tmp_path / "store")])
    assert code == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "prompt file not found" in err
    assert "Traceback" not in err
