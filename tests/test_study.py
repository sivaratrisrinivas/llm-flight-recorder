"""llmfr study: graded sampling vs decoding-config. Fake adapters only."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from pytest import CaptureFixture

from llmfr.adapters.openai import OpenAIChatAdapter
from llmfr.cli import build_parser, run
from llmfr.privacy import REDACTED
from llmfr.record.batch import BatchPromptError, load_prompt_file
from llmfr.storage import TraceStore
from llmfr.study import (
    GRADING_RULE,
    extract_last_whole_number,
    format_study_report,
    grade_output,
    pair_outcome,
    parse_float_pair,
    parse_int_pair,
    require_study_jobs,
    run_study,
    summarize_pairs,
)
from tests.fakes import FakeCausalLMAdapter
from tests.openai_fakes import (
    FakeEncoding,
    FakeOpenAIClient,
    hello_logprob_tokens,
    make_chat_response,
)

REPO = Path(__file__).resolve().parents[1]
EXAMPLE_PROMPTS = REPO / "examples" / "study" / "prompts.jsonl"


class DigitFake(FakeCausalLMAdapter):
    """Decode token ids as digits so the last-whole-number grader can run."""

    def decode_token(self, token_id: int) -> str:
        return str(int(token_id))


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _study_jsonl(path: Path, *rows: dict[str, object]) -> Path:
    return _write(path, "".join(json.dumps(row) + "\n" for row in rows))


def test_extract_last_whole_number_and_grades() -> None:
    assert extract_last_whole_number("The answer is 9.") == 9
    assert extract_last_whole_number("17 sheep, 9 run, so 8 left") == 8
    assert extract_last_whole_number("2.0") == 2
    assert extract_last_whole_number("about 3.5") is None
    assert extract_last_whole_number("no digits") is None
    assert grade_output("final 12", 12) == "correct"
    assert grade_output("final 11", 12) == "wrong"
    assert grade_output("hmm", 12) == "no_answer"
    assert pair_outcome("correct", "wrong") == "disagree"
    assert pair_outcome("no_answer", "correct") == "ungraded"


def test_parse_pairs_and_usage_shape() -> None:
    assert parse_int_pair("1,2", name="--seeds") == (1, 2)
    assert parse_float_pair("0.7, 1.2", name="--temperatures") == (0.7, 1.2)
    with pytest.raises(ValueError, match="exactly two"):
        parse_int_pair("1", name="--seeds")
    with pytest.raises(ValueError, match="exactly two"):
        parse_float_pair("a,b", name="--temperatures")


def test_example_prompts_have_gold() -> None:
    jobs = load_prompt_file(EXAMPLE_PROMPTS)
    require_study_jobs(jobs)
    assert len(jobs) == 8
    assert {job.id for job in jobs} == {
        "sheep_trick",
        "seven_plus_five",
        "twelve_minus_four",
        "three_times_six",
        "cows_sold",
        "hundred_div_four",
        "apples_left",
        "two_plus_two",
    }
    assert all(job.gold is not None for job in jobs)


def test_require_study_jobs_needs_gold(tmp_path: Path) -> None:
    path = _write(tmp_path / "prompts.jsonl", '{"prompt":"What is 2 plus 2?"}\n')
    jobs = load_prompt_file(path)
    with pytest.raises(BatchPromptError, match="study requires integer gold"):
        require_study_jobs(jobs)


def test_require_study_jobs_rejects_per_item_split_fields(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "prompts.jsonl",
        json.dumps({"prompt": "What is 2 plus 2?", "gold": 4, "seed": 9}) + "\n",
    )
    jobs = load_prompt_file(path)
    with pytest.raises(BatchPromptError, match="CLI splits"):
        require_study_jobs(jobs)


def test_run_study_records_pairs_and_grades(tmp_path: Path) -> None:
    path = _study_jsonl(
        tmp_path / "prompts.jsonl",
        {"prompt": "What is 2 plus 2?", "gold": 2, "id": "two_plus_two"},
    )
    store = TraceStore(tmp_path / "store")
    report = run_study(
        DigitFake(prompt_ids=[1], logits=(0.0, 0.0, 5.0, 0.0)),
        load_prompt_file(path),
        max_new_tokens=1,
        store=store,
        persist=True,
        source="test",
        progress=io.StringIO(),
    )
    assert report.n_items == 1
    assert report.n_pairs == 2
    assert report.grading_rule == GRADING_RULE
    assert len(store.list()) == 4
    assert "verdict" not in report.summary
    assert report.summary["n_no_first_divergence"] == 2
    table = format_study_report(report)
    assert "sampling" in table
    assert "decoding config" in table
    assert "disagree rate" in table
    assert GRADING_RULE in table
    assert "pairs with no first event divergence: 2" in table
    assert "identical pairs:" not in table
    for pair in report.pairs:
        assert pair["gold"] == 2
        assert pair["extracted_a"] == 2
        assert pair["grade_a"] == "correct"
        assert pair["intended_kind"] in ("sampling", "decoding_config")
        assert pair["output_a"]
        assert pair["observed_class"] is None
        assert pair["no_first_divergence"] is True
        assert pair["identical"] is False


def test_cli_study_prints_table(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "llmfr.cli._build_hf_adapter",
        lambda **_kwargs: DigitFake(prompt_ids=[1], logits=(0.0, 0.0, 5.0, 0.0)),
    )
    path = _study_jsonl(
        tmp_path / "prompts.jsonl",
        {"prompt": "What is 2 plus 2?", "gold": 2, "id": "two_plus_two"},
    )
    store = tmp_path / "store"
    code = run(
        [
            "study",
            str(path),
            "--store",
            str(store),
            "--max-new-tokens",
            "1",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "study n_items=1 n_pairs=2" in captured.out
    assert "two_plus_two" in captured.out
    assert "grading:" in captured.out
    assert "No LLM judge" in captured.out
    assert "no first divergence" in captured.out
    assert "identical pairs:" not in captured.out
    assert len(TraceStore(store).list()) == 4
    assert "recording 1/2" in captured.err


def test_cli_study_json_and_no_persist(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "llmfr.cli._build_hf_adapter",
        lambda **_kwargs: DigitFake(prompt_ids=[1], logits=(0.0, 0.0, 5.0, 0.0)),
    )
    path = _study_jsonl(
        tmp_path / "prompts.jsonl",
        {"prompt": "What is 2 plus 2?", "gold": 2},
    )
    store = tmp_path / "store"
    code = run(
        [
            "study",
            str(path),
            "--store",
            str(store),
            "--max-new-tokens",
            "1",
            "--no-persist",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_items"] == 1
    assert payload["n_pairs"] == 2
    assert payload["grading_rule"] == GRADING_RULE
    assert payload["sampling_seeds"] == [1, 2]
    assert payload["decoding_temperatures"] == [0.7, 1.2]
    assert "verdict" not in payload["summary"]
    assert "n_identical" not in payload["summary"]
    assert payload["summary"]["n_no_first_divergence"] == 2
    assert not (store / "index.sqlite").exists()


def test_cli_study_missing_gold(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "llmfr.cli._build_hf_adapter",
        lambda **_kwargs: DigitFake(prompt_ids=[1]),
    )
    path = _write(tmp_path / "prompts.jsonl", '{"prompt":"Hello"}\n')
    code = run(["study", str(path), "--store", str(tmp_path / "store")])
    assert code == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "study requires integer gold" in err
    assert "Traceback" not in err


def test_cli_study_text_file_has_no_gold(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "llmfr.cli._build_hf_adapter",
        lambda **_kwargs: DigitFake(prompt_ids=[1]),
    )
    path = _write(tmp_path / "prompts.txt", "What is 2 plus 2?\n")
    code = run(["study", str(path), "--store", str(tmp_path / "store")])
    assert code == 1
    assert "study requires integer gold" in capsys.readouterr().err


def test_cli_study_invalid_seeds_is_usage(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    path = _study_jsonl(tmp_path / "prompts.jsonl", {"prompt": "What is 2 plus 2?", "gold": 4})
    code = run(["study", str(path), "--seeds", "1"])
    assert code == 2
    err = capsys.readouterr().err
    assert "error:" in err
    assert "--seeds" in err
    assert "Traceback" not in err


def test_cli_study_missing_file(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    missing = tmp_path / "nope.jsonl"
    code = run(["study", str(missing), "--store", str(tmp_path / "store")])
    assert code == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "prompt file not found" in err
    assert "Traceback" not in err


def test_cli_study_openai_mocked(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeOpenAIClient([make_chat_response(hello_logprob_tokens()) for _ in range(4)])
    monkeypatch.setattr(
        "llmfr.cli._build_openai_adapter",
        lambda **_kwargs: OpenAIChatAdapter("gpt-4o-mini", client=client, encoding=FakeEncoding()),
    )
    path = _study_jsonl(tmp_path / "prompts.jsonl", {"prompt": "Hi", "gold": 4, "id": "hi"})
    code = run(
        [
            "study",
            str(path),
            "--provider",
            "openai",
            "--store",
            str(tmp_path / "store"),
            "--max-new-tokens",
            "2",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "hi" in out
    assert "ungraded" in out
    loaded = TraceStore(tmp_path / "store").list()
    assert len(loaded) == 4
    trace = TraceStore(tmp_path / "store").get(str(loaded[0].trace_id))
    assert trace.model.provider == "openai"
    assert trace.run_metadata.logits.mode == "topk"


def test_cli_study_openai_prefix_and_bare_gpt(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    openai_built: list[str] = []
    hf_built: list[str] = []
    path = _study_jsonl(tmp_path / "prompts.jsonl", {"prompt": "Hi", "gold": 1})

    def _openai(*, model_id: str | None, max_visible_tokens: int | None, capture_k: int) -> object:
        openai_built.append(str(model_id))
        return OpenAIChatAdapter(
            "gpt-4o-mini",
            client=FakeOpenAIClient([make_chat_response(hello_logprob_tokens()) for _ in range(4)]),
            encoding=FakeEncoding(),
        )

    def _hf(
        *, model_id: str | None, max_visible_tokens: int | None, revision: str | None = None
    ) -> DigitFake:
        hf_built.append(str(model_id))
        return DigitFake(prompt_ids=[1], logits=(0.0, 0.0, 5.0, 0.0))

    monkeypatch.setattr("llmfr.cli._build_openai_adapter", _openai)
    monkeypatch.setattr("llmfr.cli._build_hf_adapter", _hf)
    assert (
        run(
            [
                "study",
                str(path),
                "--model",
                "openai:gpt-4o-mini",
                "--store",
                str(tmp_path / "oa"),
                "--max-new-tokens",
                "1",
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
                "study",
                str(path),
                "--model",
                "gpt-4o-mini",
                "--store",
                str(tmp_path / "hf"),
                "--max-new-tokens",
                "1",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert openai_built == []
    assert hf_built == ["gpt-4o-mini"]


def test_cli_study_openai_omitted_logprobs_fail_closed(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeOpenAIClient(
        [make_chat_response(hello_logprob_tokens(), output_text="Hello", include_logprobs=False)]
    )
    monkeypatch.setattr(
        "llmfr.cli._build_openai_adapter",
        lambda **_kwargs: OpenAIChatAdapter("gpt-4o-mini", client=client, encoding=FakeEncoding()),
    )
    path = _study_jsonl(tmp_path / "prompts.jsonl", {"prompt": "Hi", "gold": 4})
    store = tmp_path / "store"
    code = run(
        [
            "study",
            str(path),
            "--provider",
            "openai",
            "--store",
            str(store),
            "--max-new-tokens",
            "2",
        ]
    )
    assert code == 1
    captured = capsys.readouterr()
    assert captured.out.strip() == ""
    assert "error:" in captured.err
    assert "per-token logprob content" in captured.err
    assert "Traceback" not in captured.err
    assert TraceStore(store).list() == []


def test_cli_study_redact_grades_then_hides_store(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "llmfr.cli._build_hf_adapter",
        lambda **_kwargs: DigitFake(prompt_ids=[1], logits=(0.0, 0.0, 5.0, 0.0)),
    )
    path = _study_jsonl(
        tmp_path / "secret.jsonl",
        {"prompt": "secret prompt 2 plus 2", "gold": 2, "id": "secret"},
    )
    store = tmp_path / "store"
    code = run(
        [
            "study",
            str(path),
            "--store",
            str(store),
            "--max-new-tokens",
            "1",
            "--redact",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pairs"][0]["output_a"] is None
    assert payload["pairs"][0]["grade_a"] == "correct"
    loaded = TraceStore(store).get(payload["pairs"][0]["trace_a"])
    assert loaded.run_metadata.prompt == REDACTED
    assert "secret" not in loaded.run_metadata.prompt


def test_cli_study_is_listed(capsys: CaptureFixture[str]) -> None:
    help_text = build_parser().format_help()
    assert "study" in help_text
    assert run(["study", "--help"]) == 0
    study_help = capsys.readouterr().out
    assert "gold" in study_help
    assert "No LLM judge" in study_help


def test_summarize_has_no_verdict_and_excludes_ungraded_from_wrong_rate() -> None:
    similar = []
    for _ in range(4):
        similar.append(
            {
                "observed_class": "sampling",
                "grade_a": "correct",
                "grade_b": "wrong",
                "outcome": "disagree",
            }
        )
        similar.append(
            {
                "observed_class": "decoding config",
                "grade_a": "correct",
                "grade_b": "wrong",
                "outcome": "disagree",
            }
        )
    summary = summarize_pairs(similar)
    assert "verdict" not in summary
    assert summary["by_class"]["sampling"]["disagree_rate"] == 1.0
    mixed = [
        {
            "observed_class": "sampling",
            "grade_a": "wrong",
            "grade_b": "correct",
            "outcome": "disagree",
        },
        {
            "observed_class": "sampling",
            "grade_a": "wrong",
            "grade_b": "no_answer",
            "outcome": "ungraded",
        },
        {
            "observed_class": None,
            "grade_a": "correct",
            "grade_b": "correct",
            "outcome": "agree_correct",
            "identical": False,
        },
    ]
    mixed_summary = summarize_pairs(mixed)
    sampling = mixed_summary["by_class"]["sampling"]
    assert mixed_summary["n_no_first_divergence"] == 1
    assert sampling["ungraded"] == 1
    assert sampling["n_gradeable_diverged"] == 1
    assert sampling["n_wrong_traces"] == 1
    assert sampling["n_gradeable_traces"] == 2
    assert sampling["wrong_answer_rate"] == 0.5
    assert sampling["disagree_rate"] == 1.0
