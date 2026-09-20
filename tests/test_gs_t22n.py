"""GS-T22n grader, summary, and fake-backend guards. No live Qwen download."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from llmfr.adapters.huggingface import PORTFOLIO_DEMO_MODEL_ID, PORTFOLIO_DEMO_MODEL_REVISION

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "gs_t22n_divergence_accuracy.py"
FINDING_MD = REPO / "docs" / "findings" / "gs-t22n.md"
RESULTS_JSON = REPO / "docs" / "findings" / "gs-t22n-results.json"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("llmfr_gs_t22n", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_last_whole_number_and_grades() -> None:
    script = _load_script()
    assert script.extract_last_whole_number("The answer is 9.") == 9
    assert script.extract_last_whole_number("17 sheep, 9 run, so 8 left") == 8
    assert script.extract_last_whole_number("2.0") == 2
    assert script.extract_last_whole_number("about 3.5") is None
    assert script.extract_last_whole_number("no digits") is None
    assert script.grade_output("final 12", 12) == "correct"
    assert script.grade_output("final 11", 12) == "wrong"
    assert script.grade_output("hmm", 12) == "no_answer"


def test_pair_outcome_does_not_invent_ungraded() -> None:
    script = _load_script()
    assert script.pair_outcome("correct", "correct") == "agree_correct"
    assert script.pair_outcome("wrong", "wrong") == "agree_wrong"
    assert script.pair_outcome("correct", "wrong") == "disagree"
    assert script.pair_outcome("no_answer", "correct") == "ungraded"
    assert script.pair_outcome("wrong", "no_answer") == "ungraded"


def test_summarize_null_when_rates_similar() -> None:
    script = _load_script()
    pairs = []
    for index in range(4):
        pairs.append(
            {
                "observed_class": "sampling",
                "grade_a": "correct",
                "grade_b": "wrong",
                "outcome": "disagree",
            }
        )
        pairs.append(
            {
                "observed_class": "decoding config",
                "grade_a": "correct",
                "grade_b": "wrong" if index < 2 else "correct",
                "outcome": "disagree" if index < 2 else "agree_correct",
            }
        )
    # sampling disagree 4/4 = 1.0; decoding 2/4 = 0.5; ratio 2.0 -> positive
    summary = script.summarize_pairs(pairs)
    assert summary["verdict"] == "positive"
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
    null_summary = script.summarize_pairs(similar)
    assert null_summary["verdict"] == "null"
    assert null_summary["by_class"]["sampling"]["disagree_rate"] == 1.0
    empty = script.summarize_pairs(
        [
            {
                "observed_class": None,
                "grade_a": "correct",
                "grade_b": "correct",
                "outcome": "agree_correct",
            }
        ]
    )
    assert empty["verdict"] == "null"
    assert empty["n_identical"] == 1


def test_fake_backend_refuses_docs_bound_paths(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    script = _load_script()
    assert script.main(["--backend", "fake"]) == 2
    err = capsys.readouterr().err
    assert "refusing to overwrite docs/findings" in err
    assert script.main(["--backend", "fake", "--out", str(tmp_path / "traces")]) == 2
    err_results = capsys.readouterr().err
    assert "refusing to overwrite docs/findings" in err_results


def test_fake_backend_scratch_runs_without_writing_finding(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    script = _load_script()
    traces = tmp_path / "traces"
    results = tmp_path / "results.json"
    finding = tmp_path / "finding.md"
    finding_before = FINDING_MD.read_text(encoding="utf-8") if FINDING_MD.is_file() else None
    results_before = RESULTS_JSON.read_text(encoding="utf-8") if RESULTS_JSON.is_file() else None
    code = script.main(
        [
            "--backend",
            "fake",
            "--out",
            str(traces),
            "--results",
            str(results),
            "--finding",
            str(finding),
            "--limit-items",
            "1",
            "--no-save-traces",
        ]
    )
    assert code == 0
    capsys.readouterr()
    payload = json.loads(results.read_text(encoding="utf-8"))
    assert payload["backend"] == "fake"
    assert payload["model"] == "fake-lm"
    assert payload["n_items"] == 1
    assert payload["n_pairs"] == 2
    assert payload["model"] != PORTFOLIO_DEMO_MODEL_ID
    assert not finding.is_file()
    if finding_before is None:
        assert not FINDING_MD.is_file()
    else:
        assert FINDING_MD.read_text(encoding="utf-8") == finding_before
    if results_before is None:
        assert not RESULTS_JSON.is_file()
    else:
        assert RESULTS_JSON.read_text(encoding="utf-8") == results_before


def test_fake_write_docs_refused(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    script = _load_script()
    code = script.main(
        [
            "--backend",
            "fake",
            "--out",
            str(tmp_path / "traces"),
            "--results",
            str(tmp_path / "results.json"),
            "--finding",
            str(tmp_path / "finding.md"),
            "--limit-items",
            "1",
            "--no-save-traces",
            "--write-docs",
        ]
    )
    assert code == 2
    assert "refusing to write finding docs from fake backend" in capsys.readouterr().err


def test_checked_in_finding_matches_results_json() -> None:
    script = _load_script()
    assert RESULTS_JSON.is_file(), "GS-T22n results JSON missing; run the hf experiment"
    assert FINDING_MD.is_file(), "GS-T22n finding markdown missing"
    payload = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    recomputed = script.summarize_pairs(payload["pairs"])
    assert payload["summary"] == recomputed
    assert payload["backend"] == "hf"
    assert payload["model"] == PORTFOLIO_DEMO_MODEL_ID
    assert payload["revision"] == PORTFOLIO_DEMO_MODEL_REVISION
    assert payload["n_items"] == len(script.ITEMS)
    assert payload["max_new_tokens"] == script.MAX_NEW_TOKENS
    assert payload["grading_rule"] == script.GRADING_RULE
    finding = FINDING_MD.read_text(encoding="utf-8")
    rendered = script.render_finding(payload)
    assert finding == rendered
    assert PORTFOLIO_DEMO_MODEL_ID in finding
    assert "tiny-gpt2" in finding
    summary = payload["summary"]
    assert summary["verdict"] == "null"
    sampling = summary["by_class"]["sampling"]
    decoding = summary["by_class"]["decoding config"]
    assert sampling["n_pairs"] == 8
    assert decoding["n_pairs"] == 8
    assert sampling["disagree"] == 3
    assert decoding["disagree"] == 3
    assert sampling["n_gradeable_diverged"] == 8
    assert decoding["n_gradeable_diverged"] == 8
    assert sampling["disagree_rate"] == 0.375
    assert decoding["disagree_rate"] == 0.375
    for row in (sampling, decoding):
        expected = row["disagree"] / row["n_gradeable_diverged"]
        assert row["disagree_rate"] == expected


def test_checked_in_traces_recompare_without_model() -> None:
    from llmfr.compare import compare_traces
    from llmfr.core.schema import load_path

    script = _load_script()
    payload = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    assert payload["pairs"]
    for pair in payload["pairs"]:
        path_a = REPO / pair["path_a"]
        path_b = REPO / pair["path_b"]
        assert path_a.is_file()
        assert path_b.is_file()
        trace_a = load_path(path_a)
        trace_b = load_path(path_b)
        assert trace_a.model.name == PORTFOLIO_DEMO_MODEL_ID
        assert trace_a.model.revision == PORTFOLIO_DEMO_MODEL_REVISION
        compared = compare_traces(trace_a, trace_b)
        first = compared.first_divergence
        observed = None if first is None else first.classification
        assert observed == pair["observed_class"]
        gold = int(pair["gold"])
        assert script.grade_output(trace_a.run_metadata.output_text, gold) == pair["grade_a"]
        assert script.grade_output(trace_b.run_metadata.output_text, gold) == pair["grade_b"]
