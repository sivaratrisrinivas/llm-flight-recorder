"""GS-T22q packer, prompts, and checked-in Qwen fixtures. No live 0.5B download."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

from llmfr.adapters.huggingface import (
    DEFAULT_HF_MODEL_ID,
    PORTFOLIO_DEMO_MODEL_ID,
    PORTFOLIO_DEMO_MODEL_REVISION,
)
from llmfr.compare import compare_traces
from llmfr.core.schema import load_path
from llmfr.record.batch import load_prompt_file
from llmfr.study import descriptive_verdict, grade_output, require_study_jobs, summarize_pairs

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "gs_t22q_pack_study.py"
PROMPTS = REPO / "examples" / "findings" / "gs-t22q" / "prompts.jsonl"
FINDING_MD = REPO / "docs" / "findings" / "gs-t22q.md"
RESULTS_JSON = REPO / "docs" / "findings" / "gs-t22q-results.json"
README = REPO / "README.md"
T22N_TRACES = REPO / "examples" / "findings" / "gs-t22n"
TARGET_N = 30
FINDING_MD_LINK = "docs/findings/gs-t22q.md"
FINDING_JSON_LINK = "docs/findings/gs-t22q-results.json"


def _assert_readme_findings(readme: str, payload: dict) -> None:
    headings = [line[3:].strip() for line in readme.splitlines() if line.startswith("## ")]
    assert "Findings" in headings
    finding_start = readme.index("## Findings")
    finding = readme[finding_start:]
    next_heading = finding.find("\n## ", 1)
    if next_heading != -1:
        finding = finding[:next_heading]
    sampling = payload["summary"]["by_class"]["sampling"]
    decoding = payload["summary"]["by_class"]["decoding config"]
    n_items = int(payload["n_items"])
    samp_disagree = int(sampling["disagree"])
    dec_disagree = int(decoding["disagree"])
    verdict = str(payload["summary"]["descriptive_verdict"])
    assert n_items == TARGET_N
    assert f"N={n_items}" in finding
    assert str(payload["model"]) in finding
    assert f"{samp_disagree}/{n_items}" in finding
    assert f"{dec_disagree}/{n_items}" in finding
    assert verdict == "null"
    assert "descriptive" in finding
    assert "null" in finding
    assert FINDING_MD_LINK in finding
    assert FINDING_JSON_LINK in finding
    assert "img.shields.io/badge" in finding
    assert "classDef" in finding
    assert "fill:#" in finding
    assert "bgcolor" not in finding.lower()
    assert "<td" not in finding.lower()
    assert "Pair detail" not in finding
    assert "chairs_remain" not in finding
    assert "outcome `" not in finding
    assert "Changelog" not in finding
    essentials = readme[readme.index("## Essentials") : finding_start]
    assert "Measured finding" not in essentials
    assert FINDING_JSON_LINK not in essentials


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("llmfr_gs_t22q", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prompts_are_thirty_gradeable_integer_items() -> None:
    jobs = load_prompt_file(PROMPTS)
    require_study_jobs(jobs)
    assert len(jobs) == TARGET_N
    ids = [job.id for job in jobs]
    assert all(item_id is not None for item_id in ids)
    assert len(set(ids)) == TARGET_N
    assert all(job.gold is not None for job in jobs)
    assert all("Think step by step, then give the final number." in job.prompt for job in jobs)
    assert jobs[0].id == "sheep_trick"
    assert jobs[0].gold == 9


def test_require_portfolio_qwen_rejects_tiny_gpt2_and_llama() -> None:
    script = _load_script()
    with pytest.raises(ValueError, match="no silent Llama"):
        script.require_portfolio_qwen({"model": DEFAULT_HF_MODEL_ID, "revision": "abc"})
    with pytest.raises(ValueError, match="no silent Llama"):
        script.require_portfolio_qwen(
            {
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "revision": PORTFOLIO_DEMO_MODEL_REVISION,
            }
        )
    with pytest.raises(ValueError, match="pin"):
        script.require_portfolio_qwen({"model": PORTFOLIO_DEMO_MODEL_ID, "revision": "not-the-pin"})
    script.require_portfolio_qwen(
        {"model": PORTFOLIO_DEMO_MODEL_ID, "revision": PORTFOLIO_DEMO_MODEL_REVISION}
    )


def test_pack_copies_named_traces_without_a_model(tmp_path: Path) -> None:
    script = _load_script()
    src_a = T22N_TRACES / "sheep_trick_sampling_a.jsonl"
    src_b = T22N_TRACES / "sheep_trick_sampling_b.jsonl"
    trace_a = load_path(src_a)
    trace_b = load_path(src_b)
    store = tmp_path / "store" / "traces"
    store.mkdir(parents=True)
    shutil.copyfile(src_a, store / f"{trace_a.run_metadata.trace_id}.jsonl")
    shutil.copyfile(src_b, store / f"{trace_b.run_metadata.trace_id}.jsonl")
    study = {
        "n_items": 1,
        "n_pairs": 1,
        "model": PORTFOLIO_DEMO_MODEL_ID,
        "revision": PORTFOLIO_DEMO_MODEL_REVISION,
        "sampling_seeds": [1, 2],
        "sampling_temperature": 1.0,
        "decoding_seed": 1,
        "decoding_temperatures": [0.7, 1.2],
        "max_new_tokens": 64,
        "capture_k": 5,
        "grading_rule": script.GRADING_RULE,
        "pairs": [
            {
                "item_id": "sheep_trick",
                "gold": 9,
                "intended_kind": "sampling",
                "observed_class": "sampling",
                "first_step": 0,
                "grade_a": "correct",
                "grade_b": "wrong",
                "extracted_a": 9,
                "extracted_b": 3,
                "outcome": "disagree",
                "trace_a": str(trace_a.run_metadata.trace_id),
                "trace_b": str(trace_b.run_metadata.trace_id),
            }
        ],
        "summary": {},
    }
    study_path = tmp_path / "study.json"
    study_path.write_text(json.dumps(study), encoding="utf-8")
    dest = tmp_path / "named"
    packed = script.pack_from_study(
        study_json=study_path,
        store=tmp_path / "store",
        traces_dir=dest,
        command="llmfr study ...",
        n_items_attempted=1,
        stop_reason=None,
    )
    assert packed["finding_id"] == "GS-T22q"
    assert packed["backend"] == "hf"
    assert packed["model"] == PORTFOLIO_DEMO_MODEL_ID
    assert packed["revision"] == PORTFOLIO_DEMO_MODEL_REVISION
    named_a = dest / "sheep_trick_sampling_a.jsonl"
    named_b = dest / "sheep_trick_sampling_b.jsonl"
    assert named_a.is_file()
    assert named_b.is_file()
    assert packed["pairs"][0]["path_a"].endswith("sheep_trick_sampling_a.jsonl")
    recomputed = summarize_pairs(packed["pairs"])
    assert packed["summary"]["by_class"] == recomputed["by_class"]
    assert packed["summary"]["descriptive_verdict"] == descriptive_verdict(recomputed["by_class"])
    rendered = script.render_finding(packed)
    assert PORTFOLIO_DEMO_MODEL_ID in rendered
    assert "tiny-gpt2" in rendered
    assert "Llama" in rendered
    assert "llmfr study" in rendered


def test_checked_in_finding_matches_results_json() -> None:
    script = _load_script()
    assert RESULTS_JSON.is_file(), "GS-T22q results JSON missing; run llmfr study and pack"
    assert FINDING_MD.is_file(), "GS-T22q finding markdown missing"
    payload = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    recomputed = summarize_pairs(payload["pairs"])
    recomputed["descriptive_verdict"] = descriptive_verdict(recomputed["by_class"])
    assert payload["summary"] == recomputed
    assert payload["finding_id"] == "GS-T22q"
    assert payload["backend"] == "hf"
    assert payload["model"] == PORTFOLIO_DEMO_MODEL_ID
    assert payload["revision"] == PORTFOLIO_DEMO_MODEL_REVISION
    assert payload["n_items"] >= TARGET_N
    assert payload["n_pairs"] == payload["n_items"] * 2
    assert payload["max_new_tokens"] == 64
    assert payload["capture_k"] == 5
    assert payload["grading_rule"] == script.GRADING_RULE
    finding = FINDING_MD.read_text(encoding="utf-8")
    assert finding == script.render_finding(payload)
    readme = README.read_text(encoding="utf-8")
    _assert_readme_findings(readme, payload)
    assert PORTFOLIO_DEMO_MODEL_ID in finding
    assert PORTFOLIO_DEMO_MODEL_REVISION in finding
    assert "tiny-gpt2" in finding
    assert "Llama" in finding
    assert "llmfr study" in finding
    sampling = payload["summary"]["by_class"]["sampling"]
    decoding = payload["summary"]["by_class"]["decoding config"]
    for row in (sampling, decoding):
        if row["n_gradeable_diverged"]:
            assert row["disagree_rate"] == row["disagree"] / row["n_gradeable_diverged"]
        else:
            assert row["disagree_rate"] is None
        if row["n_gradeable_traces"]:
            assert row["wrong_answer_rate"] == row["n_wrong_traces"] / row["n_gradeable_traces"]
        else:
            assert row["wrong_answer_rate"] is None
    jobs = load_prompt_file(PROMPTS)
    assert {pair["item_id"] for pair in payload["pairs"]} <= {job.id for job in jobs}


def test_checked_in_traces_recompare_without_model() -> None:
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
        assert trace_b.model.name == PORTFOLIO_DEMO_MODEL_ID
        assert trace_b.model.revision == PORTFOLIO_DEMO_MODEL_REVISION
        assert "llama" not in trace_a.model.name.lower()
        compared = compare_traces(trace_a, trace_b)
        first = compared.first_divergence
        observed = None if first is None else first.classification
        assert observed == pair["observed_class"]
        gold = int(pair["gold"])
        assert grade_output(trace_a.run_metadata.output_text, gold) == pair["grade_a"]
        assert grade_output(trace_b.run_metadata.output_text, gold) == pair["grade_b"]
        assert trace_a.run_metadata.logits.mode == "topk"
        assert trace_a.events
        assert trace_a.events[0].sampled_logit is not None
