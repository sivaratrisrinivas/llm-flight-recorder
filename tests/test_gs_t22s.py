"""GS-T22s latency script guards and checked-in Qwen capture. No live 0.5B download."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from llmfr.adapters.huggingface import (
    DEFAULT_HF_MODEL_ID,
    PORTFOLIO_DEMO_MODEL_ID,
    PORTFOLIO_DEMO_MODEL_REVISION,
)
from llmfr.record.batch import load_prompt_file
from llmfr.study import require_study_jobs

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "gs_t22s_latency.py"
PROMPTS = REPO / "examples" / "findings" / "gs-t22s" / "prompts.jsonl"
FINDING_MD = REPO / "docs" / "findings" / "gs-t22s-latency.md"
RESULTS_JSON = REPO / "docs" / "findings" / "gs-t22s-results.json"
README = REPO / "README.md"
FINDING_MD_LINK = "docs/findings/gs-t22s-latency.md"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("llmfr_gs_t22s", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_nearest_rank_percentile() -> None:
    script = _load_script()
    samples = [float(index) for index in range(1, 12)]
    assert script.percentile(samples, 0) == 1.0
    assert script.percentile(samples, 50) == 6.0
    assert script.percentile(samples, 99) == 11.0
    assert script.percentile(samples, 100) == 11.0
    stats = script.summarize_samples(samples, warmup=2, n_trials=11)
    assert stats["p50_s"] == 6.0
    assert stats["p99_s"] == 11.0
    assert stats["min_s"] == 1.0
    assert stats["max_s"] == 11.0
    assert stats["percentile_method"] == script.PERCENTILE_METHOD
    assert script.percentile_method_text(11) == script.PERCENTILE_METHOD
    assert "With N=11, p99 is the maximum timed trial." in script.PERCENTILE_METHOD
    three = script.percentile_method_text(3)
    assert "With N=3, p99 is the maximum timed trial." in three
    assert "N=11" not in three
    one = script.percentile_method_text(1)
    assert "With N=1, p99 is the maximum timed trial." in one
    assert "N=11" not in one
    with pytest.raises(ValueError, match="at least one"):
        script.percentile([], 50)
    with pytest.raises(ValueError, match="expected 11"):
        script.summarize_samples([1.0], warmup=2, n_trials=11)


def test_require_portfolio_qwen_rejects_tiny_gpt2_and_llama() -> None:
    script = _load_script()
    with pytest.raises(ValueError, match="no silent Llama"):
        script.require_portfolio_qwen(
            {
                "backend": "hf",
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "revision": PORTFOLIO_DEMO_MODEL_REVISION,
            }
        )
    with pytest.raises(ValueError, match="tiny-gpt2"):
        script.require_portfolio_qwen(
            {"backend": "hf", "model": DEFAULT_HF_MODEL_ID, "revision": "abc"}
        )
    with pytest.raises(ValueError, match="pin"):
        script.require_portfolio_qwen(
            {"backend": "hf", "model": PORTFOLIO_DEMO_MODEL_ID, "revision": "not-the-pin"}
        )
    with pytest.raises(ValueError, match="backend=hf"):
        script.require_portfolio_qwen(
            {
                "backend": "fake",
                "model": PORTFOLIO_DEMO_MODEL_ID,
                "revision": PORTFOLIO_DEMO_MODEL_REVISION,
            }
        )
    script.require_portfolio_qwen(
        {
            "backend": "hf",
            "model": PORTFOLIO_DEMO_MODEL_ID,
            "revision": PORTFOLIO_DEMO_MODEL_REVISION,
        }
    )


def test_study_prompt_is_one_gradeable_item() -> None:
    jobs = load_prompt_file(PROMPTS)
    require_study_jobs(jobs)
    assert len(jobs) == 1
    assert jobs[0].id == "sheep_trick"
    assert jobs[0].gold == 9
    assert "Think step by step, then give the final number." in jobs[0].prompt


def test_fake_backend_refuses_docs_bound_paths(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    script = _load_script()
    assert script.main(["--backend", "fake"]) == 2
    err = capsys.readouterr().err
    assert "refusing to overwrite docs/findings" in err
    assert (
        script.main(
            [
                "--backend",
                "fake",
                "--out",
                str(tmp_path / "scratch"),
            ]
        )
        == 2
    )
    err_results = capsys.readouterr().err
    assert "refusing to overwrite docs/findings" in err_results


def test_fake_backend_scratch_writes_labeled_finding_not_docs(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    script = _load_script()
    results = tmp_path / "results.json"
    finding = tmp_path / "finding.md"
    finding_before = FINDING_MD.read_text(encoding="utf-8") if FINDING_MD.is_file() else None
    results_before = RESULTS_JSON.read_text(encoding="utf-8") if RESULTS_JSON.is_file() else None
    code = script.main(
        [
            "--backend",
            "fake",
            "--out",
            str(tmp_path / "scratch"),
            "--results",
            str(results),
            "--finding",
            str(finding),
            "--warmup",
            "0",
            "--trials",
            "3",
            "--max-new-tokens",
            "4",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    payload = json.loads(results.read_text(encoding="utf-8"))
    assert payload["backend"] == "fake"
    assert payload["model"] == "fake-lm"
    assert payload["finding_id"] == "GS-T22s"
    assert payload["warmup"] == 0
    assert payload["n_trials"] == 3
    assert payload["timing_mode"] == "in-process-cli"
    assert payload["model"] != PORTFOLIO_DEMO_MODEL_ID
    assert finding.is_file()
    finding_text = finding.read_text(encoding="utf-8")
    preamble = script.container_preamble(payload)
    assert finding_text.startswith(preamble)
    assert "Container/CI capture" in finding_text
    assert "backend=`fake`" in finding_text
    assert "model=`fake-lm`" in finding_text
    assert FINDING_MD_LINK in finding_text
    assert captured.out.startswith(preamble)
    assert "CPU Qwen capture" not in captured.out
    assert "CPU Qwen capture" not in finding_text
    assert "portfolio Qwen CPU path" not in finding_text
    assert "smoke runs are not this table" not in finding_text
    assert "are not used for this finding" not in finding_text
    assert "They are not the table above" not in finding_text
    assert "0.5B-class instruct model" not in finding_text
    assert "smoke/container capture" in finding_text
    assert "Smoke/container capture" in captured.out
    assert "Not the portfolio Qwen table." in captured.out
    assert script.readme_latency_line(payload).startswith("Smoke/container capture")
    assert payload["percentile_method"] == script.percentile_method_text(3)
    assert "With N=3, p99 is the maximum timed trial." in payload["percentile_method"]
    assert "N=11" not in payload["percentile_method"]
    assert "With N=3, p99 is the maximum timed trial." in finding_text
    assert "With N=11, p99 is the maximum timed trial." not in finding_text
    assert "python scripts/gs_t22s_latency.py --backend hf --write-docs" not in finding_text
    assert "Raw JSON: `docs/findings/gs-t22s-results.json`" not in finding_text
    assert "CI smoke (fake adapter; must not overwrite this finding)" not in finding_text
    assert "python scripts/gs_t22s_latency.py --backend hf --write-docs" not in captured.out
    for name in ("record", "compare", "study"):
        row = payload["commands"][name]
        assert len(row["samples_s"]) == 3
        assert all(sample >= 0.0 for sample in row["samples_s"])
        recomputed = script.summarize_samples(row["samples_s"], warmup=0, n_trials=3)
        assert row["p50_s"] == recomputed["p50_s"]
        assert row["p99_s"] == recomputed["p99_s"]
        assert row["min_s"] <= row["p50_s"] <= row["max_s"]
        assert row["p50_s"] <= row["p99_s"]
    if finding_before is None:
        assert not FINDING_MD.is_file()
    else:
        assert FINDING_MD.read_text(encoding="utf-8") == finding_before
        assert FINDING_MD.read_text(encoding="utf-8") != finding_text
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
            str(tmp_path / "scratch"),
            "--results",
            str(tmp_path / "results.json"),
            "--finding",
            str(tmp_path / "finding.md"),
            "--warmup",
            "0",
            "--trials",
            "1",
            "--write-docs",
        ]
    )
    assert code == 2
    assert "refusing to write finding docs from a non-hf backend" in capsys.readouterr().err


def test_from_results_recomputes_percentiles(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _load_script()
    samples = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    stats = script.summarize_samples(samples, warmup=2, n_trials=11)
    payload = {
        "finding_id": "GS-T22s",
        "recorded_at": "2026-09-20T00:00:00+00:00",
        "backend": "hf",
        "timing_mode": "subprocess",
        "model": PORTFOLIO_DEMO_MODEL_ID,
        "revision": PORTFOLIO_DEMO_MODEL_REVISION,
        "warmup": 2,
        "n_trials": 11,
        "max_new_tokens": 16,
        "capture_k": 5,
        "percentile_method": "wrong",
        "hardware": {
            "platform": "test",
            "python_version": "3.12.3",
            "cpu_count": 4,
            "machine": "x86_64",
            "cpu_model": "test-cpu",
        },
        "library_versions": {"llmfr": "0.1.0"},
        "commands": {
            "record": {
                "command": "record",
                "argv": ["record"],
                "displayed_command": "llmfr record",
                "timing_mode": "subprocess",
                "allowed_exit_codes": [0],
                "workload": {},
                "samples_s": samples,
                "p50_s": 0.0,
                "p99_s": 0.0,
                "min_s": 0.0,
                "max_s": 0.0,
                "mean_s": 0.0,
            },
            "compare": {
                "command": "compare",
                "argv": ["compare"],
                "displayed_command": "llmfr compare",
                "timing_mode": "subprocess",
                "allowed_exit_codes": [0, 1],
                "workload": {},
                "samples_s": samples,
                "p50_s": 0.0,
                "p99_s": 0.0,
                "min_s": 0.0,
                "max_s": 0.0,
                "mean_s": 0.0,
            },
            "study": {
                "command": "study",
                "argv": ["study"],
                "displayed_command": "llmfr study",
                "timing_mode": "subprocess",
                "allowed_exit_codes": [0],
                "workload": {},
                "samples_s": samples,
                "p50_s": 0.0,
                "p99_s": 0.0,
                "min_s": 0.0,
                "max_s": 0.0,
                "mean_s": 0.0,
            },
        },
    }
    src = tmp_path / "hand-edited.json"
    src.write_text(json.dumps(payload), encoding="utf-8")
    code = script.main(["--from-results", str(src)])
    assert code == 0
    out = capsys.readouterr().out
    assert f"{stats['p50_s']:.3f}" in out
    assert f"{stats['p99_s']:.3f}" in out
    result_block = out.split("## Result", 1)[1].split("## Commands timed", 1)[0]
    assert "0.000" not in result_block
    recomputed = script.recompute_payload(payload)
    assert recomputed["commands"]["record"]["p50_s"] == stats["p50_s"]
    assert recomputed["commands"]["record"]["p99_s"] == stats["p99_s"]
    assert recomputed["percentile_method"] == script.PERCENTILE_METHOD


def test_checked_in_finding_matches_results_json() -> None:
    script = _load_script()
    assert RESULTS_JSON.is_file(), "GS-T22s results JSON missing; run the hf latency capture"
    assert FINDING_MD.is_file(), "GS-T22s finding markdown missing"
    payload = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    recomputed = script.recompute_payload(payload)
    for name in ("record", "compare", "study"):
        assert payload["commands"][name]["p50_s"] == recomputed["commands"][name]["p50_s"]
        assert payload["commands"][name]["p99_s"] == recomputed["commands"][name]["p99_s"]
        row = payload["commands"][name]
        assert len(row["samples_s"]) == script.DEFAULT_TRIALS
        assert row["p50_s"] == script.percentile(row["samples_s"], 50)
        assert row["p99_s"] == script.percentile(row["samples_s"], 99)
        assert min(row["samples_s"]) == row["min_s"]
        assert max(row["samples_s"]) == row["max_s"]
    assert payload["finding_id"] == "GS-T22s"
    assert payload["backend"] == "hf"
    assert payload["timing_mode"] == "subprocess"
    assert payload["model"] == PORTFOLIO_DEMO_MODEL_ID
    assert payload["revision"] == PORTFOLIO_DEMO_MODEL_REVISION
    assert payload["warmup"] == script.DEFAULT_WARMUP
    assert payload["n_trials"] == script.DEFAULT_TRIALS
    assert payload["max_new_tokens"] == script.DEFAULT_MAX_NEW_TOKENS
    finding = FINDING_MD.read_text(encoding="utf-8")
    assert finding == script.render_finding(recomputed)
    assert script.FAKE_SMOKE_COMMAND in finding
    assert "--results /tmp/llmfr-gs-t22s-fake/results.json" in finding
    assert "--finding /tmp/llmfr-gs-t22s-fake/finding.md" in finding
    assert "--out /tmp/llmfr-gs-t22s-fake" in finding
    assert PORTFOLIO_DEMO_MODEL_ID in finding
    assert PORTFOLIO_DEMO_MODEL_REVISION in finding
    assert "tiny-gpt2" in finding
    assert "Llama" in finding
    assert "llmfr record" in finding
    assert "llmfr compare" in finding
    assert "llmfr study" in finding
    assert str(script.DEFAULT_WARMUP) in finding
    assert str(script.DEFAULT_TRIALS) in finding
    readme = README.read_text(encoding="utf-8")
    line = script.readme_latency_line(recomputed)
    assert line in readme
    assert FINDING_MD_LINK in readme
    findings = readme[readme.index("## Findings") :]
    assert "p50/p99" in findings
    assert "llmfr record" in findings
    assert "llmfr compare" in findings
    assert "llmfr study" in findings
    assert "CPU Qwen capture" in findings
    assert "Smoke/container capture" not in findings


def test_readme_latency_line_gates_qwen_wording_on_portfolio_pin() -> None:
    script = _load_script()
    payload = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    recomputed = script.recompute_payload(payload)
    qwen_line = script.readme_latency_line(recomputed)
    assert qwen_line.startswith("CLI wall-clock on this CPU Qwen capture")
    assert "6.969" in qwen_line
    assert script.is_portfolio_capture(recomputed)
    smoke = dict(recomputed)
    smoke["backend"] = "fake"
    smoke["model"] = "fake-lm"
    smoke["revision"] = None
    smoke["max_new_tokens"] = 4
    assert not script.is_portfolio_capture(smoke)
    smoke_line = script.readme_latency_line(smoke)
    assert smoke_line.startswith("Smoke/container capture")
    assert "CPU Qwen capture" not in smoke_line
    assert "backend=fake" in smoke_line
    assert "model=`fake-lm`" in smoke_line
    assert "Not the portfolio Qwen table." in smoke_line
    tiny = dict(recomputed)
    tiny["backend"] = "tiny-gpt2"
    tiny["model"] = DEFAULT_HF_MODEL_ID
    tiny["revision"] = None
    tiny_line = script.readme_latency_line(tiny)
    assert tiny_line.startswith("Smoke/container capture")
    assert "CPU Qwen capture" not in tiny_line
    rendered_smoke = script.render_finding(smoke)
    assert "portfolio Qwen CPU path" not in rendered_smoke
    assert "smoke runs are not this table" not in rendered_smoke
    assert "are not used for this finding" not in rendered_smoke
    assert "They are not the table above" not in rendered_smoke
    assert "0.5B-class instruct model" not in rendered_smoke
    assert "smoke/container capture" in rendered_smoke
    assert "python scripts/gs_t22s_latency.py --backend hf --write-docs" not in rendered_smoke
    assert "Raw JSON: `docs/findings/gs-t22s-results.json`" not in rendered_smoke
    assert "CI smoke (fake adapter; must not overwrite this finding)" not in rendered_smoke
    no_footer = script.render_finding(recomputed, docs_footer=False)
    assert "python scripts/gs_t22s_latency.py --backend hf --write-docs" not in no_footer
    assert "Raw JSON: `docs/findings/gs-t22s-results.json`" not in no_footer
    assert FINDING_MD.read_text(encoding="utf-8") == script.render_finding(recomputed)
