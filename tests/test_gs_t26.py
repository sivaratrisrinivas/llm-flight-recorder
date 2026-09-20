"""GS-T26 container + Actions contract. No live model download."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from llmfr.adapters.huggingface import PORTFOLIO_DEMO_MODEL_ID

REPO = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO / "Dockerfile"
DOCKERIGNORE = REPO / ".dockerignore"
WRAPPER = REPO / "scripts" / "container_latency.sh"
WORKFLOW = REPO / ".github" / "workflows" / "container-bench.yml"
CI = REPO / ".github" / "workflows" / "ci.yml"
README = REPO / "README.md"
LATENCY = REPO / "scripts" / "gs_t22s_latency.py"
FINDING_MD_LINK = "docs/findings/gs-t22s-latency.md"
QWEN_RECORD_P50 = "6.969"


def _load_latency() -> ModuleType:
    spec = importlib.util.spec_from_file_location("llmfr_gs_t22s_for_t26", LATENCY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dockerfile_stages(text: str) -> list[str]:
    stages: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("FROM ") and " AS " in stripped.upper():
            stages.append(stripped.rsplit(" AS ", 1)[1].strip())
    return stages


def test_dockerfile_default_is_fake_latency_smoke() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    stages = _dockerfile_stages(text)
    assert "smoke" in stages
    assert "hf" in stages
    assert stages[-1] == "smoke"
    assert stages.index("hf") < stages.index("smoke")
    assert "scripts/container_latency.sh" in text
    assert 'CMD ["--backend", "fake"]' in text
    assert "tiny-gpt2" in text
    assert "download.pytorch.org/whl/cpu" in text
    assert "docs/findings/gs-t22s-latency.md" in text
    assert WRAPPER.is_file()
    wrapper = WRAPPER.read_text(encoding="utf-8")
    assert wrapper.startswith("#!/bin/sh")
    assert "gs_t22s_latency.py" in wrapper
    assert "python3" in wrapper
    assert '"$OUT/results.json"' in wrapper
    assert '"$OUT/finding.md"' in wrapper
    assert "LLMFR_OUT" in wrapper
    assert "--backend" not in wrapper.split("exec", 1)[1]
    assert DOCKERIGNORE.is_file()
    ignore = DOCKERIGNORE.read_text(encoding="utf-8")
    assert ".git" in ignore
    assert "examples/findings/gs-t22q" in ignore


def test_workflow_builds_smoke_and_uploads_measured_artifacts() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "container-bench" in text
    assert "workflow_dispatch" in text
    assert "tiny-gpt2" in text
    assert "\n          - hf\n" in text or "- hf" in text
    assert "fake" in text
    assert "docker build --target smoke" in text
    assert "docker build --target hf" in text
    assert "upload-artifact" in text
    assert "out/results.json" in text
    assert "out/finding.md" in text
    assert "if-no-files-found: error" in text
    assert "gs-t26-latency-fake" in text
    assert 'payload["backend"] == "fake"' in text
    assert "Qwen/Qwen2.5-0.5B-Instruct" in text
    assert CI.read_text(encoding="utf-8").count("pytest") >= 1
    assert "ruff check src tests" in CI.read_text(encoding="utf-8")
    assert "mypy" in CI.read_text(encoding="utf-8")


def test_readme_container_section_does_not_invent_qwen_timings() -> None:
    text = README.read_text(encoding="utf-8")
    container = text[text.index("## Container") : text.index("## Essentials")]
    findings = text[text.index("## Findings") :]
    assert "docker build --target smoke" in container
    assert "docker run" in container
    assert "--target hf" in container
    assert "container-bench.yml" in container
    assert "fake" in container
    assert "tiny-gpt2" in container
    assert "workflow_dispatch" in container
    assert FINDING_MD_LINK in container
    assert QWEN_RECORD_P50 not in container
    assert "6.969" not in container
    assert "not the Qwen" in container or "not the checked-in" in container
    assert QWEN_RECORD_P50 in findings
    assert FINDING_MD_LINK in findings
    assert PORTFOLIO_DEMO_MODEL_ID in findings


def test_container_preamble_labels_non_docs_captures() -> None:
    script = _load_latency()
    fake = script.container_preamble({"backend": "fake", "model": "fake-lm", "revision": None})
    assert fake.startswith("Container/CI capture:")
    assert "backend=`fake`" in fake
    assert "model=`fake-lm`" in fake
    assert "revision=`none`" in fake
    assert FINDING_MD_LINK in fake
    assert "--backend hf --write-docs" in fake
    qwen = script.container_preamble(
        {
            "backend": "hf",
            "model": PORTFOLIO_DEMO_MODEL_ID,
            "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        }
    )
    assert PORTFOLIO_DEMO_MODEL_ID in qwen
    assert "7ae557604adf67be50417f59c2c2f167def9a775" in qwen
