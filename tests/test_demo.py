"""Portfolio demo fixtures and README shape. No live model; no invented logits."""

from __future__ import annotations

from pathlib import Path

from pytest import CaptureFixture

from llmfr.cli import run
from llmfr.core.schema import load_path

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
DEMO_MD = REPO / "docs" / "demo.md"
DEMO_DIR = REPO / "examples" / "demo"
FIXTURES = DEMO_DIR
CAPTURES = DEMO_DIR

README_SECTIONS = ("What", "Why", "Architecture", "How", "Essentials")
ADR_LINKS = (
    "docs/adr/0001-v1-trace-schema.md",
    "docs/adr/0002-v1-storage.md",
    "docs/adr/0003-hf-adapter-default-model.md",
    "docs/adr/0004-recorder-loop.md",
    "docs/adr/0005-deterministic-replay.md",
    "docs/adr/0006-first-divergence-compare.md",
    "docs/adr/0007-root-cause-report.md",
    "docs/adr/0008-production-cli.md",
)


def test_readme_is_what_why_how_essentials_only() -> None:
    text = README.read_text(encoding="utf-8")
    headings = [line[3:].strip() for line in text.splitlines() if line.startswith("## ")]
    assert headings == list(README_SECTIONS)
    assert "Milestone" not in text
    assert "\u2014" not in text
    assert "\u2013" not in text
    assert "```mermaid" in text
    assert "TraceStore" in text
    assert "first divergence" in text
    assert "downstream effects" in text
    assert "llmfr record" in text
    assert "--seed 1" in text
    assert "--seed 2" in text
    assert "--temperature 0.7" in text
    assert "--temperature 1.2" in text
    assert "llmfr compare" in text
    assert "docs/demo.md" in text
    assert "docs/adr/" in text
    assert "examples/demo/demo1_a.jsonl" in text


def test_demo_markdown_embeds_captured_reports() -> None:
    demo = DEMO_MD.read_text(encoding="utf-8")
    for name in ("demo1_compare.txt", "demo2_compare.txt", "demo1_inspect_step0.txt"):
        captured = (CAPTURES / name).read_text(encoding="utf-8").strip()
        assert captured in demo
    assert "not a new root cause" in demo
    assert "class: sampling" in demo
    assert "seed difference likely enabled this sampling split" in demo
    assert "class: decoding config" in demo
    assert "generation_config.seed: 1 vs 2" in demo
    assert "generation_config.temperature: 0.7 vs 1.2" in demo
    for path in ADR_LINKS:
        assert path in demo
    assert "does not invent" in demo.lower() or "not invented" in demo.lower()
    source = (CAPTURES / "SOURCE.txt").read_text(encoding="utf-8")
    assert "backend=huggingface" in source
    assert "sshleifer/tiny-gpt2" in source
    assert "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be" in source


def test_fixtures_are_tiny_gpt2_with_real_logits() -> None:
    for name in ("demo1_a.jsonl", "demo1_b.jsonl", "demo2_a.jsonl", "demo2_b.jsonl"):
        trace = load_path(FIXTURES / name)
        assert trace.model.name == "sshleifer/tiny-gpt2"
        assert trace.model.revision == "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"
        assert trace.run_metadata.logits.mode == "topk"
        assert trace.events
        for event in trace.events:
            assert event.sampled_logit is not None
            assert event.top_k
            assert event.top_k[0].logit is not None


def test_cli_compare_fixtures_match_checked_in_captures(capsys: CaptureFixture[str]) -> None:
    for name in ("demo1", "demo2"):
        path_a = FIXTURES / f"{name}_a.jsonl"
        path_b = FIXTURES / f"{name}_b.jsonl"
        expected = (CAPTURES / f"{name}_compare.txt").read_text(encoding="utf-8")
        assert run(["compare", str(path_a), str(path_b)]) == 1
        assert capsys.readouterr().out == expected


def test_cli_inspect_fixture_matches_capture(capsys: CaptureFixture[str]) -> None:
    expected = (CAPTURES / "demo1_inspect_step0.txt").read_text(encoding="utf-8")
    assert run(["inspect", str(FIXTURES / "demo1_a.jsonl"), "--step", "0"]) == 0
    assert capsys.readouterr().out == expected
