"""Structured two-trace compare outcome (Milestone 5).

First divergence is the earliest causal split. Later context or logit diffs
are tagged downstream so they are not mistaken for a new root cause.
"""

from __future__ import annotations

from typing import Literal

from llmfr.core.schema import FrozenModel

DivergenceClass = Literal[
    "prompt/history",
    "tokenizer",
    "model-visible context",
    "model/version",
    "raw-logit",
    "decoding config",
    "probability distribution",
    "sampling",
    "unknown/runtime",
]

DIVERGENCE_CLASSES: tuple[DivergenceClass, ...] = (
    "prompt/history",
    "tokenizer",
    "model-visible context",
    "model/version",
    "raw-logit",
    "decoding config",
    "probability distribution",
    "sampling",
    "unknown/runtime",
)

StepRole = Literal["first", "downstream"]


class ConfigDiff(FrozenModel):
    """One run-level field that differs between the two traces."""

    field: str
    a: str | None = None
    b: str | None = None


class StepDiff(FrozenModel):
    """Per-step event fields that differ. ``role=downstream`` is not a root cause."""

    step: int
    role: StepRole
    differences: tuple[str, ...]
    a_sampled_token_id: int | None = None
    b_sampled_token_id: int | None = None
    a_sampled_token: str | None = None
    b_sampled_token: str | None = None


class FirstDivergence(FrozenModel):
    """Earliest meaningful split, classified on the recorder causal pipeline."""

    step: int
    classification: DivergenceClass
    differences: tuple[str, ...]
    reason: str
    a_sampled_token_id: int | None = None
    b_sampled_token_id: int | None = None
    a_sampled_token: str | None = None
    b_sampled_token: str | None = None


class CompareResult(FrozenModel):
    """Config diffs plus first divergence and downstream fallout.

    ``identical`` is true only when there is no first divergence and no
    config field of interest differs. ``first_divergence`` is the root-cause
    class for M5. ``downstream`` must not be treated as additional root
    causes (Case E).
    """

    trace_a: str
    trace_b: str
    identical: bool
    config_diffs: tuple[ConfigDiff, ...] = ()
    first_divergence: FirstDivergence | None = None
    downstream: tuple[StepDiff, ...] = ()
    notes: tuple[str, ...] = ()
    steps: tuple[StepDiff, ...] = ()
