"""Structured replay outcome. Token match is not a silent bit-identical claim."""

from __future__ import annotations

from typing import Literal

from llmfr.core.schema import FrozenModel

ReplayStatus = Literal[
    "reproduced",
    "partially_reproduced",
    "not_reproduced",
    "not_replayable",
]


class StepReplay(FrozenModel):
    """One recorded decode step compared to a replay forward pass and sample."""

    step: int
    recorded_token_id: int
    replayed_token_id: int
    token_matched: bool
    prefix_matched: bool
    recorded_logit: float | None = None
    replayed_logit: float | None = None
    logit_matched: bool | None = None


class ReplayResult(FrozenModel):
    """What replay did and did not reproduce.

    ``status="reproduced"`` means every recorded sampled token id was produced
    again, in order, from the stored seed and generation_config. It does not
    mean logits were bit-identical. Read ``logits_bit_identical`` and
    ``bit_identical`` for that. Partial token match is
    ``partially_reproduced``; a first-step miss is ``not_reproduced``.
    """

    trace_id: str
    status: ReplayStatus
    matched_steps: int
    total_steps: int
    first_unmatched_step: int | None = None
    token_ids_matched: bool = False
    logits_bit_identical: bool = False
    bit_identical: bool = False
    recorded_revision: str | None = None
    replayed_revision: str | None = None
    reason: str | None = None
    notes: tuple[str, ...] = ()
    steps: tuple[StepReplay, ...] = ()


BIT_IDENTICAL_CAVEAT = (
    "Bit-identical logits are not guaranteed across CPU vs GPU, PyTorch "
    "versions, or kernel implementations. Token match is reported separately "
    "from logit match. See docs/adr/0005-deterministic-replay.md."
)
