"""Replay a stored Trace by re-running the M3 sampling path.

Uses recorded seed, effective generation_config, and the Hub revision pin.
Does not invent logits. Does not claim bit-identical floats from a token match.
"""

from __future__ import annotations

from collections.abc import Sequence

from llmfr.adapters.huggingface import HuggingFaceCausalLMAdapter, HuggingFaceExtraMissingError
from llmfr.core.schema import GenerationConfig, TopKCandidate, Trace
from llmfr.record.recorder import RecordableAdapter
from llmfr.record.sample import LocalRNG, choose_token, is_greedy
from llmfr.replay.result import BIT_IDENTICAL_CAVEAT, ReplayResult, ReplayStatus, StepReplay

_CPU_DEVICES = frozenset({"cpu", "cpu:0"})


def infer_max_visible_tokens(trace: Trace) -> int | None:
    """Recover the recorder's left-clip from truncated visible windows."""
    lengths = [
        len(event.model_visible_context.token_ids)
        for event in trace.events
        if event.model_visible_context.truncated
    ]
    if not lengths:
        return None
    return min(lengths)


def build_adapter_for_trace(trace: Trace) -> tuple[HuggingFaceCausalLMAdapter, list[str]]:
    """Load the Hugging Face adapter at the revision stored on the Trace."""
    notes: list[str] = []
    if trace.model.provider != "huggingface":
        raise ValueError(
            f"cannot build a Hugging Face adapter for provider {trace.model.provider!r}"
        )
    revision = trace.model.revision
    if not revision:
        raise ValueError("trace has no Hub revision pin; refusing to replay against a moving name")
    requested = _requested_device(trace)
    device, device_note = _replay_device(requested)
    if device_note:
        notes.append(device_note)
    adapter = HuggingFaceCausalLMAdapter(
        trace.model.name,
        revision=revision,
        device=device,
        max_visible_tokens=infer_max_visible_tokens(trace),
    )
    recorded_dtype = trace.model.dtype
    if recorded_dtype is not None and recorded_dtype != adapter.model_config.dtype:
        notes.append(
            f"recorded dtype was {recorded_dtype}; replay adapter dtype is "
            f"{adapter.model_config.dtype}"
        )
    return adapter, notes


def replay_trace(
    trace: Trace,
    adapter: RecordableAdapter | None = None,
) -> ReplayResult:
    """Re-run sampling for ``trace`` as far as this runtime permits."""
    trace_id = str(trace.run_metadata.trace_id)
    recorded_revision = trace.model.revision
    notes: list[str] = [BIT_IDENTICAL_CAVEAT]

    blocked = _not_replayable_reason(trace, adapter)
    if blocked is not None:
        return _blocked(
            trace_id=trace_id,
            recorded_revision=recorded_revision,
            reason=blocked,
            notes=tuple(notes),
            total_steps=len(trace.events),
        )

    extra_notes: list[str] = []
    if adapter is None:
        try:
            adapter, extra_notes = build_adapter_for_trace(trace)
        except HuggingFaceExtraMissingError as exc:
            return _blocked(
                trace_id=trace_id,
                recorded_revision=recorded_revision,
                reason=str(exc),
                notes=tuple(notes),
                total_steps=len(trace.events),
            )
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            return _blocked(
                trace_id=trace_id,
                recorded_revision=recorded_revision,
                reason=str(exc),
                notes=tuple(notes),
                total_steps=len(trace.events),
            )
    notes.extend(extra_notes)

    adapter_blocked = _adapter_unusable(trace, adapter)
    if adapter_blocked is not None:
        return _blocked(
            trace_id=trace_id,
            recorded_revision=recorded_revision,
            replayed_revision=adapter.model_config.revision,
            reason=adapter_blocked,
            notes=tuple(notes),
            total_steps=len(trace.events),
        )

    replayed_revision = adapter.model_config.revision
    if (
        recorded_revision is not None
        and replayed_revision is not None
        and recorded_revision != replayed_revision
    ):
        return _blocked(
            trace_id=trace_id,
            recorded_revision=recorded_revision,
            replayed_revision=replayed_revision,
            reason=(
                f"adapter revision {replayed_revision} does not match trace pin {recorded_revision}"
            ),
            notes=tuple(notes),
            total_steps=len(trace.events),
        )

    prompt_ids, prompt_notes = _prompt_ids(trace, adapter)
    notes.extend(prompt_notes)
    if not prompt_ids:
        return _blocked(
            trace_id=trace_id,
            recorded_revision=recorded_revision,
            replayed_revision=replayed_revision,
            reason="prompt encoded to no tokens",
            notes=tuple(notes),
            total_steps=len(trace.events),
        )

    generation = trace.generation_config
    greedy = is_greedy(generation)
    rng = LocalRNG(generation.seed)
    history = list(prompt_ids)
    capture_k = trace.run_metadata.logits.k
    step_rows: list[StepReplay] = []
    matched_prefix = True
    matched_steps = 0
    first_unmatched: int | None = None
    logits_ok = True
    compared_a_logit = False

    for event in trace.events:
        step_logits = adapter.next_token_logits(history)
        decision = choose_token(
            step_logits.logits,
            temperature=generation.temperature,
            greedy=greedy,
            rng=None if greedy else rng,
            sampler_top_k=generation.top_k,
        )
        replayed_id = decision.sampled_token_id
        prefix_matched = history == event.full_history.token_ids
        token_matched = replayed_id == event.sampled_token_id
        replayed_logit = _logit_at(step_logits.logits, replayed_id)
        recorded_logit = event.sampled_logit
        logit_matched = _logit_match(recorded_logit, step_logits.logits, event.sampled_token_id)

        if prefix_matched:
            if recorded_logit is not None:
                compared_a_logit = True
                if logit_matched is not True:
                    logits_ok = False
            if event.top_k:
                compared_a_logit = True
                take = len(event.top_k) if capture_k is None else min(len(event.top_k), capture_k)
                replayed_top = step_logits.top_k_candidates(
                    take,
                    decode=adapter.decode_token,
                )
                if not _topk_bit_identical(event.top_k[:take], replayed_top):
                    logits_ok = False

        if matched_prefix and token_matched and prefix_matched:
            matched_steps += 1
        elif first_unmatched is None:
            first_unmatched = event.step
            matched_prefix = False

        step_rows.append(
            StepReplay(
                step=event.step,
                recorded_token_id=event.sampled_token_id,
                replayed_token_id=replayed_id,
                token_matched=token_matched,
                prefix_matched=prefix_matched,
                recorded_logit=recorded_logit,
                replayed_logit=replayed_logit,
                logit_matched=logit_matched,
            )
        )
        history.append(replayed_id)

    total = len(trace.events)
    token_ids_matched = matched_steps == total and total > 0
    logits_bit_identical = token_ids_matched and compared_a_logit and logits_ok
    bit_identical = token_ids_matched and logits_bit_identical
    status = _status(matched_steps, total)
    if token_ids_matched and not logits_bit_identical:
        notes.append(
            "sampled token ids matched; logits were not bit-identical, so this "
            "is not a full numeric replay"
        )
    env_note = _environment_note(trace, adapter)
    if env_note:
        notes.append(env_note)

    return ReplayResult(
        trace_id=trace_id,
        status=status,
        matched_steps=matched_steps,
        total_steps=total,
        first_unmatched_step=first_unmatched,
        token_ids_matched=token_ids_matched,
        logits_bit_identical=logits_bit_identical,
        bit_identical=bit_identical,
        recorded_revision=recorded_revision,
        replayed_revision=replayed_revision,
        notes=tuple(notes),
        steps=tuple(step_rows),
    )


def _status(matched_steps: int, total_steps: int) -> ReplayStatus:
    if total_steps == 0:
        return "not_replayable"
    if matched_steps == total_steps:
        return "reproduced"
    if matched_steps == 0:
        return "not_reproduced"
    return "partially_reproduced"


def _blocked(
    *,
    trace_id: str,
    recorded_revision: str | None,
    reason: str,
    notes: tuple[str, ...],
    total_steps: int,
    replayed_revision: str | None = None,
) -> ReplayResult:
    return ReplayResult(
        trace_id=trace_id,
        status="not_replayable",
        matched_steps=0,
        total_steps=total_steps,
        recorded_revision=recorded_revision,
        replayed_revision=replayed_revision,
        reason=reason,
        notes=notes,
    )


def _not_replayable_reason(trace: Trace, adapter: RecordableAdapter | None) -> str | None:
    if not trace.events:
        return "trace has no events to replay"
    logits = trace.run_metadata.logits
    if logits.mode == "none":
        return "trace stored logits.mode=none; cannot replay a sampling path without real logits"
    unsupported = _unsupported_sampler(trace.generation_config)
    if unsupported is not None:
        return unsupported
    if not is_greedy(trace.generation_config) and trace.generation_config.seed is None:
        return "sampled trace has no seed; refusing to claim a deterministic replay"
    if adapter is None:
        if trace.model.provider != "huggingface":
            return (
                f"no adapter provided for provider {trace.model.provider!r}; "
                "pass adapter=... or record a huggingface trace"
            )
        if not trace.model.revision:
            return "trace has no Hub revision pin; refusing to replay against a moving name"
        return None
    return _adapter_unusable(trace, adapter)


def _adapter_unusable(trace: Trace, adapter: RecordableAdapter) -> str | None:
    if not adapter.capabilities.supports_replay:
        return "adapter does not support replay"
    if not adapter.capabilities.supports_logits:
        return "adapter cannot return real logits; refusing to invent them"
    if trace.generation_config.seed is not None and not adapter.capabilities.supports_seed:
        return "adapter does not support seed; refusing to pretend that it does"
    return None


def _unsupported_sampler(generation: GenerationConfig) -> str | None:
    if generation.top_p is not None:
        return "top_p sampling is not implemented"
    if generation.repetition_penalty is not None:
        return "repetition_penalty is not implemented"
    return None


def _prompt_ids(trace: Trace, adapter: RecordableAdapter) -> tuple[list[int], list[str]]:
    notes: list[str] = []
    stored = trace.run_metadata.prompt_token_ids
    encoded = adapter.encode(trace.run_metadata.prompt)
    if stored is None:
        notes.append("trace had no prompt_token_ids; encoded the prompt text")
        return encoded, notes
    prompt_ids = list(stored)
    if encoded != prompt_ids:
        notes.append(
            "tokenizer encoding of prompt text differs from stored "
            "prompt_token_ids; replaying from stored ids"
        )
    return prompt_ids, notes


def _logit_at(logits: tuple[float, ...], token_id: int) -> float | None:
    if token_id < 0 or token_id >= len(logits):
        return None
    return float(logits[token_id])


def _logit_match(
    recorded: float | None,
    replay_logits: tuple[float, ...],
    token_id: int,
) -> bool | None:
    if recorded is None:
        return None
    replayed = _logit_at(replay_logits, token_id)
    if replayed is None:
        return False
    return recorded == replayed


def _topk_bit_identical(
    recorded: Sequence[TopKCandidate],
    replayed: Sequence[TopKCandidate],
) -> bool:
    if len(recorded) != len(replayed):
        return False
    for rec, rep in zip(recorded, replayed, strict=True):
        if rec.token_id != rep.token_id:
            return False
        if rec.logit is not None and rec.logit != rep.logit:
            return False
    return True


def _requested_device(trace: Trace) -> str:
    raw = (trace.environment.device or "cpu").strip().lower()
    if not raw:
        return "cpu"
    return raw.split(",", 1)[0]


def _replay_device(requested: str) -> tuple[str, str | None]:
    base = requested.split(":", 1)[0]
    if base in {"", "cpu"} or requested in _CPU_DEVICES:
        return "cpu", None
    try:
        import torch
    except ImportError:
        return "cpu", (
            f"trace device was {requested}; torch is unavailable so replay uses cpu. "
            "Bit-identical logits are not expected."
        )
    if base == "cuda" and not torch.cuda.is_available():
        return "cpu", (
            "trace device was cuda; CUDA is unavailable so replay uses cpu. "
            "Bit-identical logits are not expected."
        )
    if base == "mps":
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        available = bool(mps is not None and mps.is_available())
        if not available:
            return "cpu", (
                "trace device was mps; MPS is unavailable so replay uses cpu. "
                "Bit-identical logits are not expected."
            )
    return requested, None


def _environment_note(trace: Trace, adapter: RecordableAdapter) -> str | None:
    current = adapter.environment()
    recorded = trace.environment
    mismatches: list[str] = []
    if recorded.device and current.device and recorded.device != current.device:
        mismatches.append(f"device {recorded.device!r} vs {current.device!r}")
    if recorded.accelerator and current.accelerator and recorded.accelerator != current.accelerator:
        mismatches.append(f"accelerator {recorded.accelerator!r} vs {current.accelerator!r}")
    rec_libs = recorded.library_versions
    cur_libs = current.library_versions
    for name in ("torch", "transformers"):
        rec_ver = rec_libs.get(name)
        cur_ver = cur_libs.get(name)
        if rec_ver and cur_ver and rec_ver != cur_ver:
            mismatches.append(f"{name} {rec_ver} vs {cur_ver}")
    if not mismatches:
        return None
    return (
        "recorded environment differs from this replay runtime ("
        + ", ".join(mismatches)
        + "); bit-identical logits are not expected"
    )
