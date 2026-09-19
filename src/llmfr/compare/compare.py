"""Compare two stored traces and name the first causal divergence.

Walks events in step order. Classification follows the M3 recorder pipeline:
full_history, model_visible_context, raw logits, decoding config,
probabilities, sample. Later context or logit diffs after that point are
downstream, not a new root cause.
"""

from __future__ import annotations

import json
from typing import Any

from llmfr.compare.result import (
    CompareResult,
    ConfigDiff,
    DivergenceClass,
    FirstDivergence,
    StepDiff,
)
from llmfr.core.schema import Event, GenerationConfig, ModelConfig, Trace

_DECODING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "do_sample",
    "repetition_penalty",
    "stop",
)

_MODEL_IDENTITY_FIELDS = ("provider", "name", "revision")


def compare_traces(trace_a: Trace, trace_b: Trace) -> CompareResult:
    """Deterministic compare of two stored traces. No live model call."""
    notes: list[str] = []
    config_diffs = tuple(_config_diffs(trace_a, trace_b))
    _capture_notes(trace_a, trace_b, notes)

    step_diffs: list[StepDiff] = []
    first: FirstDivergence | None = None
    downstream: list[StepDiff] = []
    total = max(len(trace_a.events), len(trace_b.events))

    for step in range(total):
        event_a = _event_at(trace_a, step)
        event_b = _event_at(trace_b, step)
        differences = tuple(_event_differences(event_a, event_b))
        if not differences:
            continue
        row = StepDiff(
            step=step,
            role="downstream" if first is not None else "first",
            differences=differences,
            a_sampled_token_id=None if event_a is None else event_a.sampled_token_id,
            b_sampled_token_id=None if event_b is None else event_b.sampled_token_id,
            a_sampled_token=None if event_a is None else event_a.sampled_token,
            b_sampled_token=None if event_b is None else event_b.sampled_token,
        )
        step_diffs.append(row)
        if first is None:
            classification, reason = _classify_first(
                step, event_a, event_b, trace_a, trace_b, differences
            )
            first = FirstDivergence(
                step=step,
                classification=classification,
                differences=differences,
                reason=reason,
                a_sampled_token_id=row.a_sampled_token_id,
                b_sampled_token_id=row.b_sampled_token_id,
                a_sampled_token=row.a_sampled_token,
                b_sampled_token=row.b_sampled_token,
            )
        else:
            downstream.append(row)

    identical = first is None and not config_diffs
    if first is not None and downstream:
        notes.append(
            "Later context and logit diffs are downstream of the first "
            "divergence, not a separate root cause."
        )
    return CompareResult(
        trace_a=str(trace_a.run_metadata.trace_id),
        trace_b=str(trace_b.run_metadata.trace_id),
        identical=identical,
        config_diffs=config_diffs,
        first_divergence=first,
        downstream=tuple(downstream),
        notes=tuple(notes),
        steps=tuple(step_diffs),
    )


def _event_at(trace: Trace, step: int) -> Event | None:
    if step < 0 or step >= len(trace.events):
        return None
    return trace.events[step]


def _event_differences(event_a: Event | None, event_b: Event | None) -> list[str]:
    if event_a is None or event_b is None:
        return ["length"]
    diffs: list[str] = []
    if event_a.full_history.token_ids != event_b.full_history.token_ids:
        diffs.append("full_history")
    elif event_a.full_history.text != event_b.full_history.text:
        diffs.append("full_history_text")
    visible_a = event_a.model_visible_context
    visible_b = event_b.model_visible_context
    if visible_a.token_ids != visible_b.token_ids or visible_a.truncated != visible_b.truncated:
        diffs.append("model_visible_context")
    if _logit_fields_differ(event_a, event_b):
        diffs.append("raw_logits")
    if _prob_fields_differ(event_a, event_b):
        diffs.append("probabilities")
    if (
        event_a.sampled_token_id != event_b.sampled_token_id
        or event_a.sampled_token != event_b.sampled_token
    ):
        diffs.append("sampled_token")
    return diffs


def _logit_fields_differ(event_a: Event, event_b: Event) -> bool:
    if len(event_a.top_k) != len(event_b.top_k):
        return True
    for cand_a, cand_b in zip(event_a.top_k, event_b.top_k, strict=True):
        if cand_a.token_id != cand_b.token_id:
            return True
        if cand_a.logit != cand_b.logit:
            return True
    if event_a.sampled_token_id == event_b.sampled_token_id:
        return event_a.sampled_logit != event_b.sampled_logit
    return False


def _prob_fields_differ(event_a: Event, event_b: Event) -> bool:
    if event_a.sampled_prob != event_b.sampled_prob:
        return True
    if event_a.sampled_logprob != event_b.sampled_logprob:
        return True
    if len(event_a.top_k) != len(event_b.top_k):
        return False
    for cand_a, cand_b in zip(event_a.top_k, event_b.top_k, strict=True):
        if cand_a.prob != cand_b.prob:
            return True
        if cand_a.logprob != cand_b.logprob:
            return True
    return False


def _classify_first(
    step: int,
    event_a: Event | None,
    event_b: Event | None,
    trace_a: Trace,
    trace_b: Trace,
    differences: tuple[str, ...],
) -> tuple[DivergenceClass, str]:
    if "length" in differences:
        if _length_policy_differs(trace_a.generation_config, trace_b.generation_config):
            return (
                "decoding config",
                "traces share a matching prefix; max_new_tokens or stop ended one run earlier",
            )
        return (
            "unknown/runtime",
            "traces have different lengths after a matching prefix, "
            "without a sampler-length explanation",
        )

    assert event_a is not None and event_b is not None

    if step == 0 and trace_a.run_metadata.prompt != trace_b.run_metadata.prompt:
        return (
            "prompt/history",
            "prompt text differs at the start of the traces",
        )

    if "full_history" in differences:
        if _tokenizer_explains_history(step, trace_a, trace_b):
            return (
                "tokenizer",
                "same prompt text, different tokenizer or prompt_token_ids at this full_history",
            )
        return (
            "prompt/history",
            "prompt text or full_history token ids differ at this step",
        )

    if "full_history_text" in differences:
        if trace_a.run_metadata.prompt != trace_b.run_metadata.prompt:
            return (
                "prompt/history",
                "prompt text differs; full_history token ids match",
            )
        return (
            "tokenizer",
            "full_history token ids match; decoded text differs",
        )

    if "model_visible_context" in differences:
        return (
            "model-visible context",
            "full_history matches; model_visible_context differs (window or truncation)",
        )

    token_or_score_diff = any(
        name in differences for name in ("raw_logits", "probabilities", "sampled_token")
    )
    if _model_identity_differs(trace_a.model, trace_b.model) and token_or_score_diff:
        return (
            "model/version",
            "model id or revision differs at the first step where scores or tokens split",
        )

    if "raw_logits" in differences:
        if _logits_uncomparable(trace_a, trace_b):
            return _classify_without_logits(trace_a, trace_b, differences)
        return (
            "raw-logit",
            "same visible context and model identity; captured top-k or sampled logits differ",
        )

    if _decoding_config_differs(trace_a.generation_config, trace_b.generation_config) and (
        "sampled_token" in differences or "probabilities" in differences
    ):
        return (
            "decoding config",
            "captured logits match; temperature, do_sample, or other sampler settings differ",
        )

    if "probabilities" in differences:
        return (
            "probability distribution",
            "captured logits and decoding config match; stored probabilities differ",
        )

    if "sampled_token" in differences:
        if event_a.sampled_token_id == event_b.sampled_token_id:
            return (
                "tokenizer",
                "sampled token ids match; decoded token strings differ",
            )
        if _logits_uncomparable(trace_a, trace_b):
            return _classify_without_logits(trace_a, trace_b, differences)
        return (
            "sampling",
            "same context, captured logits, and decoding config; sampled tokens differ",
        )

    return (
        "unknown/runtime",
        "events differ without a classifier that the stored fields can support",
    )


def _classify_without_logits(
    trace_a: Trace, trace_b: Trace, differences: tuple[str, ...]
) -> tuple[DivergenceClass, str]:
    if _decoding_config_differs(trace_a.generation_config, trace_b.generation_config) and (
        "sampled_token" in differences or "probabilities" in differences
    ):
        return (
            "decoding config",
            "logits were not comparable; sampler settings differ and tokens or probabilities split",
        )
    if "sampled_token" in differences and _seed_differs(trace_a, trace_b):
        return (
            "sampling",
            "logits were not comparable; seeds differ and sampled tokens differ",
        )
    return (
        "unknown/runtime",
        "tokens or scores differ without two real logit vectors to compare; "
        "refusing to invent logits",
    )


def _tokenizer_explains_history(step: int, trace_a: Trace, trace_b: Trace) -> bool:
    meta_a = trace_a.run_metadata
    meta_b = trace_b.run_metadata
    if meta_a.prompt != meta_b.prompt:
        return False
    tokenizer_name_differs = trace_a.model.tokenizer != trace_b.model.tokenizer
    prompt_ids_differs = meta_a.prompt_token_ids != meta_b.prompt_token_ids
    if step == 0:
        return tokenizer_name_differs or prompt_ids_differs
    return tokenizer_name_differs and prompt_ids_differs


def _model_identity_differs(model_a: ModelConfig, model_b: ModelConfig) -> bool:
    return any(getattr(model_a, name) != getattr(model_b, name) for name in _MODEL_IDENTITY_FIELDS)


def _decoding_config_differs(gen_a: GenerationConfig, gen_b: GenerationConfig) -> bool:
    return any(getattr(gen_a, name) != getattr(gen_b, name) for name in _DECODING_FIELDS)


def _length_policy_differs(gen_a: GenerationConfig, gen_b: GenerationConfig) -> bool:
    return gen_a.max_new_tokens != gen_b.max_new_tokens or gen_a.stop != gen_b.stop


def _seed_differs(trace_a: Trace, trace_b: Trace) -> bool:
    return trace_a.generation_config.seed != trace_b.generation_config.seed


def _logits_uncomparable(trace_a: Trace, trace_b: Trace) -> bool:
    return trace_a.run_metadata.logits.mode == "none" or trace_b.run_metadata.logits.mode == "none"


def _capture_notes(trace_a: Trace, trace_b: Trace, notes: list[str]) -> None:
    mode_a = trace_a.run_metadata.logits.mode
    mode_b = trace_b.run_metadata.logits.mode
    if mode_a == "none" or mode_b == "none":
        notes.append("one or both traces stored logits.mode=none; compare will not invent logits")
    env_a = trace_a.environment
    env_b = trace_b.environment
    if env_a.device and env_b.device and env_a.device != env_b.device:
        notes.append(f"environment.device differs: {env_a.device} vs {env_b.device}")
    if env_a.accelerator and env_b.accelerator and env_a.accelerator != env_b.accelerator:
        notes.append(f"environment.accelerator differs: {env_a.accelerator} vs {env_b.accelerator}")


def _config_diffs(trace_a: Trace, trace_b: Trace) -> list[ConfigDiff]:
    rows: list[ConfigDiff] = []
    pairs: list[tuple[str, Any, Any]] = [
        ("run_metadata.prompt", trace_a.run_metadata.prompt, trace_b.run_metadata.prompt),
        (
            "run_metadata.prompt_token_ids",
            trace_a.run_metadata.prompt_token_ids,
            trace_b.run_metadata.prompt_token_ids,
        ),
        (
            "run_metadata.logits.mode",
            trace_a.run_metadata.logits.mode,
            trace_b.run_metadata.logits.mode,
        ),
        ("run_metadata.logits.k", trace_a.run_metadata.logits.k, trace_b.run_metadata.logits.k),
        (
            "run_metadata.logits.unavailable_reason",
            trace_a.run_metadata.logits.unavailable_reason,
            trace_b.run_metadata.logits.unavailable_reason,
        ),
        ("run_metadata.source", trace_a.run_metadata.source, trace_b.run_metadata.source),
        ("model.provider", trace_a.model.provider, trace_b.model.provider),
        ("model.name", trace_a.model.name, trace_b.model.name),
        ("model.revision", trace_a.model.revision, trace_b.model.revision),
        ("model.tokenizer", trace_a.model.tokenizer, trace_b.model.tokenizer),
        ("model.dtype", trace_a.model.dtype, trace_b.model.dtype),
        ("model.architecture", trace_a.model.architecture, trace_b.model.architecture),
        (
            "generation_config.seed",
            trace_a.generation_config.seed,
            trace_b.generation_config.seed,
        ),
        (
            "generation_config.temperature",
            trace_a.generation_config.temperature,
            trace_b.generation_config.temperature,
        ),
        (
            "generation_config.top_p",
            trace_a.generation_config.top_p,
            trace_b.generation_config.top_p,
        ),
        (
            "generation_config.top_k",
            trace_a.generation_config.top_k,
            trace_b.generation_config.top_k,
        ),
        (
            "generation_config.max_new_tokens",
            trace_a.generation_config.max_new_tokens,
            trace_b.generation_config.max_new_tokens,
        ),
        (
            "generation_config.do_sample",
            trace_a.generation_config.do_sample,
            trace_b.generation_config.do_sample,
        ),
        (
            "generation_config.repetition_penalty",
            trace_a.generation_config.repetition_penalty,
            trace_b.generation_config.repetition_penalty,
        ),
        (
            "generation_config.stop",
            trace_a.generation_config.stop,
            trace_b.generation_config.stop,
        ),
        ("environment.device", trace_a.environment.device, trace_b.environment.device),
        (
            "environment.accelerator",
            trace_a.environment.accelerator,
            trace_b.environment.accelerator,
        ),
        (
            "environment.python_version",
            trace_a.environment.python_version,
            trace_b.environment.python_version,
        ),
        ("environment.platform", trace_a.environment.platform, trace_b.environment.platform),
    ]
    for field, value_a, value_b in pairs:
        if value_a != value_b:
            rows.append(ConfigDiff(field=field, a=_fmt_value(value_a), b=_fmt_value(value_b)))
    rows.extend(_library_version_diffs(trace_a, trace_b))
    return rows


def _library_version_diffs(trace_a: Trace, trace_b: Trace) -> list[ConfigDiff]:
    versions_a = trace_a.environment.library_versions
    versions_b = trace_b.environment.library_versions
    rows: list[ConfigDiff] = []
    for name in sorted(set(versions_a) | set(versions_b)):
        value_a = versions_a.get(name)
        value_b = versions_b.get(name)
        if value_a != value_b:
            rows.append(
                ConfigDiff(
                    field=f"environment.library_versions.{name}",
                    a=_fmt_value(value_a),
                    b=_fmt_value(value_b),
                )
            )
    return rows


def _fmt_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)
