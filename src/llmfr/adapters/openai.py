"""OpenAI Chat Completions adapter. Optional extra `openai`.

Captures `top_logprobs` when the API returns them. Never invents a full-vocab
logit vector. `supports_logits=False` and `supports_replay=False` stay honest:
a short ranked logprob list is not raw logits, and hosted sampling is not a
pinned HF CPU replay.
"""

from __future__ import annotations

import math
import os
import platform
from collections.abc import Sequence
from typing import Any

from llmfr.adapters.base import (
    AdapterCapabilities,
    HostedCompletion,
    HostedTokenStep,
    StepLogits,
)
from llmfr.core.schema import Environment, GenerationConfig, ModelConfig, TopKCandidate
from llmfr.core.version import DEFAULT_TOP_K, MAX_TOP_K
from llmfr.record.sample import is_greedy

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
OPENAI_MAX_TOP_LOGPROBS = 20
OPENAI_KEY_ENV = "OPENAI_API_KEY"
FALLBACK_TIKTOKEN_ENCODING = "o200k_base"

_OPENAI_CAPABILITIES = AdapterCapabilities(
    supports_logits=False,
    supports_logprobs=True,
    supports_attention=False,
    supports_hidden_states=False,
    supports_seed=False,
    supports_replay=False,
)

_OPENAI_INSTALL_HINT = (
    "OpenAI adapter requires the openai extra (openai and tiktoken). "
    "Install with: pip install -e '.[openai]' "
    "(packaged: pip install 'llmfr[openai]'). "
    f"Set {OPENAI_KEY_ENV} in the environment. llmfr does not accept --api-key."
)

_UNAVAILABLE_NO_LOGPROBS = "OpenAI response did not include top_logprobs; refusing to invent them"
_UNAVAILABLE_UNSUPPORTED = "OpenAI model or API rejected logprobs; refusing to invent scores"


class OpenAIExtraMissingError(ImportError):
    """Raised when openai/tiktoken are not installed."""

    def __init__(self) -> None:
        super().__init__(_OPENAI_INSTALL_HINT)


class OpenAIAPIKeyMissingError(RuntimeError):
    """Raised when OPENAI_API_KEY is missing from the environment."""

    def __init__(self) -> None:
        super().__init__(
            f"{OPENAI_KEY_ENV} is not set. llmfr reads the key from the environment "
            "only and does not accept --api-key."
        )


def _import_backend() -> tuple[Any, Any]:
    try:
        import openai
        import tiktoken
    except ImportError as exc:
        raise OpenAIExtraMissingError from exc
    return openai, tiktoken


def looks_like_openai_model(model_id: str) -> bool:
    """True for OpenAI API model names, not Hugging Face `org/name` ids.

    `gpt2` (no hyphen after gpt) stays Hugging Face. `openai:gpt-4o-mini`
    and `gpt-4o-mini` select OpenAI.
    """
    name = model_id.strip()
    lowered = name.lower()
    if lowered.startswith("openai:"):
        return True
    if "/" in name:
        return False
    if lowered.startswith("gpt-"):
        return True
    if lowered.startswith(("o1", "o3", "o4")):
        return True
    return lowered.startswith("chatgpt-")


def openai_model_name(model_id: str) -> str:
    name = model_id.strip()
    if name.lower().startswith("openai:"):
        return name.split(":", 1)[1].strip()
    return name


def resolve_record_provider(provider: str | None, model: str | None) -> str:
    """CLI backend selection. Hugging Face remains the default for local/CI."""
    if provider is not None:
        normalized = provider.strip().lower()
        if normalized in {"hf", "huggingface"}:
            return "huggingface"
        if normalized == "openai":
            return "openai"
        raise ValueError(f"unknown provider {provider!r}; use huggingface or openai")
    if model is not None and looks_like_openai_model(model):
        return "openai"
    return "huggingface"


class OpenAIChatAdapter:
    """Chat Completions adapter. Scores come from the API or not at all.

    `complete_prompt` is the recording path: one `chat.completions.create`
    with `logprobs=True` and `top_logprobs=k`. The recorder does not run
    LocalRNG over a fake vocab. `next_token_logits` exists for the
    ModelAdapter protocol and issues a one-token completion; it still
    returns an empty `logits` tuple.

    Chat-template tokens that the hosted model prepends are not visible in
    the API response. Stored `full_history` / `model_visible_context` are
    tiktoken ids of the prompt plus sampled tokens, not a claim that those
    ids were the exact model-visible prefix.
    """

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        *,
        client: Any | None = None,
        encoding: Any | None = None,
        top_logprobs: int = DEFAULT_TOP_K,
    ) -> None:
        if top_logprobs < 1:
            raise ValueError("top_logprobs must be >= 1")
        if top_logprobs > MAX_TOP_K:
            raise ValueError(
                f"top_logprobs must be <= {MAX_TOP_K}; refusing to store an oversized logit payload"
            )
        model_name = openai_model_name(model)
        if not model_name:
            raise ValueError("OpenAI model name must be non-empty")

        openai_mod: Any | None = None
        if client is None or encoding is None:
            openai_mod, tiktoken_mod = _import_backend()
        else:
            tiktoken_mod = None

        if client is None:
            assert openai_mod is not None
            key = _api_key_from_env()
            client = openai_mod.OpenAI(api_key=key)

        if encoding is None:
            assert tiktoken_mod is not None
            encoding = _encoding_for_model(tiktoken_mod, model_name)

        self._client = client
        self._encoding = encoding
        self._model = model_name
        self._top_logprobs = min(top_logprobs, OPENAI_MAX_TOP_LOGPROBS)
        self._openai_version = _package_version(openai_mod, "openai")
        self._tiktoken_version = _package_version(tiktoken_mod, "tiktoken")
        self._model_config = ModelConfig(
            provider="openai",
            name=model_name,
            revision=None,
            tokenizer=_encoding_name(encoding),
            dtype=None,
            architecture=None,
        )

    @property
    def capabilities(self) -> AdapterCapabilities:
        return _OPENAI_CAPABILITIES

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def model_config(self) -> ModelConfig:
        return self._model_config

    def environment(self) -> Environment:
        versions = {}
        if self._openai_version:
            versions["openai"] = self._openai_version
        if self._tiktoken_version:
            versions["tiktoken"] = self._tiktoken_version
        return Environment(
            python_version=platform.python_version(),
            platform=platform.platform(),
            device="api",
            accelerator=None,
            library_versions=versions,
        )

    def encode(self, text: str) -> list[int]:
        try:
            ids = self._encoding.encode(text, allowed_special="all")
        except TypeError:
            ids = self._encoding.encode(text)
        return [int(token_id) for token_id in ids]

    def decode(self, token_ids: Sequence[int]) -> str:
        ids = [int(token_id) for token_id in token_ids]
        try:
            return str(self._encoding.decode(ids, skip_special_tokens=False))
        except TypeError:
            return str(self._encoding.decode(ids))

    def decode_token(self, token_id: int) -> str:
        return self.decode([int(token_id)])

    def complete_prompt(
        self,
        prompt: str,
        *,
        generation: GenerationConfig,
        capture_k: int,
    ) -> HostedCompletion:
        if not prompt:
            raise ValueError("prompt encoded to no tokens")
        prompt_ids = tuple(self.encode(prompt))
        if not prompt_ids:
            raise ValueError("prompt encoded to no tokens")
        max_new = generation.max_new_tokens
        if max_new is None or max_new < 1:
            raise ValueError("max_new_tokens must be >= 1")
        api_k = min(capture_k, self._top_logprobs, OPENAI_MAX_TOP_LOGPROBS)
        if api_k < 1:
            raise ValueError("capture_k must be >= 1")

        greedy = is_greedy(generation)
        temperature = generation.temperature
        if greedy:
            api_temperature = 0.0
        elif temperature is None:
            api_temperature = 1.0
        else:
            api_temperature = float(temperature)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(max_new),
            "temperature": api_temperature,
            "logprobs": True,
            "top_logprobs": api_k,
        }
        if generation.seed is not None:
            kwargs["seed"] = int(generation.seed)
        if generation.stop:
            kwargs["stop"] = list(generation.stop)

        response, requested_logprobs = self._create(kwargs)
        return self._completion_from_response(
            response,
            prompt_ids=prompt_ids,
            capture_k=api_k,
            requested_logprobs=requested_logprobs,
        )

    def next_token_logits(self, model_visible_context: Sequence[int]) -> StepLogits:
        requested = tuple(int(token_id) for token_id in model_visible_context)
        if not requested:
            raise ValueError("model_visible_context must be non-empty")
        prompt_text = self.decode(requested)
        completion = self.complete_prompt(
            prompt_text,
            generation=GenerationConfig(max_new_tokens=1, do_sample=False, temperature=0.0),
            capture_k=self._top_logprobs,
        )
        if not completion.steps:
            raise RuntimeError("OpenAI returned no tokens; refusing to invent them")
        step = completion.steps[0]
        top = step.top_k or None
        return StepLogits(
            token_ids=requested,
            requested_token_ids=requested,
            logits=(),
            logprobs=None,
            truncated=False,
            top_logprobs=top,
            backend_sampled_token_id=step.token_id,
            backend_sampled_token=step.token,
            backend_sampled_logprob=step.logprob,
        )

    def _create(self, kwargs: dict[str, Any]) -> tuple[Any, bool]:
        try:
            return self._chat_create(kwargs), True
        except Exception as exc:
            if "logprob" not in str(exc).lower():
                raise
            fallback = dict(kwargs)
            fallback.pop("logprobs", None)
            fallback.pop("top_logprobs", None)
            return self._chat_create(fallback), False

    def _chat_create(self, kwargs: dict[str, Any]) -> Any:
        chat = getattr(self._client, "chat", None)
        completions = getattr(chat, "completions", None)
        create = getattr(completions, "create", None)
        if create is None:
            raise RuntimeError("OpenAI client has no chat.completions.create")
        return create(**kwargs)

    def _completion_from_response(
        self,
        response: Any,
        *,
        prompt_ids: tuple[int, ...],
        capture_k: int,
        requested_logprobs: bool,
    ) -> HostedCompletion:
        choice = _first_choice(response)
        message = _get(choice, "message")
        output_text = _message_text(message)
        content_items = _logprob_content(choice)

        if content_items:
            steps = tuple(
                self._step_from_logprob_item(item, capture_k=capture_k) for item in content_items
            )
            if not output_text:
                output_text = "".join(step.token for step in steps)
            return HostedCompletion(
                prompt_token_ids=prompt_ids,
                output_text=output_text,
                steps=steps,
                logprobs_available=True,
                unavailable_reason=None,
                capture_k=capture_k,
            )

        if not output_text:
            finish = _get(choice, "finish_reason")
            raise RuntimeError(
                "OpenAI returned an empty completion"
                + (f" (finish_reason={finish!r})" if finish is not None else "")
            )
        token_ids = tuple(self.encode(output_text))
        steps = tuple(
            HostedTokenStep(
                token_id=token_id,
                token=self.decode_token(token_id),
                logprob=None,
                top_k=(),
                rank=None,
            )
            for token_id in token_ids
        )
        reason = _UNAVAILABLE_NO_LOGPROBS if requested_logprobs else _UNAVAILABLE_UNSUPPORTED
        return HostedCompletion(
            prompt_token_ids=prompt_ids,
            output_text=output_text,
            steps=steps,
            logprobs_available=False,
            unavailable_reason=reason,
            capture_k=None,
        )

    def _step_from_logprob_item(self, item: Any, *, capture_k: int) -> HostedTokenStep:
        token = str(_get(item, "token") or "")
        if token == "":
            raise RuntimeError("OpenAI logprobs item had no token string; refusing to invent one")
        token_id = self._token_id(token)
        raw_logprob = _get(item, "logprob")
        logprob = None if raw_logprob is None else float(raw_logprob)
        raw_top = _get(item, "top_logprobs") or ()
        ranked = _rank_top_logprobs(raw_top, capture_k=capture_k, token_id_of=self._token_id)
        rank = next(
            (candidate.rank for candidate in ranked if candidate.token_id == token_id), None
        )
        if rank is None:
            rank = next((candidate.rank for candidate in ranked if candidate.token == token), None)
        return HostedTokenStep(
            token_id=token_id,
            token=token,
            logprob=logprob,
            top_k=ranked,
            rank=rank,
        )

    def _token_id(self, token: str) -> int:
        encode_one = getattr(self._encoding, "encode_single_token", None)
        if encode_one is not None:
            try:
                return int(encode_one(token))
            except (KeyError, ValueError, TypeError):
                pass
        ids = self.encode(token)
        if len(ids) == 1:
            return ids[0]
        raise RuntimeError(
            f"OpenAI token {token!r} does not map to a single tokenizer id; "
            "refusing to invent a token id"
        )


def _api_key_from_env() -> str:
    key = os.environ.get(OPENAI_KEY_ENV)
    if key is None or not str(key).strip():
        raise OpenAIAPIKeyMissingError()
    return str(key).strip()


def _encoding_for_model(tiktoken_mod: Any, model: str) -> Any:
    try:
        return tiktoken_mod.encoding_for_model(model)
    except KeyError:
        return tiktoken_mod.get_encoding(FALLBACK_TIKTOKEN_ENCODING)


def _encoding_name(encoding: Any) -> str:
    name = getattr(encoding, "name", None)
    if isinstance(name, str) and name:
        return name
    return FALLBACK_TIKTOKEN_ENCODING


def _package_version(module: Any | None, distribution: str) -> str | None:
    if module is not None:
        version = getattr(module, "__version__", None)
        if isinstance(version, str) and version:
            return version
    try:
        from importlib.metadata import version as pkg_version
    except ImportError:
        return None
    try:
        return pkg_version(distribution)
    except Exception:
        return None


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _first_choice(response: Any) -> Any:
    choices = _get(response, "choices") or ()
    if not choices:
        raise RuntimeError("OpenAI response had no choices; refusing to invent tokens")
    return choices[0]


def _message_text(message: Any) -> str:
    content = _get(message, "content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = _get(item, "text")
            if isinstance(text, str) and text:
                parts.append(text)
        return "".join(parts)
    return str(content)


def _logprob_content(choice: Any) -> list[Any] | None:
    logprobs = _get(choice, "logprobs")
    if logprobs is None:
        return None
    content = _get(logprobs, "content")
    if not content:
        return None
    return list(content)


def _rank_top_logprobs(
    raw_top: Any,
    *,
    capture_k: int,
    token_id_of: Any,
) -> tuple[TopKCandidate, ...]:
    rows: list[tuple[str, float]] = []
    for item in raw_top:
        token = str(_get(item, "token") or "")
        raw_logprob = _get(item, "logprob")
        if token == "" or raw_logprob is None:
            continue
        rows.append((token, float(raw_logprob)))
    rows.sort(key=lambda row: row[1], reverse=True)
    take = rows[:capture_k]
    candidates: list[TopKCandidate] = []
    for rank, (token, logprob) in enumerate(take, start=1):
        prob = min(1.0, max(0.0, math.exp(logprob)))
        candidates.append(
            TopKCandidate(
                rank=rank,
                token_id=int(token_id_of(token)),
                token=token,
                logit=None,
                prob=prob,
                logprob=logprob,
            )
        )
    return tuple(candidates)
