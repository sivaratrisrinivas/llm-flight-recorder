"""Hugging Face causal LM adapter. Optional extra `hf` (CPU torch first)."""

from __future__ import annotations

import platform
import re
from collections.abc import Sequence
from typing import Any

from llmfr.adapters.base import AdapterCapabilities, StepLogits
from llmfr.core.schema import Environment, ModelConfig

DEFAULT_HF_MODEL_ID = "sshleifer/tiny-gpt2"
PORTFOLIO_DEMO_MODEL_ID = "distilbert/distilgpt2"

_HF_CAPABILITIES = AdapterCapabilities(
    supports_logits=True,
    supports_logprobs=True,
    supports_attention=False,
    supports_hidden_states=False,
    supports_seed=False,
    supports_replay=True,
)

_COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")


_HF_INSTALL_HINT = (
    "Hugging Face adapter requires torch and transformers. "
    "Install CPU torch first so pip does not pull the CUDA-default PyPI wheel: "
    "pip install --index-url https://download.pytorch.org/whl/cpu torch "
    "&& pip install --upgrade-strategy only-if-needed -e '.[dev,hf]' "
    "(packaged: same CPU torch command, then "
    "pip install --upgrade-strategy only-if-needed 'llmfr[hf]')."
)


class HuggingFaceExtraMissingError(ImportError):
    """Raised when torch/transformers are not installed."""

    def __init__(self) -> None:
        super().__init__(_HF_INSTALL_HINT)


def _import_backend() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise HuggingFaceExtraMissingError from exc
    return torch, transformers


class HuggingFaceCausalLMAdapter:
    """Local causal LM via `AutoModelForCausalLM`. Defaults to CPU tiny-gpt2.

    `next_token_logits` runs one eval-mode forward pass and returns the last
    position's raw logits. Logprobs are `log_softmax` of those logits, not a
    second model API. Attention and hidden states are not returned here even
    though some HF models can compute them.

    Sampling is Milestone 3. This adapter does not take a seed and does not
    call `torch.manual_seed`; `supports_seed` is False until then.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_HF_MODEL_ID,
        *,
        revision: str | None = None,
        device: str = "cpu",
        max_visible_tokens: int | None = None,
    ) -> None:
        torch, transformers = _import_backend()

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=False
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=False,
            low_cpu_mem_usage=False,
        )
        model.to(device=torch.device(device), dtype=torch.float32)
        model.eval()
        pinned = _resolved_hub_revision(model=model, model_id=model_id, requested=revision)

        config = model.config
        model_max = _max_positions(config)
        if model_max < 1:
            raise ValueError(f"{model_id} reported a non-positive context length")
        if max_visible_tokens is None:
            clip = model_max
        else:
            if max_visible_tokens < 1:
                raise ValueError("max_visible_tokens must be >= 1")
            clip = min(max_visible_tokens, model_max)

        architecture = None
        architectures = getattr(config, "architectures", None)
        if isinstance(architectures, list) and architectures:
            architecture = str(architectures[0])

        tokenizer_name = getattr(tokenizer, "name_or_path", None) or model_id
        dtype_name = str(next(model.parameters()).dtype).replace("torch.", "")

        self._torch = torch
        self._transformers_version = str(transformers.__version__)
        self._model = model
        self._tokenizer = tokenizer
        self._device = torch.device(device)
        self._max_visible_tokens = clip
        self._model_id = model_id
        self._model_config = ModelConfig(
            provider="huggingface",
            name=model_id,
            revision=pinned,
            tokenizer=str(tokenizer_name),
            dtype=dtype_name,
            architecture=architecture,
        )

    @property
    def capabilities(self) -> AdapterCapabilities:
        return _HF_CAPABILITIES

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def max_visible_tokens(self) -> int:
        return self._max_visible_tokens

    @property
    def vocab_size(self) -> int:
        vocab = getattr(self._model.config, "vocab_size", None)
        if isinstance(vocab, int) and vocab > 0:
            return vocab
        raise RuntimeError(f"{self._model_id} config has no vocab_size")

    @property
    def model_config(self) -> ModelConfig:
        return self._model_config

    def environment(self) -> Environment:
        return Environment(
            python_version=platform.python_version(),
            platform=platform.platform(),
            device=str(self._device),
            accelerator="cpu" if self._device.type == "cpu" else self._device.type,
            library_versions={
                "torch": self._torch.__version__,
                "transformers": self._transformers_version,
            },
        )

    def encode(self, text: str) -> list[int]:
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        return [int(token_id) for token_id in ids]

    def decode(self, token_ids: Sequence[int]) -> str:
        return str(self._tokenizer.decode(list(token_ids), skip_special_tokens=False))

    def decode_token(self, token_id: int) -> str:
        return str(self._tokenizer.decode([int(token_id)], skip_special_tokens=False))

    def next_token_logits(self, model_visible_context: Sequence[int]) -> StepLogits:
        requested = tuple(int(token_id) for token_id in model_visible_context)
        if not requested:
            raise ValueError("model_visible_context must be non-empty")
        self._require_in_vocab(requested)
        visible, truncated = self._clip(requested)

        torch = self._torch
        input_ids = torch.tensor([list(visible)], dtype=torch.long, device=self._device)
        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, use_cache=False)
        raw = getattr(outputs, "logits", None)
        if raw is None:
            raise RuntimeError(f"{self._model_id} did not return logits; refusing to invent them")
        last = raw[0, -1].detach().to(dtype=torch.float32, device="cpu")
        if last.ndim != 1:
            raise RuntimeError(f"expected a 1-d vocab vector, got shape {tuple(last.shape)}")
        logits = tuple(float(value) for value in last.tolist())
        logprobs = tuple(float(value) for value in torch.log_softmax(last, dim=-1).tolist())
        return StepLogits(
            token_ids=visible,
            requested_token_ids=requested,
            logits=logits,
            logprobs=logprobs,
            truncated=truncated,
        )

    def _require_in_vocab(self, token_ids: tuple[int, ...]) -> None:
        vocab = self.vocab_size
        for index, token_id in enumerate(token_ids):
            if token_id < 0 or token_id >= vocab:
                raise ValueError(
                    f"model_visible_context token id {token_id} at position {index} "
                    f"is outside [0, {vocab})"
                )

    def _clip(self, token_ids: tuple[int, ...]) -> tuple[tuple[int, ...], bool]:
        limit = self._max_visible_tokens
        if len(token_ids) <= limit:
            return token_ids, False
        return token_ids[-limit:], True


def _resolved_hub_revision(*, model: Any, model_id: str, requested: str | None) -> str:
    """Pin ModelConfig.revision to the Hub commit actually loaded.

    `supports_replay=True` is only honest if this SHA is known. Moving names
    such as `main` are not stored.
    """
    sha = _as_commit_hash(getattr(getattr(model, "config", None), "_commit_hash", None))
    if sha is None:
        sha = _as_commit_hash(getattr(model, "_commit_hash", None))
    if sha is None:
        sha = _hub_commit_sha(model_id, requested)
    if sha is None:
        raise RuntimeError(
            f"could not resolve Hub commit hash for {model_id!r} "
            f"(requested revision={requested!r}); refusing to advertise replay without a pin"
        )
    return sha


def _hub_commit_sha(model_id: str, requested: str | None) -> str | None:
    try:
        from huggingface_hub import model_info
    except ImportError:
        return None
    try:
        info = model_info(model_id, revision=requested)
    except Exception:
        return None
    return _as_commit_hash(getattr(info, "sha", None))


def _as_commit_hash(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not _COMMIT_SHA.fullmatch(text):
        return None
    return text.lower()


def _max_positions(config: Any) -> int:
    for attr in ("n_positions", "max_position_embeddings", "n_ctx"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return 0
