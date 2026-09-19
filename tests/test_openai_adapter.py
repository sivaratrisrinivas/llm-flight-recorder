"""OpenAI adapter tests. Mocked Chat Completions; no live key."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from llmfr.adapters.base import ModelAdapter
from llmfr.adapters.openai import (
    DEFAULT_OPENAI_MODEL,
    OPENAI_KEY_ENV,
    OpenAIAPIKeyMissingError,
    OpenAIChatAdapter,
    OpenAIExtraMissingError,
    OpenAILogprobsUnavailableError,
    looks_like_openai_model,
    openai_model_name,
    resolve_record_provider,
)
from llmfr.core.schema import GenerationConfig
from tests.openai_fakes import (
    FakeEncoding,
    FakeOpenAIClient,
    hello_logprob_tokens,
    make_chat_response,
)


def _adapter(client: FakeOpenAIClient, encoding: FakeEncoding | None = None) -> OpenAIChatAdapter:
    return OpenAIChatAdapter(
        DEFAULT_OPENAI_MODEL,
        client=client,
        encoding=FakeEncoding() if encoding is None else encoding,
    )


def test_capability_flags_do_not_claim_full_vocab_logits() -> None:
    adapter = _adapter(FakeOpenAIClient([make_chat_response(hello_logprob_tokens())]))
    assert isinstance(adapter, ModelAdapter)
    caps = adapter.capabilities
    assert caps.supports_logits is False
    assert caps.supports_logprobs is True
    assert caps.supports_replay is False
    assert caps.supports_seed is False
    assert caps.supports_attention is False
    assert caps.supports_hidden_states is False
    assert adapter.model_config.provider == "openai"
    assert adapter.model_config.name == DEFAULT_OPENAI_MODEL
    assert adapter.model_config.revision is None
    assert adapter.environment().device == "api"


def test_complete_prompt_captures_top_logprobs_not_logits() -> None:
    encoding = FakeEncoding()
    client = FakeOpenAIClient([make_chat_response(hello_logprob_tokens())])
    adapter = _adapter(client, encoding)
    completion = adapter.complete_prompt(
        "Hi",
        generation=GenerationConfig(max_new_tokens=2, do_sample=False, temperature=0.0),
        capture_k=3,
    )
    assert completion.logprobs_available is True
    assert len(completion.steps) == 2
    first = completion.steps[0]
    assert first.token == "He"
    assert first.logprob == -0.1
    assert first.top_k
    assert all(candidate.logit is None for candidate in first.top_k)
    assert [candidate.logprob for candidate in first.top_k] == [-0.1, -1.2, -2.0]
    assert [candidate.token for candidate in first.top_k] == ["He", "Hi", "ho"]
    assert client.calls
    request = client.calls[0]
    assert request["model"] == DEFAULT_OPENAI_MODEL
    assert request["logprobs"] is True
    assert request["top_logprobs"] == 3
    assert request["max_tokens"] == 2
    assert request["temperature"] == 0.0
    assert "api_key" not in request
    dumped = str(request)
    assert "sk-" not in dumped


def test_next_token_logits_leaves_vocab_vector_empty() -> None:
    encoding = FakeEncoding()
    client = FakeOpenAIClient([make_chat_response(hello_logprob_tokens()[:1])])
    adapter = _adapter(client, encoding)
    prompt_ids = adapter.encode("Hi")
    step = adapter.next_token_logits(prompt_ids)
    assert step.logits == ()
    assert step.logprobs is None
    assert step.backend_sampled_token == "He"
    assert step.backend_sampled_logprob == -0.1
    top = step.top_k_candidates(3, decode=adapter.decode_token)
    assert top[0].logit is None
    assert top[0].logprob == -0.1
    with pytest.raises(ValueError, match="no logits"):
        _ = step.greedy_token_id


def test_omitted_logprobs_fail_closed() -> None:
    client = FakeOpenAIClient(
        [make_chat_response(hello_logprob_tokens(), output_text="Hello", include_logprobs=False)]
    )
    adapter = _adapter(client)
    with pytest.raises(OpenAILogprobsUnavailableError, match="per-token logprob content"):
        adapter.complete_prompt(
            "Hi",
            generation=GenerationConfig(max_new_tokens=2, do_sample=False),
            capture_k=5,
        )
    assert len(client.calls) == 1
    assert client.calls[0]["logprobs"] is True
    assert "top_logprobs" in client.calls[0]


def test_empty_logprob_content_fail_closed() -> None:
    client = FakeOpenAIClient(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Hello"),
                        logprobs=SimpleNamespace(content=[]),
                        finish_reason="stop",
                    )
                ]
            )
        ]
    )
    adapter = _adapter(client)
    with pytest.raises(OpenAILogprobsUnavailableError, match="per-token logprob content"):
        adapter.complete_prompt(
            "Hi",
            generation=GenerationConfig(max_new_tokens=2, do_sample=False),
            capture_k=5,
        )


def test_logprob_items_without_scores_fail_closed() -> None:
    client = FakeOpenAIClient(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Hello"),
                        logprobs=SimpleNamespace(
                            content=[
                                SimpleNamespace(token="He", logprob=None, top_logprobs=[]),
                                SimpleNamespace(token="llo", logprob=None, top_logprobs=[]),
                            ]
                        ),
                        finish_reason="stop",
                    )
                ]
            )
        ]
    )
    adapter = _adapter(client)
    with pytest.raises(OpenAILogprobsUnavailableError, match="per-token logprob content"):
        adapter.complete_prompt(
            "Hi",
            generation=GenerationConfig(max_new_tokens=2, do_sample=False),
            capture_k=5,
        )


def test_logprobs_rejected_by_api_fail_closed() -> None:
    client = FakeOpenAIClient(
        errors=[RuntimeError("logprobs are not supported for this model")],
    )
    adapter = _adapter(client)
    with pytest.raises(OpenAILogprobsUnavailableError, match="rejected logprobs"):
        adapter.complete_prompt(
            "Hi",
            generation=GenerationConfig(max_new_tokens=2, do_sample=False),
            capture_k=5,
        )
    assert len(client.calls) == 1
    assert client.calls[0]["logprobs"] is True
    assert client.calls[0]["top_logprobs"] == 5


def test_seed_is_forwarded_not_used_as_local_rng() -> None:
    client = FakeOpenAIClient([make_chat_response(hello_logprob_tokens())])
    adapter = _adapter(client)
    adapter.complete_prompt(
        "Hi",
        generation=GenerationConfig(max_new_tokens=2, do_sample=True, temperature=0.7, seed=99),
        capture_k=3,
    )
    assert client.calls[0]["seed"] == 99
    assert client.calls[0]["temperature"] == 0.7
    assert adapter.capabilities.supports_seed is False
    assert adapter.capabilities.supports_replay is False


def test_missing_api_key_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(OPENAI_KEY_ENV, raising=False)

    class _Openai:
        class OpenAI:
            def __init__(self, **_kwargs: object) -> None:
                raise AssertionError("must not construct a client without an env key")

        __version__ = "1.0"

    class _Tiktoken:
        __version__ = "1.0"

        @staticmethod
        def encoding_for_model(_name: str) -> FakeEncoding:
            return FakeEncoding()

    monkeypatch.setattr("llmfr.adapters.openai._import_backend", lambda: (_Openai(), _Tiktoken()))
    with pytest.raises(OpenAIAPIKeyMissingError, match=OPENAI_KEY_ENV):
        OpenAIChatAdapter()


def test_extra_missing_error_mentions_optional_extra() -> None:
    err = str(OpenAIExtraMissingError())
    assert ".[openai]" in err
    assert "llmfr[openai]" in err
    assert OPENAI_KEY_ENV in err
    assert "--api-key" in err


def test_constructor_requires_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    def _missing() -> tuple[object, object]:
        raise OpenAIExtraMissingError()

    monkeypatch.setattr("llmfr.adapters.openai._import_backend", _missing)
    with pytest.raises(OpenAIExtraMissingError, match=r"\[openai\]"):
        OpenAIChatAdapter()


def test_empty_choices_are_refused() -> None:
    client = FakeOpenAIClient([SimpleNamespace(choices=[])])
    adapter = _adapter(client)
    with pytest.raises(RuntimeError, match="no choices"):
        adapter.complete_prompt(
            "Hi",
            generation=GenerationConfig(max_new_tokens=1, do_sample=False),
            capture_k=5,
        )


def test_provider_selection_keeps_hf_default() -> None:
    assert resolve_record_provider(None, None) == "huggingface"
    assert resolve_record_provider(None, "sshleifer/tiny-gpt2") == "huggingface"
    assert resolve_record_provider(None, "gpt2") == "huggingface"
    assert resolve_record_provider(None, "gpt-4o-mini") == "huggingface"
    assert resolve_record_provider(None, "gpt-neo") == "huggingface"
    assert resolve_record_provider(None, "gpt-j") == "huggingface"
    assert resolve_record_provider(None, "o1-mini") == "huggingface"
    assert resolve_record_provider(None, "openai:gpt-4o-mini") == "openai"
    assert resolve_record_provider("openai", "sshleifer/tiny-gpt2") == "openai"
    assert resolve_record_provider("openai", "gpt-neo") == "openai"
    assert resolve_record_provider("openai", "gpt-4o-mini") == "openai"
    assert resolve_record_provider("huggingface", "gpt-4o-mini") == "huggingface"
    assert resolve_record_provider("huggingface", "openai:gpt-4o-mini") == "huggingface"
    assert resolve_record_provider("hf", None) == "huggingface"
    with pytest.raises(ValueError, match="unknown provider"):
        resolve_record_provider("langchain", None)


def test_openai_model_name_strips_prefix() -> None:
    assert openai_model_name("openai:gpt-4o-mini") == "gpt-4o-mini"
    assert openai_model_name("gpt-4o-mini") == "gpt-4o-mini"
    assert looks_like_openai_model("openai:gpt-4o-mini") is True
    assert looks_like_openai_model("OPENAI:gpt-4o-mini") is True
    assert looks_like_openai_model("gpt-4o-mini") is False
    assert looks_like_openai_model("gpt-neo") is False
    assert looks_like_openai_model("gpt-j") is False
    assert looks_like_openai_model("o1-mini") is False
    assert looks_like_openai_model("chatgpt-4o") is False
    assert looks_like_openai_model("openai-community/gpt2") is False
