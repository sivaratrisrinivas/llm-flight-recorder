"""Eval: does compare name the cause we injected?

Every case records two real traces through ``record_generation`` with a small
deterministic language model (``HashLM``), changing exactly one thing between
side A and side B. Then ``compare_traces`` must either find no event split
(allowed only for causes that may not change any token, like a seed change on
a peaked distribution) or name the injected cause as the first divergence.

Cases are built from dimensions, not hand-picked:
  cause x logit shape x prompt length x sampler x run length x capture_k x base seed

Run: python evals/divergence_eval.py [--json out.json]
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from llmfr.adapters.base import AdapterCapabilities, StepLogits
from llmfr.compare import compare_traces
from llmfr.core.schema import Environment, GenerationConfig, ModelConfig
from llmfr.record.recorder import record_generation
from llmfr.record.sample import softmax

VOCAB = 16

# Cause -> classes compare may report. None means "no event divergence" is
# acceptable because the change may not alter any sampled token.
EXPECTED: dict[str, set[str | None]] = {
    "identical": {None},
    "seed": {None, "sampling"},
    "temperature": {None, "decoding config"},
    "top_k": {None, "decoding config"},
    "max_new_tokens": {"decoding config"},
    "prompt_text": {"prompt/history"},
    "tokenizer": {"tokenizer"},
    "visible_window": {"model-visible context"},
    "model_revision": {"model/version"},
    "runtime_logits": {"raw-logit"},
}


class HashLM:
    """Context-dependent logits from a hash of the visible tokens."""

    def __init__(
        self,
        *,
        weights: str,
        sharpness: float,
        max_visible_tokens: int = 1024,
        tokenizer_offset: int = 0,
        noise: float = 0.0,
        name: str = "hash-lm",
        revision: str = "r1",
    ) -> None:
        self.weights = weights
        self.sharpness = sharpness
        self._max_visible = max_visible_tokens
        self.tokenizer_offset = tokenizer_offset
        self.noise = noise
        self._name = name
        self._revision = revision

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            supports_logits=True,
            supports_logprobs=True,
            supports_attention=False,
            supports_hidden_states=False,
            supports_seed=True,
            supports_replay=True,
        )

    @property
    def model_config(self) -> ModelConfig:
        return ModelConfig(provider="eval", name=self._name, revision=self._revision)

    @property
    def vocab_size(self) -> int:
        return VOCAB

    @property
    def max_visible_tokens(self) -> int:
        return self._max_visible

    def environment(self) -> Environment:
        return Environment(device="cpu", accelerator="cpu")

    def encode(self, text: str) -> list[int]:
        return [(ord(ch) + self.tokenizer_offset) % VOCAB for ch in text]

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(self.decode_token(t) for t in token_ids)

    def decode_token(self, token_id: int) -> str:
        return f"<{int(token_id)}>"

    def next_token_logits(self, model_visible_context: Sequence[int]) -> StepLogits:
        requested = tuple(int(t) for t in model_visible_context)
        visible = requested[-self._max_visible :]
        digest = hashlib.sha256((self.weights + repr(visible)).encode()).digest()
        logits = tuple(
            self.sharpness * ((digest[i] / 255.0) - 0.5) + self.noise * (i % 3 == 0)
            for i in range(VOCAB)
        )
        probs = softmax(logits)
        logprobs = tuple(math.log(p) if p > 0 else float("-inf") for p in probs)
        return StepLogits(
            token_ids=visible,
            requested_token_ids=requested,
            logits=logits,
            logprobs=logprobs,
            truncated=len(visible) < len(requested),
        )


def build_cases() -> list[dict[str, Any]]:
    dims = itertools.product(
        EXPECTED.keys(),
        (("peaked", 12.0), ("medium", 4.0), ("flat", 1.0)),
        (
            ("short", "What is 7 plus 5?"),
            (
                "long",
                "A farmer has 17 sheep. All but 9 run away. How many are left? Think step by step.",
            ),
        ),
        ("greedy", "sampled"),
        (8, 24),
        (1, 5),
        ("w1", "w2"),
    )
    cases = []
    for cause, (shape, sharp), (plen, prompt), sampler, length, k, base in dims:
        # Greedy decoding ignores seed and temperature, so those causes only
        # make sense with sampling.
        if cause in ("seed", "temperature", "top_k") and sampler == "greedy":
            continue
        cases.append(
            {
                "id": f"{cause}-{shape}-{plen}-{sampler}-L{length}-k{k}-{base}",
                "cause": cause,
                "shape": shape,
                "sharpness": sharp,
                "prompt_len": plen,
                "prompt": prompt,
                "sampler": sampler,
                "length": length,
                "capture_k": k,
                "base": base,
            }
        )
    return cases


def _gen(case: dict[str, Any], **over: Any) -> GenerationConfig:
    greedy = case["sampler"] == "greedy"
    base = {
        "max_new_tokens": case["length"],
        "do_sample": not greedy,
        "temperature": 0.0 if greedy else 1.0,
        "seed": 1,
    }
    base.update(over)
    return GenerationConfig(**base)


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    cause = case["cause"]
    lm_a = HashLM(weights=case["base"], sharpness=case["sharpness"])
    lm_b = HashLM(weights=case["base"], sharpness=case["sharpness"])
    gen_a = _gen(case)
    gen_b = _gen(case)
    prompt_b = case["prompt"]
    if cause == "seed":
        gen_b = _gen(case, seed=2)
    elif cause == "temperature":
        gen_a = _gen(case, temperature=0.7)
        gen_b = _gen(case, temperature=1.3)
    elif cause == "top_k":
        gen_b = _gen(case, top_k=2)
    elif cause == "max_new_tokens":
        gen_b = _gen(case, max_new_tokens=case["length"] // 2)
    elif cause == "prompt_text":
        prompt_b = case["prompt"].replace("7", "8").replace("17", "18")
    elif cause == "tokenizer":
        lm_b = HashLM(weights=case["base"], sharpness=case["sharpness"], tokenizer_offset=3)
    elif cause == "visible_window":
        n_prompt = len(case["prompt"])
        lm_a = HashLM(weights=case["base"], sharpness=case["sharpness"], max_visible_tokens=4096)
        lm_b = HashLM(
            weights=case["base"], sharpness=case["sharpness"], max_visible_tokens=n_prompt - 4
        )
    elif cause == "model_revision":
        lm_b = HashLM(weights=case["base"] + "-v2", sharpness=case["sharpness"], revision="r2")
    elif cause == "runtime_logits":
        lm_b = HashLM(weights=case["base"], sharpness=case["sharpness"], noise=0.25)
    trace_a = record_generation(
        lm_a, case["prompt"], generation=gen_a, capture_k=case["capture_k"], persist=False
    )
    trace_b = record_generation(
        lm_b, prompt_b, generation=gen_b, capture_k=case["capture_k"], persist=False
    )
    result = compare_traces(trace_a, trace_b)
    first = result.first_divergence
    observed = None if first is None else first.classification
    ok = observed in EXPECTED[cause]
    if cause == "identical":
        ok = ok and result.identical
    return {
        "id": case["id"],
        "cause": cause,
        "observed": observed,
        "step": None if first is None else first.step,
        "pass": ok,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="write per-case results here")
    parser.add_argument("--min-pass-rate", type=float, default=1.0)
    args = parser.parse_args(argv)
    cases = build_cases()
    rows = [run_case(c) for c in cases]
    by_cause: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_cause[row["cause"]]["pass" if row["pass"] else "fail"] += 1
        by_cause[row["cause"]][f"observed={row['observed']}"] += 1
    n_pass = sum(r["pass"] for r in rows)
    print(f"divergence eval: {n_pass}/{len(rows)} cases name the injected cause")
    for cause in EXPECTED:
        c = by_cause[cause]
        detail = ", ".join(f"{k}:{v}" for k, v in sorted(c.items()) if k.startswith("observed"))
        print(f"  {cause:15s} pass {c['pass']}/{c['pass'] + c['fail']}  ({detail})")
    for row in rows:
        if not row["pass"]:
            print(f"  FAIL {row['id']}: observed {row['observed']} at step {row['step']}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"n": len(rows), "n_pass": n_pass, "cases": rows}, fh, indent=1)
    rate = n_pass / len(rows)
    return 0 if rate >= args.min_pass_rate else 1


if __name__ == "__main__":
    sys.exit(main())
