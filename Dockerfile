# GS-T26: reproducible latency / study job.
# Default target is `smoke` (fake adapter, no torch). `--target hf` adds CPU torch
# for tiny-gpt2 or the portfolio Qwen pin. Do not treat smoke/tiny timings as the
# checked-in Qwen table in docs/findings/gs-t22s-latency.md.
FROM python:3.12-slim AS common

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TOKENIZERS_PARALLELISM=false \
    LLMFR_OUT=/out

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
COPY examples/demo ./examples/demo
COPY examples/study ./examples/study
COPY examples/findings/gs-t22s ./examples/findings/gs-t22s

RUN chmod +x /app/scripts/container_latency.sh

FROM common AS hf
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir --upgrade-strategy only-if-needed ".[hf]"
ENTRYPOINT ["/app/scripts/container_latency.sh"]
CMD ["--backend", "tiny-gpt2", "--warmup", "0", "--trials", "1", "--max-new-tokens", "4"]

FROM common AS smoke
RUN pip install --no-cache-dir .
ENTRYPOINT ["/app/scripts/container_latency.sh"]
CMD ["--backend", "fake"]
