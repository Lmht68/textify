FROM nvidia/cuda:12.9.2-cudnn-runtime-ubuntu24.04@sha256:070f8f2672df1b05b84c0409a5fd1d54ddfd646e5b9d8dee7878131271b563fc

COPY --from=ghcr.io/astral-sh/uv:0.12.1@sha256:cf4eedcaa81655197f625739489effcbe71b61ceb1506f332c3facae5deceded /uv /uvx /bin/

ENV UV_PYTHON=python3.12 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/textify/.venv \
    UV_COMPILE_BYTECODE=1 \
    PATH=/opt/textify/.venv/bin:$PATH \
    HF_HOME=/var/cache/textify/huggingface \
    TEXTIFY_HOST=0.0.0.0 \
    TEXTIFY_TEMPORARY_MEDIA_ROOT=/var/lib/textify/media \
    TEXTIFY_WHISPER_MODEL=large-v3-turbo \
    TEXTIFY_WHISPER_REVISION=0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf \
    TEXTIFY_WHISPER_DEVICE=cuda \
    TEXTIFY_WHISPER_DEVICE_INDEX=0 \
    TEXTIFY_WHISPER_COMPUTE_TYPE=float16

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates python3.12 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 textify \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin textify \
    && mkdir --parents "$HF_HOME" /var/lib/textify/media \
    && chown --recursive textify:textify "$HF_HOME" /var/lib/textify

WORKDIR /opt/textify

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

VOLUME ["/var/cache/textify/huggingface", "/var/lib/textify/media"]
EXPOSE 8182
USER 10001:10001
CMD ["textify"]
