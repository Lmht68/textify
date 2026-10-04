# Textify

Textify is an asynchronous API for transcribing eligible public YouTube,
Facebook, Instagram, TikTok, and X videos.
It uses available YouTube captions and falls back to Faster-Whisper on an
NVIDIA CUDA GPU.

## Requirements

Textify runs as three separate processes backed by an externally managed
PostgreSQL database:

| Role | Command | Additional dependency |
| --- | --- | --- |
| API | `textify` | None |
| Reconciler | `textify-reconciler` | Redis |
| Worker | `textify-worker` | Redis and one NVIDIA CUDA GPU |

Source deployments require Python 3.12, [uv](https://docs.astral.sh/uv/), and
FFmpeg.
Container deployments require Docker and NVIDIA Container Toolkit for the
worker.

## Configuration

For a source deployment, copy the environment template and update it:

```shell
cp .env.example .env
```

At minimum, configure:

- `TEXTIFY_DATABASE_URL` for every role. It must use the
  `postgresql+asyncpg` scheme.
- `TEXTIFY_BROKER_URL` for the reconciler and worker.
- `TEXTIFY_GPU_IDENTITY` for the worker. Use the stable physical GPU UUID from:

  ```shell
  nvidia-smi --query-gpu=uuid --format=csv,noheader
  ```

See `.env.example` for optional listener, queue, model, storage, timeout, and
inference settings.
Keep `.env` private because it may contain database, Redis, or Hugging Face
credentials.

## Run from source

Install dependencies and apply migrations:

```shell
uv sync --frozen
uv run alembic upgrade head
```

Start each role in a separate terminal or process supervisor:

```shell
uv run textify
uv run textify-reconciler
uv run textify-worker
```

The API listens on `http://127.0.0.1:8182` by default.
Check readiness with:

```shell
curl --fail http://127.0.0.1:8182/health
```

The worker downloads the configured Whisper model on first startup.

## Deploy with Docker

Build one image for all roles:

```shell
docker build -t textify .
```

The examples below assume the PostgreSQL and Redis hostnames are reachable
from the containers.
Attach the containers to the appropriate Docker network when those services
also run in Docker.

Apply migrations once before starting or updating the roles:

```shell
docker run --rm \
  -e TEXTIFY_DATABASE_URL='postgresql+asyncpg://textify:change-me@postgres:5432/textify' \
  textify \
  alembic upgrade head
```

Start the API:

```shell
docker run -d \
  --name textify-api \
  --restart unless-stopped \
  -p 8182:8182 \
  -e TEXTIFY_DATABASE_URL='postgresql+asyncpg://textify:change-me@postgres:5432/textify' \
  textify
```

Start the reconciler:

```shell
docker run -d \
  --name textify-reconciler \
  --restart unless-stopped \
  -e TEXTIFY_DATABASE_URL='postgresql+asyncpg://textify:change-me@postgres:5432/textify' \
  -e TEXTIFY_BROKER_URL='redis://redis:6379/0' \
  textify \
  textify-reconciler
```

Start one worker for each GPU, using a distinct GPU identity:

```shell
docker run -d \
  --name textify-worker-0 \
  --restart unless-stopped \
  --gpus 'device=0' \
  -e TEXTIFY_DATABASE_URL='postgresql+asyncpg://textify:change-me@postgres:5432/textify' \
  -e TEXTIFY_BROKER_URL='redis://redis:6379/0' \
  -e TEXTIFY_GPU_IDENTITY='GPU-REPLACE-WITH-NVIDIA-UUID' \
  -v textify-model-cache:/var/cache/textify/huggingface \
  -v textify-worker-0-state:/var/lib/textify \
  textify \
  textify-worker
```

Use a process supervisor or container orchestrator to keep all three roles
running.
Each role checks the expected database migration at startup; roles do not
apply migrations automatically.
