# Textify

Textify is an asynchronous API for transcribing public YouTube, Facebook, Instagram, TikTok, and X videos.
It uses available YouTube captions and falls back to Faster-Whisper on an NVIDIA CUDA GPU.

## Requirements

For the recommended Compose setup:

- Docker Engine
- Docker Compose 2.20 or newer
- An NVIDIA driver and NVIDIA Container Toolkit for the GPU worker

Running from source also requires Python 3.12, [uv](https://docs.astral.sh/uv/), and FFmpeg.

## Quick start

Copy the environment template and set a URL-safe PostgreSQL password:

```shell
cp .env.example .env
```

Set the GPU selector and its stable UUID in `.env`:

```shell
nvidia-smi --id=0 --query-gpu=uuid --format=csv,noheader
```

Use the returned value for `TEXTIFY_GPU_IDENTITY`.
Change `TEXTIFY_GPU_DEVICE_ID` as well if the worker should use a different GPU.

Start the API and GPU worker:

```shell
docker compose --profile gpu up --build -d
```

The API is available at `http://127.0.0.1:8182`.
Open `http://127.0.0.1:8182/docs` for the interactive API documentation, or check readiness with:

```shell
curl --fail http://127.0.0.1:8182/health
```

Job status and cancellation URLs returned by the API are bearer capabilities.
Keep them private.

## Run from source

Install dependencies and start PostgreSQL and Redis:

```shell
uv sync --frozen
docker compose up -d --wait postgres broker
uv run alembic upgrade head
```

Start each process in its own terminal:

```shell
uv run textify-reconciler
uv run textify
uv run textify-worker
```

Source processes read configuration from `.env`.
The worker requires CUDA, FFmpeg, and a valid `TEXTIFY_GPU_IDENTITY`.

## Configuration

See [`.env.example`](.env.example) for available settings, including ports, queue limits, model selection, storage paths, and transcription timeouts.
Keep `.env` private because it may contain database or Hugging Face credentials.

The default Compose topology starts PostgreSQL, Redis, migrations, the reconciler, and the API.
Add the `gpu` profile to process transcription jobs.

## Development

```shell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
```

## Shutdown

Stop the worker before the remaining services:

```shell
docker compose stop worker
docker compose down
```

`docker compose down` preserves named volumes.
Adding `--volumes` deletes all local persisted data.
