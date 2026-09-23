# Textify

Textify is an asynchronous HTTP API that creates durable Transcription Jobs for eligible public YouTube, Facebook, Instagram, TikTok, and X videos.
It uses an eligible YouTube caption track when available and otherwise uses Faster-Whisper on an NVIDIA CUDA GPU.

## Deployment boundary

One Textify process owns one SQLite database and one loaded GPU model.
Run one process or one container per GPU.
Do not run multiple Textify processes or containers against the same SQLite database path or durable volume.
SQLite provides the durable queue only within that single-instance boundary.

Textify requires Python 3.12 and [uv](https://docs.astral.sh/uv/) for source deployments.
Container deployments require Docker, NVIDIA Container Toolkit, and a GPU exposed to the container.

## Persistent state and migrations

`TEXTIFY_DATABASE_PATH` is required when migrations or application lifespan run.
Set it to a durable SQLite file that survives process replacement.
Importing `textify.main.app` and requesting `/openapi.json` do not read this setting.

Run the source migration before starting the API.

```shell
uv sync --frozen
cp .env.example .env
uv run alembic upgrade head
uv run textify
```

The application verifies the expected Alembic revision during startup.
It never creates tables or migrates the database automatically.
To reverse the most recent migration, stop the application and run:

```shell
uv run alembic downgrade -1
```

The first startup downloads the configured Whisper model when it is absent from the configured cache.
The default source listener is `http://127.0.0.1:8182`.

## Run with Docker

The image migrates the durable SQLite database before it starts Textify.
Mount one persistent volume at `/var/lib/textify`.
That directory contains the default database at `/var/lib/textify/textify.sqlite3` and the default temporary media root at `/var/lib/textify/media`.

```shell
docker build -t textify .

docker volume create textify-data
docker volume create textify-model-cache

docker run --rm \
  --gpus "device=0" \
  -p 8182:8182 \
  -v textify-data:/var/lib/textify \
  -v textify-model-cache:/var/cache/textify/huggingface \
  textify
```

The container runs as UID and GID `10001`.
For a bind mount, create the directory and grant that identity ownership before starting the container.

```shell
mkdir -p ./textify-data
sudo chown 10001:10001 ./textify-data

docker run --rm \
  --gpus "device=0" \
  -p 8182:8182 \
  -v "$(pwd)/textify-data:/var/lib/textify" \
  -v textify-model-cache:/var/cache/textify/huggingface \
  textify
```

The default container command is `alembic upgrade head && exec textify`.
Run a reversible downgrade by overriding that command while no application container owns the database.

```shell
docker run --rm \
  -v textify-data:/var/lib/textify \
  textify \
  alembic downgrade -1
```

Choose another GPU device in `--gpus "device=N"` for another isolated deployment.
Do not share the same `/var/lib/textify` volume between those deployments.

## Configuration

Copy `.env.example` to a private `.env` file for source deployment.
Keep `.env` private, especially `TEXTIFY_HF_TOKEN` when a model requires Hugging Face authentication.

| Setting | Default | Purpose |
| --- | --- | --- |
| `TEXTIFY_DATABASE_PATH` | Required | Durable SQLite database file. The container sets `/var/lib/textify/textify.sqlite3`. |
| `TEXTIFY_JOB_WORKER_COUNT` | `4` | In-process Transcription Job consumers. |
| `TEXTIFY_MAX_OUTSTANDING_JOBS` | `8` | Maximum queued and processing jobs. |
| `TEXTIFY_JOB_QUEUE_TIMEOUT_SECONDS` | `20` | Maximum time a queued job may wait before it fails with `queue_timeout`. |
| `TEXTIFY_JOB_RETENTION_SECONDS` | `86400` | Terminal job retention in seconds. |
| `TEXTIFY_TRANSCRIPTION_CONCURRENCY` | `2` | Maximum simultaneous Faster-Whisper operations. |
| `TEXTIFY_MAX_MEDIA_BYTES` | `536870912` | Maximum bytes for one downloaded media allocation. |
| `TEXTIFY_TEMPORARY_MEDIA_ROOT` | `/tmp/textify` | Writable root for temporary media outside the container default. |
| `TEXTIFY_INITIAL_PROMPT` | Empty | Optional Faster-Whisper initial prompt. |
| `TEXTIFY_HF_TOKEN` | Unset | Optional Hugging Face credential used only while loading the model. |

The temporary-media root must have at least `(job worker count + transcription concurrency + 1) * max media bytes` free space before startup.
The defaults reserve `(4 + 2 + 1) * 512 MiB = 3.5 GiB`.

See `.env.example` for all listener, model, decoding, timeout, and inference settings.

## API contract

Check readiness before submitting work.

```shell
curl http://127.0.0.1:8182/health
```

`GET /health` returns `{"status":"ok"}` only after migration verification, recovery, cleanup, model loading, and consumer startup complete.
It returns `503` if startup is incomplete or if a runtime durable-store failure makes the instance unready.
Every response includes `X-Request-ID` for request correlation.

Submit a Transcription Job with an eligible public video URL.

```shell
curl -i -X POST http://127.0.0.1:8182/api/transcription-jobs \
  -H 'Content-Type: application/json' \
  --data '{"url":"https://www.youtube.com/watch?v=YOUR_VIDEO_ID"}'
```

A committed submission returns `202 Accepted` with a relative `Location`, relative `links.self`, and relative `links.cancel` values.
Those links are same-origin bearer capabilities.
Possession authorizes status inspection and cancellation, so do not log, disclose, or cache them.
Textify returns `Cache-Control: no-store` for every Transcription Job response, including errors.

Poll `links.self` after the `Retry-After: 2` interval while the status is `queued` or `processing`.
The terminal status is `finished` with outcome `succeeded`, `failed`, or `cancelled`.
A successful job includes its projected transcript in `result`.
A failed job includes a stable safe error reason in `error`.
A cancelled job contains neither `result` nor `error`.

Submit an empty-body `PUT` request to `links.cancel`.
A queued or already cancelled job returns `200` with the cancelled representation.
A processing job returns `202`, sets `cancellation_requested: true`, and continues to own resources until cleanup completes.
Succeeded and failed jobs reject cancellation with `409 job_already_finished`.

Textify rejects immediate malformed requests and unsupported URLs before a job exists.
It returns provider, download, timeout, queue-timeout, worker-interrupted, and unexpected processing problems as terminal failed-job reasons in a `200` status representation.

## Recovery and retention

Textify does not retry or requeue Transcription Jobs automatically.
On restart, unexpired queued jobs remain eligible until their original queue deadline.
Interrupted processing jobs finish as `worker_interrupted`, and interrupted accepted cancellations finish as `cancelled`.

Terminal jobs remain addressable for `86400` seconds by default.
Startup and hourly retention cleanup remove expired terminal jobs and their related persisted result data.
After expiration, status and cancellation return the same `404 job_not_found` response as an unknown capability.
SQLite `secure_delete=ON` and application-level deletion do not guarantee physical erasure from storage media, snapshots, backups, or filesystems.

## Live API contract check

The live test is opt-in and requires a running GPU-backed Textify instance plus one or more public source URLs.

```shell
TEXTIFY_LIVE_BASE_URL=http://127.0.0.1:8182 \
TEXTIFY_LIVE_YOUTUBE_URL='https://www.youtube.com/watch?v=YOUR_VIDEO_ID' \
uv run pytest tests/test_live_api.py -m live
```

Set any combination of `TEXTIFY_LIVE_YOUTUBE_URL`, `TEXTIFY_LIVE_FACEBOOK_URL`, `TEXTIFY_LIVE_INSTAGRAM_URL`, `TEXTIFY_LIVE_TIKTOK_URL`, and `TEXTIFY_LIVE_X_URL`.

Interactive OpenAPI documentation is available at `http://127.0.0.1:8182/docs`.
