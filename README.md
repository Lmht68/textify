# Textify

Textify is an asynchronous HTTP API for normalized transcripts from eligible public YouTube, Facebook, Instagram, TikTok, and X videos.
It accepts durable Transcription Jobs, uses an eligible YouTube caption track when available, and otherwise uses Faster-Whisper for audio transcription.
Inference requires an NVIDIA CUDA GPU.

## What you need

- Python 3.12 and [uv](https://docs.astral.sh/uv/) to run from source.
- An NVIDIA GPU with CUDA available to the process.
- Docker and NVIDIA Container Toolkit when running the container.

## Run from source

Create the locked environment, copy the local configuration template, and start the API.

```shell
uv sync --frozen
cp .env.example .env
uv run textify
```

The first startup downloads the configured Whisper model if it is not already cached.
The API listens on `http://127.0.0.1:8182` by default.

## Run with Docker

Build the image, then start it with one GPU and persistent caches for the model and temporary media.

```shell
docker build -t textify .

docker run --rm \
  --gpus "device=0" \
  -p 8182:8182 \
  -v textify-model-cache:/var/cache/textify/huggingface \
  -v textify-media:/var/lib/textify/media \
  textify
```

Use a different GPU index for `device` when needed.
Run one container per GPU when serving multiple GPUs.

## Use the API

Check that the service is ready.

```shell
curl http://127.0.0.1:8182/health
```

Submit a Transcription Job by replacing the example URL with an eligible public video URL.

```shell
curl -i -X POST http://127.0.0.1:8182/api/transcription-jobs \
  -H 'Content-Type: application/json' \
  --data '{"url":"https://www.youtube.com/watch?v=YOUR_VIDEO_ID"}'
```

The `202 Accepted` response contains a `Location` header and a capability link in `links.self`.
Poll that location after the response's `Retry-After` interval until it returns `status: "finished"` and `outcome: "succeeded"`.
The terminal response nests source metadata and a transcript with plain text and timed segments under `result`.
Interactive API documentation is available at `http://127.0.0.1:8182/docs`.

Textify accepts supported public video URLs only.
It rejects unavailable, private, live, non-video, oversized, and over-duration sources before execution.

## Configuration

Copy `.env.example` to `.env` and change only the values required for your deployment.
Keep `.env` private, especially `TEXTIFY_HF_TOKEN` when model access requires it.

| Setting | Purpose |
| --- | --- |
| `TEXTIFY_HOST`, `TEXTIFY_PORT` | API listen address and port. |
| `TEXTIFY_MAX_DURATION_SECONDS` | Maximum accepted video duration. |
| `TEXTIFY_TEMPORARY_MEDIA_ROOT` | Writable directory for request media. |
| `TEXTIFY_WHISPER_MODEL`, `TEXTIFY_WHISPER_REVISION` | Whisper model and pinned revision to load. |
| `TEXTIFY_WHISPER_DEVICE_INDEX` | CUDA device visible to the process. |
| `TEXTIFY_TRANSCRIPTION_CONCURRENCY` | Maximum concurrent native inference operations. |
| `TEXTIFY_JOB_WORKER_COUNT` | Number of application-owned durable job consumers. |
| `TEXTIFY_MAX_OUTSTANDING_JOBS`, `TEXTIFY_JOB_QUEUE_TIMEOUT_SECONDS` | Durable admission capacity and database queue-lock deadline. |
| `TEXTIFY_MAX_MEDIA_BYTES` | Maximum downloaded media size. |
| `TEXTIFY_HF_TOKEN` | Optional Hugging Face credential for restricted models. |

The temporary media directory needs enough free space for configured job workers, native inference, and one cleanup reserve.
See `.env.example` for the remaining inference and timeout settings.
