# Textify

Textify is a synchronous HTTP API that returns normalized transcripts for eligible public YouTube, Facebook, Instagram, TikTok, and X videos.
It uses an eligible YouTube caption track when available and Faster-Whisper for audio transcription when needed.
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

Request a transcript by replacing the example URL with an eligible public video URL.

```shell
curl -X POST http://127.0.0.1:8182/api/transcripts \
  -H 'Content-Type: application/json' \
  --data '{"url":"https://www.youtube.com/watch?v=YOUR_VIDEO_ID"}'
```

A successful response contains source metadata and a transcript with plain text and timed segments.
Interactive API documentation is available at `http://127.0.0.1:8182/docs`.

Textify accepts supported public video URLs only.
It rejects unavailable, private, live, non-video, oversized, and over-duration sources.
Requests run synchronously, so keep client timeouts appropriate for the source duration and transcription workload.

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
| `TEXTIFY_TRANSCRIPTION_CONCURRENCY`, `TEXTIFY_MAX_PENDING_TRANSCRIPTIONS` | Active and queued transcription capacity. |
| `TEXTIFY_MAX_MEDIA_BYTES` | Maximum downloaded media size. |
| `TEXTIFY_HF_TOKEN` | Optional Hugging Face credential for restricted models. |

The temporary media directory needs enough free space for concurrent work.
See `.env.example` for the remaining inference and timeout settings.
