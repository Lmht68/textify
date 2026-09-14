# Textify

Textify is a synchronous HTTP API for transcripts from eligible public YouTube, Facebook, Instagram, TikTok, and X videos.
It runs one preloaded Faster-Whisper model per process and is intentionally CUDA-only.

## Python package

Textify requires Python 3.12 or later and [uv](https://docs.astral.sh/uv/).
Install Python 3.12 if needed, then create the locked development environment from the repository root.

```shell
uv python install 3.12
uv sync --frozen
```

`pyproject.toml` and `uv.lock` are the production installation source.
The deployment image installs the production package with `uv sync --frozen --no-dev --no-editable`.

`requirements.txt` is the reproducible combined runtime and development export for consumers that do not use uv.
Install it from the repository root with Python 3.12.

```shell
python3.12 -m pip install -r requirements.txt
```

Regenerate the export only after resolving the lock.

```shell
uv export --frozen --all-groups --no-hashes --output-file requirements.txt
```

Build distributable artifacts with:

```shell
uv build
```

## CUDA container

The `Dockerfile` pins CUDA 12.9.2 with the cuDNN 9 runtime and uv 0.12.1.
Building the image requires Docker and network access to the pinned image registries.
Running transcription requires an NVIDIA GPU, a compatible host driver, and NVIDIA Container Toolkit.
The image starts one Uvicorn worker, rejects CPU device configuration, and fails startup if CUDA is not visible.

Build the image from the repository root.

```shell
docker build -t textify:issue-08 .
```

Create persistent named volumes before model prepopulation and service startup.

```shell
docker volume create textify-model-cache
docker volume create textify-media
```

Prepopulate the pinned model revision without starting Uvicorn.

```shell
docker run --rm \
  --entrypoint python \
  -v textify-model-cache:/var/cache/textify/huggingface \
  textify:issue-08 \
  -c 'from faster_whisper.utils import download_model; download_model("large-v3-turbo", revision="0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf")'
```

Run exactly one container for each host GPU.
Set `HOST_GPU_INDEX` to the one host GPU exposed to this container.
Inside the container, that selected device is index `0`, so leave `TEXTIFY_WHISPER_DEVICE_INDEX=0` unchanged.

```shell
export HOST_GPU_INDEX=0

docker run --rm \
  --name "textify-gpu-${HOST_GPU_INDEX}" \
  --gpus "device=${HOST_GPU_INDEX}" \
  -p 8182:8182 \
  -v textify-model-cache:/var/cache/textify/huggingface \
  -v textify-media:/var/lib/textify/media \
  textify:issue-08
```

The container defaults `TEXTIFY_HOST` to `0.0.0.0`, places temporary media in `/var/lib/textify/media`, and uses the persistent Hugging Face cache at `/var/cache/textify/huggingface`.
Both paths are declared volumes and owned by the fixed unprivileged UID/GID `10001`.

### Local smoke request

After the container reports startup completion, verify readiness.

```shell
curl -i http://127.0.0.1:8182/health
```

Supply an operator-owned eligible public video URL, construct the JSON body with Python, and make a synchronous request.

```shell
export SOURCE_URL='OPERATOR_OWNED_PUBLIC_VIDEO_URL'
REQUEST_BODY="$(python3.12 -c 'import json, os; print(json.dumps({"url": os.environ["SOURCE_URL"]}))')"
curl -i -X POST \
  -H 'Content-Type: application/json' \
  --data "$REQUEST_BODY" \
  http://127.0.0.1:8182/api/transcripts
```

The health response must be HTTP 200 with `{"status":"ok"}`.
The transcript response must contain a server-generated `X-Request-ID` header and either a 200 body matching the response schema below or a documented non-200 body shaped as `{"error":{"code":"<stable code>","message":"<nonempty safe message>"}}`.

## Configuration

Copy `.env.example` to the untracked `.env` file for local configuration.
Never commit `TEXTIFY_HF_TOKEN` or other credentials.
The `TEXTIFY_LIVE_*` values in the example file control opt-in integration checks and are not application settings.

| Setting | Shipped default | Validation | Operational meaning |
| --- | --- | --- | --- |
| `TEXTIFY_ENVIRONMENT` | `local` | `local`, `staging`, or `production` | Deployment environment label. |
| `TEXTIFY_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL` | Textify logger threshold. |
| `TEXTIFY_HOST` | `127.0.0.1` | Nonempty string | Uvicorn listen host. |
| `TEXTIFY_PORT` | `8182` | Integer from 1 through 65535 | Uvicorn listen port. |
| `TEXTIFY_MAX_DURATION_SECONDS` | `1800` | Positive integer | Maximum accepted source duration in seconds. |
| `TEXTIFY_TEMPORARY_MEDIA_ROOT` | `/tmp/textify` | Filesystem path | Dedicated request-scoped media root. |
| `TEXTIFY_WHISPER_MODEL` | `large-v3-turbo` | Nonempty string | Faster-Whisper model identifier. |
| `TEXTIFY_WHISPER_REVISION` | `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf` | Nonempty string | Immutable model revision. |
| `TEXTIFY_WHISPER_DEVICE` | `cuda` | Fixed value `cuda` | CUDA-only Faster-Whisper device. |
| `TEXTIFY_WHISPER_DEVICE_INDEX` | `0` | Integer greater than or equal to 0 | Visible CUDA device index. |
| `TEXTIFY_WHISPER_COMPUTE_TYPE` | `float16` | Nonempty string | Faster-Whisper compute type. |
| `TEXTIFY_BEAM_SIZE` | `1` | Positive integer | Faster-Whisper beam size. |
| `TEXTIFY_VAD_FILTER` | `true` | Boolean | Enable voice activity detection. |
| `TEXTIFY_TEMPERATURE` | `0.0` | Finite number from 0.0 through 1.0 | Faster-Whisper decoding temperature. |
| `TEXTIFY_CONDITION_ON_PREVIOUS_TEXT` | `true` | Boolean | Condition decoding on prior transcript text. |
| `TEXTIFY_TRANSCRIPTION_CONCURRENCY` | `1` | Positive integer | Native inference concurrency. |
| `TEXTIFY_MAX_PENDING_TRANSCRIPTIONS` | `2` | Integer greater than or equal to 0 | Additional queued transcription admissions. |
| `TEXTIFY_MAX_MEDIA_BYTES` | `536870912` | Positive integer | Inclusive byte limit for selected and downloaded media. |
| `TEXTIFY_METADATA_TIMEOUT_SECONDS` | `30.0` | Positive finite number | Metadata-stage deadline. |
| `TEXTIFY_AUDIO_DOWNLOAD_TIMEOUT_SECONDS` | `300.0` | Positive finite number | Audio-download-stage deadline. |
| `TEXTIFY_TRANSCRIPTION_QUEUE_TIMEOUT_SECONDS` | `300.0` | Positive finite number | Native-inference queue deadline. |
| `TEXTIFY_TRANSCRIPTION_TIMEOUT_SECONDS` | `1800.0` | Positive finite number | Native-inference deadline after admission. |
| `TEXTIFY_HF_TOKEN` | Unset | Optional secret | Hugging Face authentication when the model requires it. |

The temporary media root must be a writable dedicated volume.
Startup creates the root, verifies that it can create a temporary file there, and checks available capacity before model loading.
The required free space is `(max pending transcriptions + active concurrency + 1) * max media bytes`.
With default settings, `(2 pending + 1 active + 1 retained) * 536870912 = 2147483648` bytes.
Equality is accepted.
Changing capacity or media-byte settings changes the required quota by the same formula.

## HTTP API

Every HTTP response carries a server-generated `X-Request-ID` header.
The OpenAPI document is available at `/openapi.json`, with interactive documentation at `/docs` and `/redoc`.

### `GET /health`

`GET /health` is a readiness endpoint.
It returns HTTP 200 and `{"status":"ok"}` only after lifespan startup has created the adapter bundle and loaded the model.
A startup failure, including unavailable CUDA or insufficient temporary-media capacity, prevents the process from becoming ready.

```http
GET /health HTTP/1.1
Host: 127.0.0.1:8182
```

```http
HTTP/1.1 200 OK
X-Request-ID: 00000000-0000-4000-8000-000000000000
Content-Type: application/json

{"status":"ok"}
```

### `POST /api/transcripts`

`POST /api/transcripts` processes one source synchronously.
It accepts one strict JSON object with only the `url` string field, limited to 2048 characters.

```http
POST /api/transcripts HTTP/1.1
Host: 127.0.0.1:8182
Content-Type: application/json

{"url":"https://www.youtube.com/watch?v=AbCdEf12345"}
```

A successful response contains canonical source metadata and a normalized transcript.
The response transcript `text` is the exact space-joined text of the ordered `segments`.

```json
{
  "source": {
    "platform": "youtube",
    "video_id": "AbCdEf12345",
    "url": "https://www.youtube.com/watch?v=AbCdEf12345",
    "title": "Synthetic example video",
    "description": "",
    "channel": "Synthetic channel",
    "duration_seconds": 42
  },
  "transcript": {
    "method": "youtube_captions",
    "language": "en",
    "text": "First synthetic segment. Second synthetic segment.",
    "segments": [
      {"start": 0.0, "end": 1.0, "text": "First synthetic segment."},
      {"start": 1.0, "end": 2.0, "text": "Second synthetic segment."}
    ]
  }
}
```

Failures use the safe error envelope.
Messages are safe for display but are not byte-stable contracts.

```json
{
  "error": {
    "code": "invalid_url",
    "message": "The submitted URL is invalid."
  }
}
```

| HTTP status | Stable codes |
| --- | --- |
| 400 | `invalid_url`, `unsupported_platform` |
| 422 | `invalid_request`, `unsupported_content`, `video_too_long`, `invalid_media_duration`, `unsupported_media`, `no_usable_transcript` |
| 500 | `internal_error` |
| 502 | `metadata_retrieval_failed`, `audio_download_failed`, `transcription_failed` |
| 503 | `transcription_capacity_exceeded` |
| 504 | `metadata_timeout`, `audio_download_timeout`, `transcription_timeout` |

### Accepted public-video URLs

Only HTTPS URLs without credentials, ports, control characters, malformed percent escapes, or encoded paths are accepted.
Fragments are discarded before provider access.
The input must be 1 through 2048 characters.
Only one lowercase `v` identity query parameter is permitted where a form requires it.
Duplicate `v` parameters, uppercase `V`, and identity queries on path-based YouTube forms are rejected.
Canonical responses remove fragments and nonidentity queries, retain a Facebook `v` identity query when necessary, and use `x.com` for X responses.

| Platform | Exact hosts | Accepted video-path forms |
| --- | --- | --- |
| YouTube | `youtube.com`, `www.youtube.com`, `m.youtube.com`, `music.youtube.com`, `youtu.be` | Full hosts: `/watch?v=<11-char-id>`, `/shorts/<11-char-id>`, `/embed/<11-char-id>`, or `/live/<11-char-id>`. Short host: `/<11-char-id>`. |
| Instagram | `instagram.com`, `www.instagram.com` | `/p/<shortcode>`, `/tv/<shortcode>`, `/reel/<shortcode>`, or `/reels/<shortcode>`, with one trailing slash normalized. |
| Facebook | `facebook.com`, `www.facebook.com`, `m.facebook.com`, `fb.watch` | Full hosts: `/watch?v=<id>`, `/video.php?v=<id>`, `/video/video.php?v=<id>`, `/reel/<id>`, `/share/v/<token>`, `/share/<segment>/<segment>/<token>`, `/<owner>/videos/<id>`, `/<owner>/<series>/videos/<id>`, or `/<owner>/posts/<id>`. Short host: `/<token>`. |
| TikTok | `www.tiktok.com`, `vm.tiktok.com`, `vt.tiktok.com` | Full host: `/@<user>/video/<numeric-id>`, `/embed/<numeric-id>`, or `/t/<token>`. Short hosts: `/<token>`. |
| X / Twitter | `x.com`, `www.x.com`, `m.x.com`, `mobile.x.com`, `twitter.com`, `www.twitter.com`, `m.twitter.com`, `mobile.twitter.com`, `t.co` | Full hosts: `/<user>/status/<numeric-id>`, `/i/web/status/<numeric-id>`, or `/statuses/<numeric-id>`, each optionally followed by `/video/<positive-index>`. Short host: `/<token>`. |

Textify rejects playlists, collections, ambiguous paths, live or upcoming content, non-video media, private or login-gated content, provider-blocked content, unavailable content, oversized media, invalid durations, and media exceeding the configured duration limit.
YouTube first attempts an eligible original caption track and falls back to Faster-Whisper when no usable caption transcript is available.
Facebook, Instagram, TikTok, and X use Faster-Whisper.

## Provider and security limits

Each yt-dlp retrieval operation makes at most ten immediate attempts within its stage deadline.
Textify ignores user yt-dlp configuration, disables cookie files, browser cookies, netrc, remote components, external downloaders, file URLs, generic extraction, and insecure transport.
It allows only the Facebook, Instagram, TikTok, X/Twitter, and YouTube extractors and follows provider redirects.

Textify validates accepted URL forms and validates provider-reported Facebook, Instagram, TikTok, and X webpage URLs against the same platform.
It does not inspect every redirect hop, resolve or deny destination IP ranges, or enforce deployment egress rules.
Operators must apply network egress controls appropriate to their deployment.

Normal log output is limited to the request ID, normalized platform, known source ID, stage, acquisition method, source duration, elapsed time, and safe error code.
Submitted URLs, redirected URLs, media URLs, transcript text, caption text, model text, commands, headers, credentials, cookies, and local paths are excluded from Textify logs.

## Opt-in live API checks

The live check performs network requests only when `TEXTIFY_LIVE_BASE_URL` and at least one `TEXTIFY_LIVE_<PLATFORM>_URL` are configured.
Use only operator-owned public, non-live videos that meet the configured duration and media limits.
No public URLs or credentials belong in the repository.

Start the locally built image, then set the running API URL and one or more platform inputs.

```shell
export TEXTIFY_LIVE_BASE_URL=http://127.0.0.1:8182
export TEXTIFY_LIVE_YOUTUBE_URL='OPERATOR_OWNED_PUBLIC_VIDEO_URL'
uv run pytest -m live tests/test_live_api.py
```

The check validates only stable API invariants because public provider metadata and transcript content can change.
