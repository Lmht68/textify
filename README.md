# Textify

## Temporary media volume

`TEXTIFY_TEMPORARY_MEDIA_ROOT` must be a writable dedicated volume.
Startup requires at least `(max pending transcriptions + active concurrency + 1) * max media bytes` free bytes.
With default settings, `(2 + 1 + 1) * 536_870_912 = 2_147_483_648` bytes (2 GiB).
This quota assumes the documented one-worker process.