"""Adapt caption, download, and Faster-Whisper providers for Textify."""

import asyncio
import logging
import tempfile
import threading
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from decimal import Decimal
from numbers import Real
from pathlib import Path
from typing import Final, Protocol, cast
from xml.etree import ElementTree

import ctranslate2
import yt_dlp
from faster_whisper import WhisperModel
from requests.exceptions import RequestException, Timeout
from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    YouTubeTranscriptApi,
)
from yt_dlp.utils import DownloadCancelled, DownloadError, YoutubeDLError

from textify.transcription.config import TranscriptionConfig
from textify.transcription.exceptions import (
    AudioDownloadFailedError,
    AudioDownloadTimeoutError,
    TranscriptionCapacityExceededError,
    TranscriptionFailedError,
    TranscriptionTimeoutError,
)
from textify.transcription.inspection import (
    PreparedAudio,
    _cleanup_request_directory,
    _is_timeout_exception,
)
from textify.transcription.types import (
    RawSegment,
    Transcript,
    TranscriptMethod,
    normalize_transcript,
)
from textify.transcription.util import (
    build_yt_dlp_progress_hook,
    extract_info_with_retries,
)

logger = logging.getLogger(__name__)

_YTDLP_OPTIONS: Final[dict[str, object]] = {
    "allowed_extractors": [r"^(?:tiktok|vm\.tiktok)$"],
    "format": "bestaudio/best",
    "ignoreconfig": True,
    "no_warnings": True,
    "noplaylist": True,
    "quiet": True,
}
_AUDIO_OUTPUT_TEMPLATE: Final[str] = "audio.%(ext)s"
_WHISPER_TEMP_PREFIX: Final[str] = "textify-whisper-"


class _CaptionProviderFailure(Exception):
    """Represent an ordinary failure at the caption provider boundary."""


class _CaptionProviderTimeout(Exception):
    """Represent a timeout at the caption provider boundary."""


class _AudioDownloadProviderFailure(Exception):
    """Represent an ordinary failure at the audio download boundary."""


class _AudioDownloadProviderTimeout(Exception):
    """Represent a timeout at the audio download boundary."""


class _TranscriptionProviderFailure(Exception):
    """Represent an ordinary failure at the native inference boundary."""


class _TranscriptionProviderTimeout(Exception):
    """Represent a timeout at the native inference boundary."""


def _log_acquisition_error(event: str, reason: str) -> None:
    """Log a safe, structured reason for an acquisition failure."""
    logger.debug(
        event,
        extra={
            "stage": "transcription",
            "reason": reason,
        },
    )


_CAPTION_EXTERNAL_FAILURES: Final[tuple[type[BaseException], ...]] = (
    CouldNotRetrieveTranscript,
    ElementTree.ParseError,
    RequestException,
    AttributeError,
    IndexError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)
_TRANSCRIPTION_EXTERNAL_FAILURES: Final[tuple[type[BaseException], ...]] = (
    AttributeError,
    EOFError,
    IndexError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


class CaptionTrack(Protocol):
    """Expose timed caption metadata for one source video."""

    @property
    def language_code(self) -> str:
        """Return the track's original BCP 47 language code."""
        ...

    @property
    def is_generated(self) -> bool:
        """Return whether the provider generated the track automatically."""
        ...

    def fetch_segments(self) -> Sequence[RawSegment]:
        """Fetch timed provider segments in original provider order.

        Returns:
            ``(start, end, text)`` values from the caption provider.

        Raises:
            _CaptionProviderFailure: If the provider payload is unusable.
            _CaptionProviderTimeout: If the provider request times out.
        """
        ...


class CaptionProvider(Protocol):
    """List Caption Tracks for a validated Source."""

    def list_tracks(self, video_id: str) -> Sequence[CaptionTrack]:
        """Return tracks in the provider's insertion order.

        Args:
            video_id: Stable external video identity.

        Returns:
            Available Caption Tracks.

        Raises:
            _CaptionProviderFailure: If track listing fails.
            _CaptionProviderTimeout: If track listing times out.
        """
        ...


class AudioDownloader(Protocol):
    """Download one Source's native best-audio representation."""

    def download(
        self,
        source_url: str,
        destination: Path,
        *,
        deadline: float,
        cancellation_event: threading.Event,
    ) -> Path:
        """Download audio into the provided request directory.

        Args:
            source_url: Validated provider URL to download.
            destination: Existing private request directory.
            deadline: Monotonic absolute deadline for audio download.
            cancellation_event: Signal set when request work must stop.

        Returns:
            Completed audio file path owned by ``destination``.
        """
        ...


class WhisperTranscriber(Protocol):
    """Transcribe one local audio file with a preloaded model."""

    def transcribe(self, audio_path: Path) -> Transcript:
        """Transcribe the provided local audio file.

        Args:
            audio_path: Validated completed audio file.

        Returns:
            Normalized timed Transcript.
        """
        ...


class _LibrarySnippet(Protocol):
    """Expose one timed youtube-transcript-api snippet."""

    start: object
    duration: object
    text: object


class _LibraryFetchedTranscript(Protocol):
    """Expose the iterable snippets returned by the provider."""

    def __iter__(self) -> Iterator[_LibrarySnippet]:
        """Return snippets in provider segment order."""
        ...


class _LibraryTranscript(Protocol):
    """Expose the provider track fields used by the adapter."""

    language_code: str
    is_generated: bool

    def fetch(self, preserve_formatting: bool = False) -> object:
        """Fetch provider timed-text data."""
        ...


class _YouTubeCaptionTrack:
    """Adapt one youtube-transcript-api track to CaptionTrack."""

    def __init__(self, transcript: _LibraryTranscript) -> None:
        try:
            language_code = transcript.language_code
            is_generated = transcript.is_generated
        except (AttributeError, TypeError, ValueError) as exc:
            _log_acquisition_error("caption provider error", "invalid_track_metadata")
            raise _CaptionProviderFailure from exc
        if not isinstance(language_code, str) or not isinstance(is_generated, bool):
            _log_acquisition_error("caption provider error", "invalid_track_metadata")
            raise _CaptionProviderFailure
        self._transcript = transcript
        self._language_code = language_code
        self._is_generated = is_generated

    @property
    def language_code(self) -> str:
        """Return the provider's original language code."""
        return self._language_code

    @property
    def is_generated(self) -> bool:
        """Return the provider's generated-track flag."""
        return self._is_generated

    def fetch_segments(self) -> Sequence[RawSegment]:
        """Fetch timed segments with provider formatting disabled.

        Returns:
            Provider ``(start, end, text)`` values in original order.

        Raises:
            _CaptionProviderFailure: If the provider payload is malformed.
            _CaptionProviderTimeout: If the provider request times out.
        """
        try:
            fetched = cast(
                _LibraryFetchedTranscript,
                self._transcript.fetch(preserve_formatting=False),
            )
            return tuple(
                (
                    snippet.start,
                    _caption_segment_end(snippet.start, snippet.duration),
                    snippet.text,
                )
                for snippet in fetched
            )
        except (Timeout, TimeoutError) as exc:
            _log_acquisition_error("caption provider timeout", "track_fetch_timeout")
            raise _CaptionProviderTimeout from exc
        except _CaptionProviderFailure:
            raise
        except _CAPTION_EXTERNAL_FAILURES as exc:
            _log_acquisition_error("caption provider error", "track_fetch_failed")
            raise _CaptionProviderFailure from exc


def _caption_segment_end(start: object, duration: object) -> float:
    """Derive a caption segment end time without accepting invalid booleans."""
    if (
        isinstance(start, bool)
        or isinstance(duration, bool)
        or not isinstance(start, (Real, Decimal))
        or not isinstance(duration, (Real, Decimal))
    ):
        raise _CaptionProviderFailure
    try:
        return float(start) + float(duration)
    except (OverflowError, TypeError, ValueError) as exc:
        raise _CaptionProviderFailure from exc


class YouTubeCaptionProvider:
    """Adapt youtube-transcript-api to the Textify CaptionProvider contract."""

    def list_tracks(self, video_id: str) -> Sequence[CaptionTrack]:
        """List Caption Tracks using one provider client instance.

        Args:
            video_id: Stable external video identity.

        Returns:
            Sequence[CaptionTrack]: Wrapped provider tracks in provider order.

        Raises:
            _CaptionProviderFailure: If the provider payload cannot be adapted.
            _CaptionProviderTimeout: If the listing request times out.
        """
        try:
            api = YouTubeTranscriptApi()
            transcript_list = cast(Iterable[_LibraryTranscript], api.list(video_id))
            wrapped_tracks: list[CaptionTrack] = []
            for track in transcript_list:
                try:
                    wrapped_tracks.append(_YouTubeCaptionTrack(track))
                except _CaptionProviderFailure:
                    logger.debug(
                        "caption track unavailable",
                        extra={
                            "stage": "transcription",
                            "reason": "invalid_track_metadata",
                        },
                    )
                    continue
            return tuple(wrapped_tracks)
        except (Timeout, TimeoutError) as exc:
            _log_acquisition_error("caption provider timeout", "track_listing_timeout")
            raise _CaptionProviderTimeout from exc
        except _CAPTION_EXTERNAL_FAILURES as exc:
            _log_acquisition_error("caption provider error", "track_listing_failed")
            raise _CaptionProviderFailure from exc


class _WhisperSegment(Protocol):
    """Expose one raw timed segment returned by faster-whisper."""

    start: object
    end: object
    text: object


class _WhisperInfo(Protocol):
    """Expose detected language returned by faster-whisper."""

    language: object


class _WhisperModel(Protocol):
    """Expose the faster-whisper method used by the adapter."""

    def transcribe(
        self,
        audio: str,
        *,
        beam_size: int,
        vad_filter: bool,
        temperature: float,
        condition_on_previous_text: bool,
        initial_prompt: str,
    ) -> tuple[Iterable[_WhisperSegment], _WhisperInfo]:
        """Return a lazy segment iterator and transcription metadata."""
        ...


class YtDlpAudioDownloader:
    """Download native best audio into a private request directory."""

    def download(
        self,
        source_url: str,
        destination: Path,
        *,
        deadline: float,
        cancellation_event: threading.Event,
    ) -> Path:
        """Download one Source's native best-audio representation.

        Args:
            source_url: Validated provider URL to download.
            destination: Existing private request directory.
            deadline: Monotonic absolute deadline for audio download.
            cancellation_event: Signal set when request work must stop.

        Returns:
            Validated completed audio file inside ``destination``.

        Raises:
            _AudioDownloadProviderFailure: If yt-dlp returns unusable output.
            _AudioDownloadProviderTimeout: If yt-dlp reports a typed timeout.
        """
        request_directory = destination.resolve()
        options = {
            **_YTDLP_OPTIONS,
            "outtmpl": str(request_directory / _AUDIO_OUTPUT_TEMPLATE),
            "progress_hooks": [
                build_yt_dlp_progress_hook(
                    deadline=deadline,
                    cancellation_event=cancellation_event,
                )
            ],
        }
        try:
            with yt_dlp.YoutubeDL(options) as youtube_dl:
                raw_info = extract_info_with_retries(
                    youtube_dl.extract_info,
                    source_url,
                    download=True,
                    deadline=deadline,
                    cancellation_event=cancellation_event,
                )
                if not isinstance(raw_info, Mapping) or "entries" in raw_info:
                    _log_acquisition_error(
                        "audio download provider error",
                        "invalid_download_result",
                    )
                    raise _AudioDownloadProviderFailure
                prepared_path = youtube_dl.prepare_filename(raw_info)
        except _AudioDownloadProviderFailure:
            raise
        except DownloadError as exc:
            if _is_timeout_exception(exc):
                _log_acquisition_error(
                    "audio download provider timeout",
                    "audio_download_timeout",
                )
                raise _AudioDownloadProviderTimeout from exc
            _log_acquisition_error(
                "audio download provider error",
                "download_failed",
            )
            raise _AudioDownloadProviderFailure from exc
        except YoutubeDLError as exc:
            _log_acquisition_error("audio download provider error", "download_failed")
            raise _AudioDownloadProviderFailure from exc
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            _log_acquisition_error(
                "audio download provider error",
                "invalid_download_result",
            )
            raise _AudioDownloadProviderFailure from exc

        if not isinstance(prepared_path, (str, Path)):
            _log_acquisition_error("audio download provider error", "invalid_path")
            raise _AudioDownloadProviderFailure
        completed_path = Path(prepared_path).resolve()
        if completed_path.parent != request_directory:
            _log_acquisition_error(
                "audio download provider error",
                "audio_path_outside_request_directory",
            )
            raise _AudioDownloadProviderFailure
        if not completed_path.is_file():
            _log_acquisition_error(
                "audio download provider error", "audio_file_missing"
            )
            raise _AudioDownloadProviderFailure
        return completed_path


class FasterWhisperTranscriber:
    """Adapt one preloaded faster-whisper model to WhisperTranscriber."""

    def __init__(self, model: object, settings: TranscriptionConfig) -> None:
        """Initialize the adapter around an already-loaded model.

        Args:
            model: Preloaded faster-whisper model instance.
            settings: Environment-backed inference tuning settings.
        """
        self._model = model
        self._settings = settings

    def transcribe(self, audio_path: Path) -> Transcript:
        """Transcribe one local audio file into normalized timed segments.

        Args:
            audio_path: Validated completed audio file.

        Returns:
            Normalized Transcript, including successful silent media.

        Raises:
            _TranscriptionProviderFailure: If model output or inference is unusable.
        """
        try:
            settings = self._settings
            segments, info = cast(_WhisperModel, self._model).transcribe(
                str(audio_path),
                beam_size=settings.beam_size,
                vad_filter=settings.vad_filter,
                temperature=settings.temperature,
                condition_on_previous_text=settings.condition_on_previous_text,
                initial_prompt=settings.initial_prompt,
            )
            transcript = normalize_transcript(
                TranscriptMethod.FASTER_WHISPER,
                info.language,
                ((segment.start, segment.end, segment.text) for segment in segments),
            )
        except _TRANSCRIPTION_EXTERNAL_FAILURES as exc:
            _log_acquisition_error(
                "native transcription provider error",
                "transcription_failed",
            )
            raise _TranscriptionProviderFailure from exc

        logger.debug(
            "transcript acquired",
            extra={
                "stage": "transcription",
                "method": transcript.method.value,
                "segment_count": len(transcript.segments),
            },
        )
        return transcript


def load_whisper_transcriber(
    settings: TranscriptionConfig,
) -> FasterWhisperTranscriber:
    """Load the configured Faster-Whisper model and wrap it in the provider adapter.

    Args:
        settings: Environment-backed inference tuning and token settings.

    Returns:
        Adapter around one process-lifetime Faster-Whisper model.

    Raises:
        RuntimeError: If CUDA is unavailable.
        Exception: If Faster-Whisper cannot load its configured model.
    """
    if ctranslate2.get_cuda_device_count() == 0:
        _log_acquisition_error(
            "native transcription provider error", "cuda_unavailable"
        )
        raise RuntimeError("CUDA is unavailable for Faster-Whisper.")

    model_options: dict[str, str | int] = {
        "compute_type": settings.whisper_compute_type,
        "device": settings.whisper_device,
        "device_index": settings.whisper_device_index,
        "revision": settings.whisper_revision,
    }
    if settings.hf_token is not None:
        model_options["use_auth_token"] = settings.hf_token.get_secret_value()
    model = WhisperModel(settings.whisper_model, **model_options)
    return FasterWhisperTranscriber(model, settings)


async def _finish_cancelled_worker(
    worker: asyncio.Task[Transcript],
) -> None:
    """Wait for native inference before releasing its permit after cancellation."""
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            continue
        except (
            _TranscriptionProviderFailure,
            _TranscriptionProviderTimeout,
        ):
            return
    try:
        worker.result()
    except (_TranscriptionProviderFailure, _TranscriptionProviderTimeout):
        return
    except (
        AttributeError,
        EOFError,
        IndexError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        logger.error("native inference failed after request cancellation")


async def _finish_cancelled_download(worker: asyncio.Task[Path]) -> None:
    """Wait for a download before deleting its request directory after cancellation."""
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            continue
        except (
            _AudioDownloadProviderFailure,
            _AudioDownloadProviderTimeout,
            DownloadCancelled,
        ):
            return
    try:
        worker.result()
    except (
        _AudioDownloadProviderFailure,
        _AudioDownloadProviderTimeout,
        DownloadCancelled,
    ):
        return
    except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        logger.error("audio download failed after request cancellation")


def _validate_audio_path(audio_path: Path, request_directory: Path) -> Path:
    """Validate that downloaded audio remains owned by its request directory."""
    if not isinstance(audio_path, Path):
        _log_acquisition_error(
            "audio download provider error", "invalid_audio_path_type"
        )
        raise _AudioDownloadProviderFailure
    resolved_path = audio_path.resolve()
    if resolved_path.parent != request_directory:
        _log_acquisition_error(
            "audio download provider error",
            "audio_path_outside_request_directory",
        )
        raise _AudioDownloadProviderFailure
    if not resolved_path.is_file():
        _log_acquisition_error("audio download provider error", "audio_file_missing")
        raise _AudioDownloadProviderFailure
    return resolved_path


def _rank_caption_tracks(
    tracks: Sequence[CaptionTrack],
) -> tuple[CaptionTrack, ...]:
    """Return caption tracks in stable English-first preference order."""
    buckets: list[list[CaptionTrack]] = [[], [], [], [], [], []]
    for track in tracks:
        language_code = track.language_code.casefold()
        is_english = language_code == "en" or language_code.startswith("en-")
        if is_english:
            if track.is_generated:
                bucket = 2 if language_code == "en" else 3
            else:
                bucket = 0 if language_code == "en" else 1
        else:
            bucket = 5 if track.is_generated else 4
        buckets[bucket].append(track)
    return tuple(track for bucket in buckets for track in bucket)


def acquire_transcript(
    provider: CaptionProvider,
    video_id: str,
) -> Transcript | None:
    """Acquire the first usable dormant YouTube caption Transcript.

    Args:
        provider: Caption provider boundary.
        video_id: Stable YouTube video identity.

    Returns:
        First usable normalized caption Transcript, if any.

    Raises:
        _CaptionProviderFailure: If listing fails.
        _CaptionProviderTimeout: If listing times out.
    """
    try:
        tracks = provider.list_tracks(video_id)
        ranked_tracks = _rank_caption_tracks(tracks)
    except _CaptionProviderTimeout:
        raise
    except _CaptionProviderFailure:
        raise
    except (Timeout, TimeoutError) as exc:
        _log_acquisition_error("caption provider timeout", "track_listing_timeout")
        raise _CaptionProviderTimeout from exc
    except _CAPTION_EXTERNAL_FAILURES as exc:
        _log_acquisition_error("caption provider error", "track_listing_failed")
        raise _CaptionProviderFailure from exc

    for track in ranked_tracks:
        try:
            transcript = normalize_transcript(
                TranscriptMethod.YOUTUBE_CAPTIONS,
                track.language_code,
                track.fetch_segments(),
            )
        except _CaptionProviderTimeout:
            raise
        except (Timeout, TimeoutError) as exc:
            _log_acquisition_error("caption provider timeout", "track_fetch_timeout")
            raise _CaptionProviderTimeout from exc
        except (
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            logger.debug(
                "caption track unavailable",
                extra={
                    "stage": "transcription",
                    "reason": "track_fetch_failed",
                },
            )
            continue

        if not transcript.segments:
            continue
        logger.debug(
            "transcript acquired",
            extra={
                "stage": "transcription",
                "method": transcript.method.value,
                "segment_count": len(transcript.segments),
            },
        )
        return transcript

    return None


class WhisperAcquirer:
    """Own bounded audio acquisition and native transcription resources."""

    def __init__(
        self,
        audio_downloader: AudioDownloader,
        transcriber: WhisperTranscriber,
        settings: TranscriptionConfig,
    ) -> None:
        """Initialize process-lifetime audio and native transcription resources.

        Args:
            audio_downloader: Synchronous native-audio provider boundary.
            transcriber: Preloaded synchronous Faster-Whisper boundary.
            settings: Validated acquisition capacity and deadline configuration.
        """
        self._audio_downloader = audio_downloader
        self._transcriber = transcriber
        self._temporary_media_root = settings.temporary_media_root
        self._audio_download_timeout_seconds = settings.audio_download_timeout_seconds
        self._transcription_queue_timeout_seconds = (
            settings.transcription_queue_timeout_seconds
        )
        self._transcription_timeout_seconds = settings.transcription_timeout_seconds
        self._inference_semaphore = asyncio.Semaphore(
            settings.transcription_concurrency
        )

    async def acquire(
        self,
        source_url: str,
        prepared_audio: PreparedAudio | None,
        cancellation_event: threading.Event,
    ) -> Transcript:
        """Acquire source audio and transcribe it under the native permit.

        Args:
            source_url: Original submitted TikTok URL for yt-dlp.
            prepared_audio: Audio downloaded while recovering a missing duration.
            cancellation_event: Signal set when request work must stop.

        Returns:
            Normalized Transcript, including successful silent media.

        Raises:
            AudioDownloadFailedError: If the audio download is unusable.
            AudioDownloadTimeoutError: If the audio download times out.
            TranscriptionCapacityExceededError: If native queue capacity expires.
            TranscriptionFailedError: If native inference is unusable.
            TranscriptionTimeoutError: If native inference does not respond in time.
        """
        request_directory: Path | None = None
        try:
            audio_path, request_directory = await self._prepare_audio(
                source_url,
                prepared_audio,
                cancellation_event,
            )
            inference_permit_acquired = False
            try:
                try:
                    async with asyncio.timeout(
                        self._transcription_queue_timeout_seconds
                    ):
                        await self._inference_semaphore.acquire()
                except TimeoutError as exc:
                    raise TranscriptionCapacityExceededError() from exc
                inference_permit_acquired = True
                worker = asyncio.create_task(
                    asyncio.to_thread(
                        _transcribe_audio,
                        audio_path,
                        self._transcriber,
                    ),
                    name="native-transcription",
                )
                try:
                    async with asyncio.timeout(self._transcription_timeout_seconds):
                        return await asyncio.shield(worker)
                except TimeoutError as exc:
                    cancellation_event.set()
                    await _finish_cancelled_worker(worker)
                    raise TranscriptionTimeoutError() from exc
                except asyncio.CancelledError:
                    cancellation_event.set()
                    await _finish_cancelled_worker(worker)
                    raise
            except _TranscriptionProviderTimeout as exc:
                raise TranscriptionTimeoutError() from exc
            except _TranscriptionProviderFailure as exc:
                raise TranscriptionFailedError() from exc
            finally:
                if inference_permit_acquired:
                    self._inference_semaphore.release()
        except _AudioDownloadProviderTimeout as exc:
            raise AudioDownloadTimeoutError() from exc
        except _AudioDownloadProviderFailure as exc:
            raise AudioDownloadFailedError() from exc
        finally:
            if request_directory is not None:
                _cleanup_request_directory(request_directory)

    async def _prepare_audio(
        self,
        source_url: str,
        prepared_audio: PreparedAudio | None,
        cancellation_event: threading.Event,
    ) -> tuple[Path, Path | None]:
        """Return validated audio and any request directory this call owns."""
        if prepared_audio is not None:
            audio_path = await asyncio.to_thread(
                _validate_audio_path,
                prepared_audio.path,
                prepared_audio.request_directory,
            )
            return audio_path, None

        try:
            self._temporary_media_root.mkdir(parents=True, exist_ok=True)
            request_directory = Path(
                tempfile.mkdtemp(
                    prefix=_WHISPER_TEMP_PREFIX,
                    dir=str(self._temporary_media_root),
                )
            ).resolve()
        except OSError as exc:
            _log_acquisition_error(
                "audio download provider error",
                "temporary_root_failed",
            )
            raise _AudioDownloadProviderFailure from exc

        deadline = time.monotonic() + self._audio_download_timeout_seconds
        worker = asyncio.create_task(
            asyncio.to_thread(
                self._download_audio,
                source_url,
                request_directory,
                deadline=deadline,
                cancellation_event=cancellation_event,
            ),
            name="audio-download",
        )
        completed = False
        try:
            async with asyncio.timeout(self._audio_download_timeout_seconds):
                audio_path = await asyncio.shield(worker)
            completed = True
            return audio_path, request_directory
        except TimeoutError as exc:
            cancellation_event.set()
            await _finish_cancelled_download(worker)
            raise _AudioDownloadProviderTimeout from exc
        except asyncio.CancelledError:
            cancellation_event.set()
            await _finish_cancelled_download(worker)
            raise
        finally:
            if not completed:
                _cleanup_request_directory(request_directory)

    def _download_audio(
        self,
        source_url: str,
        request_directory: Path,
        *,
        deadline: float,
        cancellation_event: threading.Event,
    ) -> Path:
        """Download and validate one audio file inside its request directory."""
        try:
            downloaded_path = self._audio_downloader.download(
                source_url,
                request_directory,
                deadline=deadline,
                cancellation_event=cancellation_event,
            )
            return _validate_audio_path(downloaded_path, request_directory)
        except _AudioDownloadProviderTimeout:
            raise
        except _AudioDownloadProviderFailure:
            raise
        except DownloadCancelled as exc:
            raise _AudioDownloadProviderFailure from exc
        except (Timeout, TimeoutError) as exc:
            _log_acquisition_error(
                "audio download provider timeout",
                "audio_download_timeout",
            )
            raise _AudioDownloadProviderTimeout from exc
        except (
            AttributeError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            _log_acquisition_error(
                "audio download provider error",
                "audio_download_failed",
            )
            raise _AudioDownloadProviderFailure from exc


def _transcribe_audio(
    audio_path: Path,
    transcriber: WhisperTranscriber,
) -> Transcript:
    """Run native inference against one validated audio file."""
    try:
        transcript = transcriber.transcribe(audio_path)
    except _TranscriptionProviderTimeout:
        raise
    except _TranscriptionProviderFailure:
        raise
    except (Timeout, TimeoutError) as exc:
        _log_acquisition_error(
            "native transcription provider timeout",
            "transcription_timeout",
        )
        raise _TranscriptionProviderTimeout from exc
    except (
        AttributeError,
        EOFError,
        IndexError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        _log_acquisition_error(
            "native transcription provider error",
            "transcription_failed",
        )
        raise _TranscriptionProviderFailure from exc

    if not isinstance(transcript, Transcript):
        _log_acquisition_error(
            "native transcription provider error",
            "invalid_transcription_result",
        )
        raise _TranscriptionProviderFailure
    return transcript
