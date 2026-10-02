"""Application-facing orchestration for one Textify transcript request."""

import asyncio
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from requests.exceptions import RequestException, Timeout
from yt_dlp.utils import YoutubeDLError

from textify.logging import bind_request_log_fields
from textify.transcription import acquisition, inspection
from textify.transcription.config import TranscriptionConfig
from textify.transcription.exceptions import (
    MetadataRetrievalFailedError,
    MetadataTimeoutError,
    TranscriptionCancellationRequestedError,
    TranscriptionError,
    UnsupportedContentError,
    UnsupportedMediaError,
    VideoTooLongError,
)
from textify.transcription.types import (
    Platform,
    Source,
    Transcript,
    TranscriptionResult,
)
from textify.transcription.util import MediaByteLimitExceeded


@dataclass(frozen=True, slots=True)
class TranscriptionAdapters:
    """Group replaceable provider boundaries for transcript composition.

    Attributes:
        metadata_extractor: Provider boundary for Source metadata.
        caption_provider: Active YouTube captions provider.
        audio_downloader: Provider boundary for request-scoped audio.
        whisper_transcriber: Preloaded native inference provider.
    """

    metadata_extractor: inspection.MetadataExtractor
    caption_provider: acquisition.CaptionProvider
    audio_downloader: acquisition.AudioDownloader
    whisper_transcriber: acquisition.WhisperTranscriber


type TranscriptionAdaptersFactory = Callable[
    [TranscriptionConfig], TranscriptionAdapters
]


@dataclass(slots=True)
class TranscriptionExecutionControl:
    """Own cancellation and cleanup signals for one executor invocation."""

    provider_cancellation: threading.Event = field(default_factory=threading.Event)
    cancellation_requested: asyncio.Event = field(default_factory=asyncio.Event)
    cleanup_complete: asyncio.Event = field(default_factory=asyncio.Event)

    def request_cancellation(self) -> None:
        """Signal an accepted durable cancellation to cooperative providers."""
        self.provider_cancellation.set()
        self.cancellation_requested.set()


def _raise_if_cancellation_requested(
    control: TranscriptionExecutionControl,
) -> None:
    """Stop result production after durable cancellation has been accepted."""
    if control.cancellation_requested.is_set():
        raise TranscriptionCancellationRequestedError()


def build_transcription_adapters(
    settings: TranscriptionConfig,
) -> TranscriptionAdapters:
    """Construct the production provider adapters once per process.

    Args:
        settings: Validated transcription configuration.

    Returns:
        Fully constructed production adapter bundle.
    """
    return TranscriptionAdapters(
        metadata_extractor=inspection.YtDlpMetadataExtractor(
            settings.max_duration_seconds,
            settings.max_media_bytes,
            settings.temporary_media_root,
        ),
        caption_provider=acquisition.YouTubeCaptionProvider(),
        audio_downloader=acquisition.YtDlpAudioDownloader(settings.max_media_bytes),
        whisper_transcriber=acquisition.load_whisper_transcriber(settings),
    )


async def _finish_cancelled_metadata(
    worker: asyncio.Task[inspection.ExtractedMetadata],
) -> None:
    """Consume cancelled metadata work and clean unadopted prepared audio."""
    expected_failures = (
        inspection._MetadataDurationLimitExceeded,
        TranscriptionError,
        RequestException,
        YoutubeDLError,
        Timeout,
        TimeoutError,
        AttributeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    )
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            continue
        except expected_failures:
            return
    try:
        extracted_metadata = worker.result()
    except expected_failures:
        return
    if (
        isinstance(extracted_metadata, inspection.ExtractedMetadata)
        and extracted_metadata.prepared_audio is not None
    ):
        extracted_metadata.prepared_audio.cleanup()


class TranscriptionExecutor:
    """Execute one validated transcription lifecycle outside HTTP concerns.

    Callers supply a supported SubmittedSource and execution control owned by the
    orchestration layer.
    """

    def __init__(
        self,
        adapters: TranscriptionAdapters,
        settings: TranscriptionConfig,
    ) -> None:
        """Initialize process-lifetime provider execution resources.

        Args:
            adapters: Complete bundle of provider boundaries.
            settings: Validated transcription configuration.
        """
        self._metadata_extractor = adapters.metadata_extractor
        self._caption_provider = adapters.caption_provider
        self._settings = settings
        self._whisper_acquirer = acquisition.WhisperAcquirer(
            adapters.audio_downloader,
            adapters.whisper_transcriber,
            settings,
        )

    async def execute(
        self,
        submitted: inspection.SubmittedSource,
        *,
        control: TranscriptionExecutionControl,
        include_segments: bool = True,
    ) -> TranscriptionResult:
        """Execute one validated supported source through transcript composition.

        Args:
            submitted: Supported SubmittedSource produced by URL inspection.
            control: Caller-owned cancellation and cleanup control.
            include_segments: Whether to retain normalized timed segments.

        Returns:
            Canonical source metadata paired with a normalized transcript.

        Raises:
            TranscriptionCancellationRequestedError: If durable cancellation was
                accepted before a result can be returned.
            TranscriptionError: If provider, media, or native work fails safely.
            asyncio.CancelledError: If the caller cancels execution.
        """
        prepared_audio: inspection.PreparedAudio | None = None
        ownership: acquisition.TranscriptionOwnership | None = None
        try:
            extracted_metadata = await self._extract_metadata(
                submitted.provider_url,
                control.provider_cancellation,
            )
            prepared_audio = extracted_metadata.prepared_audio
            _raise_if_cancellation_requested(control)
            inspection._raise_if_selected_media_exceeds_limit(
                extracted_metadata.metadata,
                self._settings.max_media_bytes,
            )
            normalized_metadata = inspection.normalize_processed_metadata(
                extracted_metadata.metadata,
                submitted,
            )
            source = Source(
                platform=submitted.platform,
                video_id=normalized_metadata.video_id,
                url=normalized_metadata.canonical_url,
                title=normalized_metadata.title,
                description=normalized_metadata.description,
                channel=normalized_metadata.channel,
                duration_seconds=normalized_metadata.duration_seconds,
            )
            bind_request_log_fields(
                source_id=source.video_id,
                duration_seconds=source.duration_seconds,
            )
            if source.duration_seconds > self._settings.max_duration_seconds:
                raise VideoTooLongError()

            if source.platform is Platform.YOUTUBE:
                video_id = submitted.youtube_video_id
                if video_id is not None:
                    caption_transcript = await self._acquire_youtube_caption(
                        video_id,
                        normalized_metadata.declared_language,
                        control.provider_cancellation,
                        include_segments=include_segments,
                    )
                    _raise_if_cancellation_requested(control)
                    if caption_transcript is not None:
                        bind_request_log_fields(
                            method=caption_transcript.method.value,
                        )
                        return TranscriptionResult(source, caption_transcript)

            _raise_if_cancellation_requested(control)
            ownership = acquisition.TranscriptionOwnership(
                prepared_audio,
                control.provider_cancellation,
                control.cleanup_complete,
            )
            prepared_audio = None
            _raise_if_cancellation_requested(control)
            transcript = await self._whisper_acquirer.acquire(
                submitted.provider_url,
                ownership,
                cancellation_requested=control.cancellation_requested,
                include_segments=include_segments,
            )
            _raise_if_cancellation_requested(control)
            bind_request_log_fields(method=transcript.method.value)
            return TranscriptionResult(source, transcript)
        except MediaByteLimitExceeded as exc:
            raise UnsupportedMediaError() from exc
        except TranscriptionError:
            raise
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise MetadataRetrievalFailedError() from exc
        finally:
            if ownership is None:
                control.provider_cancellation.set()
                try:
                    if prepared_audio is not None:
                        prepared_audio.cleanup()
                finally:
                    control.cleanup_complete.set()
            else:
                ownership.finish_caller()

    async def shutdown(self) -> None:
        """Drain retained native work and its caller-owned media cleanup.

        Returns:
            None after every retained native finalizer has completed.
        """
        await self._whisper_acquirer.shutdown()

    async def _acquire_youtube_caption(
        self,
        video_id: str,
        declared_language: str | None,
        cancellation_event: threading.Event,
        *,
        include_segments: bool = True,
    ) -> Transcript | None:
        """Acquire optional YouTube captions without blocking the event loop.

        Args:
            video_id: Stable external YouTube source identifier.
            declared_language: Canonical provider-declared source language.
            cancellation_event: Cooperative worker cancellation signal.
            include_segments: Whether to retain normalized timed segments.

        Returns:
            A usable caption transcript, or ``None`` when captions are unavailable.

        Raises:
            TranscriptionCancellationRequestedError: If Cancellation wins during
                caption acquisition.
        """
        try:
            return await asyncio.to_thread(
                acquisition.acquire_transcript,
                self._caption_provider,
                video_id,
                declared_language,
                cancellation_event,
                include_segments=include_segments,
            )
        except acquisition._CaptionProviderTimeout:
            return None
        except acquisition._CaptionProviderFailure:
            return None

    async def _extract_metadata(
        self,
        provider_url: str,
        cancellation_event: threading.Event,
    ) -> inspection.ExtractedMetadata:
        """Retrieve metadata within its full provider-operation deadline.

        Args:
            provider_url: Validated provider URL.
            cancellation_event: Signal set when request work must stop.

        Returns:
            Raw metadata and any duration-probe audio.

        Raises:
            TranscriptionError: If the metadata stage cannot succeed safely.
        """
        deadline = time.monotonic() + self._settings.metadata_timeout_seconds
        worker = asyncio.create_task(
            asyncio.to_thread(
                self._metadata_extractor.extract,
                provider_url,
                deadline=deadline,
                cancellation_event=cancellation_event,
            ),
            name="metadata-extraction",
        )
        try:
            async with asyncio.timeout(self._settings.metadata_timeout_seconds):
                extracted_metadata = await asyncio.shield(worker)
        except TimeoutError as exc:
            cancellation_event.set()
            await _finish_cancelled_metadata(worker)
            raise MetadataTimeoutError() from exc
        except asyncio.CancelledError:
            cancellation_event.set()
            await _finish_cancelled_metadata(worker)
            raise
        except inspection._MetadataDurationLimitExceeded as exc:
            raise VideoTooLongError() from exc
        except TranscriptionError:
            raise
        except Timeout as exc:
            raise MetadataTimeoutError() from exc
        except RequestException as exc:
            if inspection._is_timeout_exception(exc):
                raise MetadataTimeoutError() from exc
            raise MetadataRetrievalFailedError() from exc
        except MediaByteLimitExceeded as exc:
            raise UnsupportedMediaError() from exc
        except YoutubeDLError as exc:
            if inspection._is_timeout_exception(exc):
                raise MetadataTimeoutError() from exc
            if inspection._is_unsupported_media_error(str(exc)):
                raise UnsupportedMediaError() from exc
            if inspection._is_unavailable_error(str(exc)):
                raise UnsupportedContentError() from exc
            raise MetadataRetrievalFailedError() from exc
        except (
            AttributeError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            if inspection._is_timeout_exception(exc):
                raise MetadataTimeoutError() from exc
            raise MetadataRetrievalFailedError() from exc

        if not isinstance(extracted_metadata, inspection.ExtractedMetadata):
            raise MetadataRetrievalFailedError()
        return extracted_metadata
