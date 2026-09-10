"""Application-facing orchestration for one Textify transcript request."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from requests.exceptions import RequestException, Timeout
from yt_dlp.utils import YoutubeDLError

from textify.transcription import acquisition, inspection
from textify.transcription.config import TranscriptionConfig
from textify.transcription.exceptions import (
    MetadataRetrievalFailedError,
    MetadataTimeoutError,
    TranscriptionError,
    UnsupportedContentError,
    UnsupportedPlatformError,
    VideoTooLongError,
)
from textify.transcription.types import (
    Platform,
    Source,
    TranscriptionResult,
)


@dataclass(frozen=True, slots=True)
class TranscriptionAdapters:
    """Group replaceable provider boundaries for transcript composition.

    Attributes:
        metadata_extractor: Provider boundary for Source metadata.
        caption_provider: Dormant YouTube captions provider.
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
            settings.temporary_media_root,
        ),
        caption_provider=acquisition.YouTubeCaptionProvider(),
        audio_downloader=acquisition.YtDlpAudioDownloader(),
        whisper_transcriber=acquisition.load_whisper_transcriber(settings),
    )


class TranscriptService:
    """Transcribe one submitted TikTok URL through the complete lifecycle."""

    def __init__(
        self,
        adapters: TranscriptionAdapters,
        settings: TranscriptionConfig,
        inference_semaphore: asyncio.Semaphore,
    ) -> None:
        """Initialize the service with process-lifetime dependencies.

        Args:
            adapters: Complete bundle of provider boundaries.
            settings: Validated transcription configuration.
            inference_semaphore: Gate acquired only while native inference runs.
        """
        self._adapters = adapters
        self._settings = settings
        self._inference_semaphore = inference_semaphore

    async def transcribe(self, submitted_url: str) -> TranscriptionResult:
        """Retrieve one normalized TikTok Source and Transcript.

        Args:
            submitted_url: URL supplied by the API caller.

        Returns:
            Canonical source metadata paired with a normalized timed transcript.

        Raises:
            TranscriptionError: If URL, provider, media, or transcription fails.
        """
        submitted = inspection.classify_submitted_url(submitted_url)
        if submitted.platform is not Platform.TIKTOK:
            raise UnsupportedPlatformError()

        prepared_audio: inspection.PreparedAudio | None = None
        try:
            extracted_metadata = await self._extract_metadata(submitted.provider_url)
            prepared_audio = extracted_metadata.prepared_audio
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
            if source.duration_seconds > self._settings.max_duration_seconds:
                raise VideoTooLongError()

            transcript = await acquisition.acquire_whisper(
                submitted_url,
                self._adapters.audio_downloader,
                self._adapters.whisper_transcriber,
                self._settings.temporary_media_root,
                prepared_audio,
                self._inference_semaphore,
            )
            return TranscriptionResult(source, transcript)
        except TranscriptionError:
            raise
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise MetadataRetrievalFailedError() from exc
        finally:
            if prepared_audio is not None:
                prepared_audio.cleanup()

    async def _extract_metadata(
        self,
        provider_url: str,
    ) -> inspection.ExtractedMetadata:
        """Call the metadata provider outside the event loop.

        Args:
            provider_url: Validated minimal provider URL.

        Returns:
            Raw metadata and any duration-probe audio.

        Raises:
            TranscriptionError: If the metadata stage cannot succeed safely.
        """
        try:
            extracted_metadata = await asyncio.to_thread(
                self._adapters.metadata_extractor.extract,
                provider_url,
            )
        except inspection._MetadataDurationLimitExceeded as exc:
            raise VideoTooLongError() from exc
        except TranscriptionError:
            raise
        except (Timeout, TimeoutError) as exc:
            raise MetadataTimeoutError() from exc
        except RequestException as exc:
            if inspection._is_timeout_exception(exc):
                raise MetadataTimeoutError() from exc
            raise MetadataRetrievalFailedError() from exc
        except YoutubeDLError as exc:
            if inspection._is_timeout_exception(exc):
                raise MetadataTimeoutError() from exc
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
