"""Domain types and normalization for Textify transcription."""

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from math import isfinite
from numbers import Real
from typing import Literal

from langcodes import standardize_tag
from langcodes.tag_parser import LanguageTagError

RawSegment = tuple[object, object, object]

ResponseFieldPath = Literal[
    "source.platform",
    "source.video_id",
    "source.url",
    "source.title",
    "source.description",
    "source.channel",
    "source.duration_seconds",
    "transcript.method",
    "transcript.language",
    "transcript.text",
    "transcript.segments",
]


class Platform(StrEnum):
    """Content platforms known to the transcription context."""

    YOUTUBE = "youtube"
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"
    TIKTOK = "tiktok"
    X = "x"


class TranscriptMethod(StrEnum):
    """Methods used to acquire a normalized transcript."""

    YOUTUBE_CAPTIONS = "youtube_captions"
    FASTER_WHISPER = "faster_whisper"


@dataclass(frozen=True, slots=True)
class Source:
    """Contain canonical source identity and normalized video metadata.

    Attributes:
        platform: Platform hosting the source.
        video_id: Stable external identifier for the source.
        url: Validated canonical URL returned or reconstructed for the source.
        title: Provider-supplied video title.
        description: Complete provider-supplied video description.
        channel: Provider-supplied channel name.
        duration_seconds: Positive duration rounded up to whole seconds.
    """

    platform: Platform
    video_id: str
    url: str
    title: str
    description: str
    channel: str
    duration_seconds: int

    def __post_init__(self) -> None:
        """Validate source identity and duration invariants."""
        if not isinstance(self.platform, Platform):
            raise TypeError("platform must be a Platform.")
        if not all(
            isinstance(value, str)
            for value in (
                self.video_id,
                self.url,
                self.title,
                self.description,
                self.channel,
            )
        ):
            raise TypeError("source text fields must be strings.")
        if (
            isinstance(self.duration_seconds, bool)
            or not isinstance(self.duration_seconds, int)
            or self.duration_seconds <= 0
        ):
            raise ValueError("duration_seconds must be a positive integer.")


@dataclass(frozen=True, slots=True)
class Segment:
    """Contain one normalized timed transcript segment.

    Attributes:
        start: Segment start offset in seconds.
        end: Segment end offset in seconds.
        text: Nonempty whitespace-normalized segment text.
    """

    start: float
    end: float
    text: str

    def __post_init__(self) -> None:
        """Validate timing and normalized text invariants."""
        start = _validated_time(self.start, "start")
        end = _validated_time(self.end, "end")
        if end < start:
            raise ValueError("end must not precede start.")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string.")
        if not self.text or self.text != _collapse_whitespace(self.text):
            raise ValueError("text must be nonempty and whitespace-normalized.")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)


@dataclass(frozen=True, slots=True)
class Transcript:
    """Contain normalized transcript text and its acquisition method.

    Attributes:
        method: Provider-independent method that produced the transcript.
        language: Canonical BCP 47 language tag, or ``und`` when unknown.
        text: Whitespace-normalized transcript text.
    """

    method: TranscriptMethod
    language: str
    text: str

    def __post_init__(self) -> None:
        """Validate transcript text and canonical language."""
        if not isinstance(self.method, TranscriptMethod):
            raise TypeError("method must be a TranscriptMethod.")
        if not isinstance(
            self.language, str
        ) or self.language != normalize_language_tag(self.language):
            raise ValueError("language must be a canonical BCP 47 tag.")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string.")
        if self.text != _collapse_whitespace(self.text):
            raise ValueError("text must be whitespace-normalized.")
        if not self.text and self.language != "und":
            raise ValueError("silent media must use the und language tag.")


@dataclass(frozen=True, slots=True)
class TimedTranscript(Transcript):
    """Contain normalized transcript text with its timed segments.

    Attributes:
        segments: Start-sorted normalized transcript segments.
    """

    segments: tuple[Segment, ...]

    def __post_init__(self) -> None:
        """Validate segment ordering and exact text composition."""
        Transcript.__post_init__(self)
        if not isinstance(self.segments, tuple) or not all(
            isinstance(segment, Segment) for segment in self.segments
        ):
            raise TypeError("segments must be a tuple of Segment values.")
        if any(
            later.start < earlier.start
            for earlier, later in zip(self.segments, self.segments[1:])
        ):
            raise ValueError("segments must be sorted by start time.")
        expected_text = " ".join(segment.text for segment in self.segments)
        if self.text != expected_text:
            raise ValueError("text must exactly join the segment text.")


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Pair normalized source metadata with its transcript.

    Attributes:
        source: Canonical source identity and metadata.
        transcript: Normalized timed transcript.
    """

    source: Source
    transcript: Transcript

    def __post_init__(self) -> None:
        """Validate the paired domain values."""
        if not isinstance(self.source, Source) or not isinstance(
            self.transcript, Transcript
        ):
            raise TypeError("source and transcript must be domain values.")


def normalize_transcript(
    method: TranscriptMethod,
    language: object,
    raw_segments: Iterable[RawSegment],
    *,
    include_segments: bool = True,
) -> Transcript:
    """Normalize raw provider segments into a valid transcript.

    Args:
        method: Provider-independent acquisition method.
        language: Provider-reported language tag.
        raw_segments: ``(start, end, text)`` values from a provider.
        include_segments: Whether to retain normalized timed segments.

    Returns:
        A text-only Transcript or TimedTranscript with collapsed whitespace.

    Raises:
        TypeError: If a raw segment has an invalid shape, time, or text type.
        ValueError: If a raw segment has invalid timing.
    """
    normalized_records: list[tuple[float, float, str]] = []
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, tuple) or len(raw_segment) != 3:
            raise TypeError("raw segments must contain start, end, and text.")
        start, end, text = raw_segment
        normalized_start = _validated_time(start, "start")
        normalized_end = _validated_time(end, "end")
        if normalized_end < normalized_start:
            raise ValueError("end must not precede start.")
        if not isinstance(text, str):
            raise TypeError("raw segment text must be a string.")
        normalized_text = _collapse_whitespace(text)
        if normalized_text:
            normalized_records.append(
                (normalized_start, normalized_end, normalized_text)
            )

    normalized_records.sort(key=lambda record: record[0])
    normalized_language = normalize_language_tag(language)
    if not normalized_records:
        normalized_language = "und"
    normalized_text = " ".join(record[2] for record in normalized_records)

    if not include_segments:
        return Transcript(
            method=method,
            language=normalized_language,
            text=normalized_text,
        )

    return TimedTranscript(
        method=method,
        language=normalized_language,
        text=normalized_text,
        segments=tuple(
            Segment(start, end, text) for start, end, text in normalized_records
        ),
    )


def normalize_language_tag(language: object) -> str:
    """Return a canonical BCP 47 language tag or ``und``.

    Args:
        language: Provider-reported language tag.

    Returns:
        A canonical BCP 47 language tag, or ``und`` when the value is invalid.
    """
    if not isinstance(language, str) or not language.strip():
        return "und"
    try:
        return standardize_tag(language.strip())
    except LanguageTagError:
        return "und"


def _collapse_whitespace(text: str) -> str:
    """Collapse all Unicode whitespace into single spaces."""
    return " ".join(text.split())


def _validated_time(value: object, name: str) -> float:
    """Return one finite nonnegative time value.

    Args:
        value: Provider value intended to represent seconds.
        name: Timing-field name for the validation message.

    Returns:
        The validated time as a float.

    Raises:
        TypeError: If the value is not numeric or is a boolean.
        ValueError: If the value is nonfinite or negative.
    """
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise TypeError(f"{name} must be numeric.")
    normalized_value = float(value)
    if not isfinite(normalized_value) or normalized_value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return normalized_value
