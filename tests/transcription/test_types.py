"""Observable invariants for transcription domain values."""

import math
from typing import cast

import pytest

from textify.transcription.types import (
    Platform,
    Segment,
    Source,
    TimedTranscript,
    Transcript,
    TranscriptMethod,
    normalize_language_tag,
    normalize_transcript,
)


def test_source_requires_positive_integer_duration() -> None:
    """Source rejects zero, booleans, and fractional duration values."""
    for duration in (0, True, 1.5):
        with pytest.raises((TypeError, ValueError)):
            Source(
                Platform.TIKTOK,
                "123",
                "https://www.tiktok.com/@creator/video/123",
                "Title",
                "",
                "Creator",
                cast(int, duration),
            )


def test_normalize_transcript_sorts_and_collapses_segments() -> None:
    """Raw segments become stable start-sorted, whitespace-normalized output."""
    transcript = normalize_transcript(
        TranscriptMethod.FASTER_WHISPER,
        "EN_us",
        (
            (2.0, 3.0, "  second   segment "),
            (1.0, 1.0, " first "),
            (2.0, 2.5, " overlapping "),
            (4.0, 4.0, "\t"),
        ),
    )

    assert transcript.language == "en-US"
    assert transcript.text == "first second segment overlapping"
    assert transcript.segments == (
        Segment(1.0, 1.0, "first"),
        Segment(2.0, 3.0, "second segment"),
        Segment(2.0, 2.5, "overlapping"),
    )


@pytest.mark.parametrize("include_segments", (True, False))
def test_normalize_transcript_handles_invalid_language_and_silence(
    include_segments: bool,
) -> None:
    """Invalid language tags and silent media retain canonical output forms."""
    unknown_language = normalize_transcript(
        TranscriptMethod.FASTER_WHISPER,
        "not a language tag",
        ((0.0, 1.0, "text"),),
        include_segments=include_segments,
    )
    silent = normalize_transcript(
        TranscriptMethod.FASTER_WHISPER,
        "en",
        (),
        include_segments=include_segments,
    )

    assert unknown_language.language == "und"
    if include_segments:
        assert silent == TimedTranscript(
            method=TranscriptMethod.FASTER_WHISPER,
            language="und",
            text="",
            segments=(),
        )
    else:
        assert silent == Transcript(
            method=TranscriptMethod.FASTER_WHISPER,
            language="und",
            text="",
        )


def test_normalize_transcript_omits_segment_construction_when_not_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Text-only normalization retains sorted text without constructing Segments."""

    def fail_segment_construction(*args: object, **kwargs: object) -> None:
        """Fail when text-only normalization tries to construct a Segment."""
        del args, kwargs
        raise AssertionError("text-only normalization constructed a Segment")

    monkeypatch.setattr(
        "textify.transcription.types.Segment",
        fail_segment_construction,
    )

    transcript = normalize_transcript(
        TranscriptMethod.FASTER_WHISPER,
        "EN_us",
        (
            (2.0, 3.0, "  second   segment "),
            (1.0, 1.0, " first "),
            (2.0, 2.5, " overlapping "),
            (4.0, 4.0, "\t"),
        ),
        include_segments=False,
    )

    assert type(transcript) is Transcript
    assert transcript.language == "en-US"
    assert transcript.text == "first second segment overlapping"
    assert not hasattr(transcript, "segments")


@pytest.mark.parametrize(
    ("language", "expected"),
    (
        ("en_us", "en-US"),
        ("not a language tag", "und"),
        (None, "und"),
        (1, "und"),
    ),
)
def test_normalize_language_tag_handles_provider_values(
    language: object,
    expected: str,
) -> None:
    """Provider language declarations canonicalize only valid string tags."""
    assert normalize_language_tag(language) == expected


@pytest.mark.parametrize(
    "raw_segment",
    (
        (True, 1.0, "text"),
        (0.0, math.nan, "text"),
        (1.0, 0.0, "text"),
        (0.0, 1.0, 1),
    ),
)
@pytest.mark.parametrize("include_segments", (True, False))
def test_normalize_transcript_rejects_invalid_segment_values(
    raw_segment: tuple[object, object, object],
    include_segments: bool,
) -> None:
    """Provider timing and text violations cannot enter either transcript form."""
    with pytest.raises((TypeError, ValueError)):
        normalize_transcript(
            TranscriptMethod.FASTER_WHISPER,
            "en",
            (raw_segment,),
            include_segments=include_segments,
        )
