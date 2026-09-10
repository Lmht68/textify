"""Observable invariants for transcription domain values."""

import math
from typing import cast

import pytest

from textify.transcription.types import (
    Platform,
    Segment,
    Source,
    Transcript,
    TranscriptMethod,
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


def test_normalize_transcript_handles_invalid_language_and_silence() -> None:
    """Invalid language tags and silent media have the documented und form."""
    unknown_language = normalize_transcript(
        TranscriptMethod.FASTER_WHISPER,
        "not a language tag",
        ((0.0, 1.0, "text"),),
    )
    silent = normalize_transcript(TranscriptMethod.FASTER_WHISPER, "en", ())

    assert unknown_language.language == "und"
    assert silent == Transcript(TranscriptMethod.FASTER_WHISPER, "und", (), "")


@pytest.mark.parametrize(
    "raw_segment",
    (
        (True, 1.0, "text"),
        (0.0, math.nan, "text"),
        (1.0, 0.0, "text"),
        (0.0, 1.0, 1),
    ),
)
def test_normalize_transcript_rejects_invalid_segment_values(
    raw_segment: tuple[object, object, object],
) -> None:
    """Provider timing and text violations cannot enter the domain model."""
    with pytest.raises((TypeError, ValueError)):
        normalize_transcript(TranscriptMethod.FASTER_WHISPER, "en", (raw_segment,))
