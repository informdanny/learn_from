from __future__ import annotations

import math

import pytest

from learn_from.transcripts import (
    TranscriptNotAvailableError,
    TranscriptRetrievalError,
    _select_track,
    fetch_transcript,
    format_timestamp,
    format_transcript_with_timestamps,
)


def test_format_timestamp_supports_mm_ss_and_hh_mm_ss() -> None:
    assert format_timestamp(59) == "00:59"
    assert format_timestamp(3723) == "01:02:03"


def test_fetch_transcript_normalizes_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_fetch_raw_transcript(**kwargs):
        return (
            [
                {"text": " Hello ", "start": 0, "duration": 1.5},
                {"text": "world", "start": 62.0, "duration": 2},
            ],
            "en",
            False,
        )

    monkeypatch.setattr("learn_from.transcripts._fetch_raw_transcript", _fake_fetch_raw_transcript)
    transcript = fetch_transcript(video_id_or_url="dQw4w9WgXcQ")

    assert transcript.video_id == "dQw4w9WgXcQ"
    assert transcript.language_code == "en"
    assert transcript.is_generated is False
    assert transcript.full_text == "Hello\nworld"
    assert len(transcript.timestamped_segments) == 2
    assert transcript.timestamped_segments[1]["start_seconds"] == 62.0


def test_fetch_transcript_supports_object_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Entry:
        def __init__(self, text: str, start: float, duration: float | None) -> None:
            self.text = text
            self.start = start
            self.duration = duration

    def _fake_fetch_raw_transcript(**kwargs):
        return ([_Entry("One", 1.0, 2.0), _Entry("Two", 4.0, None)], "en", True)

    monkeypatch.setattr("learn_from.transcripts._fetch_raw_transcript", _fake_fetch_raw_transcript)
    transcript = fetch_transcript(video_id_or_url="dQw4w9WgXcQ")

    assert transcript.is_generated is True
    assert transcript.timestamped_segments[0]["text"] == "One"
    assert transcript.timestamped_segments[1]["duration_seconds"] is None


def test_format_transcript_with_timestamps_renders_lines() -> None:
    transcript = fetch_transcript_for_render_test()
    rendered = format_transcript_with_timestamps(transcript)
    assert "[00:00] First line" in rendered
    assert "[01:02] Second line" in rendered


def fetch_transcript_for_render_test():
    class _DummyTranscript:
        segments = [
            type("S", (), {"text": "First line", "start_seconds": 0})(),
            type("S", (), {"text": "Second line", "start_seconds": 62})(),
        ]

    return _DummyTranscript()


def test_fetch_transcript_propagates_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_fetch_raw_transcript(**kwargs):
        raise TranscriptNotAvailableError("no transcript")

    monkeypatch.setattr("learn_from.transcripts._fetch_raw_transcript", _fake_fetch_raw_transcript)
    with pytest.raises(TranscriptNotAvailableError):
        fetch_transcript(video_id_or_url="dQw4w9WgXcQ")


def test_fetch_transcript_propagates_retrieval_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_fetch_raw_transcript(**kwargs):
        raise TranscriptRetrievalError("network issue")

    monkeypatch.setattr("learn_from.transcripts._fetch_raw_transcript", _fake_fetch_raw_transcript)
    with pytest.raises(TranscriptRetrievalError):
        fetch_transcript(video_id_or_url="dQw4w9WgXcQ")


def test_fetch_transcript_forwards_proxy_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_fetch_raw_transcript(**kwargs):
        captured.update(kwargs)
        return ([{"text": "x", "start": 0, "duration": 1}], "en", False)

    monkeypatch.setattr("learn_from.transcripts._fetch_raw_transcript", _fake_fetch_raw_transcript)
    fetch_transcript(
        video_id_or_url="dQw4w9WgXcQ",
        proxy_http_url="http://proxy.example:8080",
        proxy_https_url="https://proxy.example:8443",
    )

    assert captured["proxy_http_url"] == "http://proxy.example:8080"
    assert captured["proxy_https_url"] == "https://proxy.example:8443"


def test_fetch_transcript_uses_transcript_io_when_api_key_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_fetch_from_transcript_io(**kwargs):
        captured.update(kwargs)
        return ([{"text": "hello", "start": "1.5", "dur": "2.5"}], "en")

    def _fake_fetch_raw_transcript(**kwargs):
        raise AssertionError("Should not call youtube-transcript-api path when transcript.io key is set")

    monkeypatch.setattr(
        "learn_from.transcripts._fetch_from_youtube_transcript_io",
        _fake_fetch_from_transcript_io,
    )
    monkeypatch.setattr("learn_from.transcripts._fetch_raw_transcript", _fake_fetch_raw_transcript)

    transcript = fetch_transcript(
        video_id_or_url="dQw4w9WgXcQ",
        transcript_io_api_key="test-api-key",
        languages=["en-US", "en"],
    )
    assert transcript.language_code == "en"
    assert transcript.timestamped_segments[0]["start_seconds"] == 1.5
    assert transcript.timestamped_segments[0]["duration_seconds"] == 2.5
    assert captured["api_key"] == "test-api-key"
    assert captured["languages"] == ["en-US", "en"]


def test_select_track_prefers_requested_language() -> None:
    tracks = [
        {"language": "fr", "transcript": []},
        {"language": "en", "transcript": []},
    ]
    selected = _select_track(tracks=tracks, languages=["en-US", "en"])
    assert selected is not None
    assert selected["language"] == "en"


def test_fetch_transcript_normalizes_non_finite_numeric_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_fetch_raw_transcript(**kwargs):
        return (
            [
                {"text": "A", "start": "NaN", "duration": "Infinity"},
                {"text": "B", "start": math.inf, "duration": 1.2},
            ],
            "en",
            False,
        )

    monkeypatch.setattr("learn_from.transcripts._fetch_raw_transcript", _fake_fetch_raw_transcript)
    transcript = fetch_transcript(video_id_or_url="dQw4w9WgXcQ")

    assert transcript.timestamped_segments[0]["start_seconds"] == 0.0
    assert transcript.timestamped_segments[0]["duration_seconds"] is None
    assert transcript.timestamped_segments[1]["start_seconds"] == 0.0
    assert transcript.timestamped_segments[1]["duration_seconds"] == 1.2
