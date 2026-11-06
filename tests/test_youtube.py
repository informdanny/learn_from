from __future__ import annotations

import json
from typing import Any

import pytest

from learn_from.youtube import (
    VideoNotFoundError,
    extract_video_id,
    fetch_channel_metadata,
    fetch_video_metadata,
    list_channel_video_ids,
    parse_iso8601_duration_to_seconds,
)


class _DummyResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> "_DummyResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def test_extract_video_id_from_supported_inputs() -> None:
    video_id = "dQw4w9WgXcQ"
    assert extract_video_id(video_id) == video_id
    assert extract_video_id(f"https://www.youtube.com/watch?v={video_id}") == video_id
    assert extract_video_id(f"https://youtu.be/{video_id}") == video_id
    assert extract_video_id(f"https://www.youtube.com/shorts/{video_id}") == video_id
    assert extract_video_id(f"https://www.youtube.com/embed/{video_id}") == video_id


def test_extract_video_id_raises_for_invalid_input() -> None:
    with pytest.raises(ValueError):
        extract_video_id("not-a-valid-video-url")


def test_parse_iso8601_duration_to_seconds() -> None:
    assert parse_iso8601_duration_to_seconds("PT5M32S") == 332
    assert parse_iso8601_duration_to_seconds("PT1H2M3S") == 3723
    assert parse_iso8601_duration_to_seconds("P1DT2H") == 93600


def test_fetch_video_metadata_success(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "items": [
            {
                "id": "dQw4w9WgXcQ",
                "snippet": {
                    "title": "Never Gonna Give You Up",
                    "description": "Classic.",
                    "channelId": "UC-CHANNEL",
                    "channelTitle": "Rick",
                    "publishedAt": "2009-10-25T06:57:33Z",
                    "thumbnails": {"high": {"url": "https://img.example/high.jpg"}},
                },
                "contentDetails": {"duration": "PT3M33S"},
                "statistics": {"viewCount": "1000", "likeCount": "22", "commentCount": "3"},
            }
        ]
    }

    def _fake_urlopen(*args: Any, **kwargs: Any) -> _DummyResponse:
        return _DummyResponse(payload)

    monkeypatch.setattr("learn_from.youtube.urlopen", _fake_urlopen)
    metadata = fetch_video_metadata(video_id_or_url="dQw4w9WgXcQ", api_key="test-key")

    assert metadata.video_id == "dQw4w9WgXcQ"
    assert metadata.duration_seconds == 213
    assert metadata.views == 1000
    assert metadata.likes == 22
    assert metadata.comments == 3
    assert metadata.thumbnail_url == "https://img.example/high.jpg"


def test_fetch_video_metadata_raises_when_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_urlopen(*args: Any, **kwargs: Any) -> _DummyResponse:
        return _DummyResponse({"items": []})

    monkeypatch.setattr("learn_from.youtube.urlopen", _fake_urlopen)
    with pytest.raises(VideoNotFoundError):
        fetch_video_metadata(video_id_or_url="dQw4w9WgXcQ", api_key="test-key")


def test_fetch_channel_metadata_success(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "items": [
            {
                "id": "UC-test-channel",
                "snippet": {
                    "title": "Data Over Dogma",
                    "description": "Channel description",
                    "thumbnails": {"high": {"url": "https://img.example/channel.jpg"}},
                },
                "statistics": {"subscriberCount": "12345"},
            }
        ]
    }

    def _fake_urlopen(*args: Any, **kwargs: Any) -> _DummyResponse:
        return _DummyResponse(payload)

    monkeypatch.setattr("learn_from.youtube.urlopen", _fake_urlopen)
    channel = fetch_channel_metadata(channel_id="UC-test-channel", api_key="test-key")

    assert channel.channel_id == "UC-test-channel"
    assert channel.channel_name == "Data Over Dogma"
    assert channel.subscriber_count == 12345
    assert channel.thumbnail_url == "https://img.example/channel.jpg"


def test_list_channel_video_ids_filters_invalid_and_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "items": [
            {"id": {"videoId": "dQw4w9WgXcQ"}},
            {"id": {"videoId": "dQw4w9WgXcQ"}},
            {"id": {"videoId": "M7lc1UVf-VE"}},
            {"id": {"videoId": "short"}},
            {"id": {"videoId": ""}},
        ]
    }

    def _fake_urlopen(*args: Any, **kwargs: Any) -> _DummyResponse:
        return _DummyResponse(payload)

    monkeypatch.setattr("learn_from.youtube.urlopen", _fake_urlopen)
    video_ids = list_channel_video_ids(channel_id="UC-test-channel", api_key="test-key", max_results=10)

    assert video_ids == ["dQw4w9WgXcQ", "M7lc1UVf-VE"]
