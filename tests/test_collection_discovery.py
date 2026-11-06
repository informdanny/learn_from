from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from learn_from.collection_discovery import _channel_selector, discover_collection_plan
from learn_from.youtube import YouTubeVideoMetadata


CHANNEL_A = "UC" + "A" * 22
CHANNEL_B = "UC" + "B" * 22
AFTER = datetime(2025, 11, 6, tzinfo=UTC)
BEFORE = datetime(2026, 11, 6, tzinfo=UTC)


def _video(number: int, *, channel: str = CHANNEL_A, date: str = "2026-10-01T00:00:00Z", duration: int | None = 600):
    video_id = f"v{number:010}"
    return YouTubeVideoMetadata(
        video_id=video_id, title=f"Episode {number}", description="Not retained in plan",
        channel_id=channel, channel_name=channel, url=f"https://www.youtube.com/watch?v={video_id}",
        publish_date=date, duration_seconds=duration, views=1, likes=1, comments=1, thumbnail_url=None,
    )


def _item(video: YouTubeVideoMetadata):
    return {"contentDetails": {"videoId": video.video_id, "videoPublishedAt": video.publish_date}}


def _mock_api(monkeypatch, channel_pages, videos):
    calls = []

    def _request(**kwargs):
        calls.append(kwargs)
        params = kwargs["params"]
        if kwargs["endpoint"].endswith("/channels"):
            channel_id = params.get("id") or (CHANNEL_A if params["forHandle"] == "alpha" else CHANNEL_B)
            return {"items": [{
                "id": channel_id, "snippet": {"title": channel_id, "description": "Channel"},
                "statistics": {"subscriberCount": "100"},
                "contentDetails": {"relatedPlaylists": {"uploads": f"uploads-{channel_id}"}},
            }]}
        channel_id = params["playlistId"].removeprefix("uploads-")
        page = int(params.get("pageToken", "0"))
        pages = channel_pages[channel_id]
        result = {"items": pages[page]}
        if page + 1 < len(pages):
            result["nextPageToken"] = str(page + 1)
        return result

    batches = []

    def _metadata(**kwargs):
        ids = list(kwargs["video_ids"])
        batches.append(ids)
        assert len(ids) <= 50
        return [videos[video_id] for video_id in ids if video_id in videos]

    monkeypatch.setattr("learn_from.collection_discovery._request_youtube_json", _request)
    monkeypatch.setattr("learn_from.collection_discovery.fetch_videos_metadata", _metadata)
    return calls, batches


@pytest.mark.parametrize("value,expected", [
    (CHANNEL_A, ("id", CHANNEL_A)),
    (f"https://www.youtube.com/channel/{CHANNEL_A}/videos", ("id", CHANNEL_A)),
    ("@alpha", ("forHandle", "alpha")),
    ("alpha", ("forHandle", "alpha")),
    ("https://www.youtube.com/@alpha/videos", ("forHandle", "alpha")),
    ("youtube.com/@alpha", ("forHandle", "alpha")),
])
def test_resolves_supported_channel_inputs(value, expected):
    assert _channel_selector(value) == expected


@pytest.mark.parametrize("value", ["", "https://youtube.com.evil.test/@alpha", "https://example.com/@alpha", "https://youtube.com/watch?v=abc", "bad handle"])
def test_rejects_invalid_channel_inputs(value):
    with pytest.raises(ValueError):
        _channel_selector(value)


def test_all_pages_date_duration_filter_and_fair_channel_order(monkeypatch):
    a_new = _video(1, date="2026-10-03T00:00:00Z")
    a_old = _video(2, date="2026-09-03T00:00:00Z")
    short = _video(3, duration=179)
    excluded_old = _video(4, date="2025-11-05T23:59:59Z")
    b = _video(5, channel=CHANNEL_B)
    unknown_duration = _video(6, duration=None)
    missing = _video(7)
    videos = {video.video_id: video for video in [a_new, a_old, short, excluded_old, b, unknown_duration]}
    calls, _ = _mock_api(monkeypatch, {
        CHANNEL_A: [[_item(a_new), _item(excluded_old), _item(short)], [_item(a_old), _item(a_new), _item(unknown_duration), _item(missing)]],
        CHANNEL_B: [[_item(b)]],
    }, videos)
    result = discover_collection_plan(channel_inputs=["@alpha", "https://www.youtube.com/@beta"], youtube_api_key="test", published_after=AFTER, published_before=BEFORE)
    assert [item["video_id"] for item in result["items"]] == [a_new.video_id, b.video_id, a_old.video_id]
    assert result["discovery_complete"] is True
    assert result["selected_count"] == 3
    status = result["channel_discovery"][0]
    assert status["pages_scanned"] == 2
    assert status["skipped_counts"]["outside_window"] == 1
    assert status["skipped_counts"]["below_min_duration"] == 1
    assert status["skipped_counts"]["unknown_duration"] == 1
    assert status["skipped_counts"]["missing_metadata"] == 1
    assert status["skipped_counts"]["duplicate"] == 1
    assert "description" not in result["items"][0]
    assert all("search" not in call["endpoint"] for call in calls)
    json.dumps(result)


def test_page_cap_marks_partial_and_never_claims_complete(monkeypatch):
    first, second = _video(1), _video(2)
    _mock_api(monkeypatch, {CHANNEL_A: [[_item(first)], [_item(second)]]}, {first.video_id: first, second.video_id: second})
    result = discover_collection_plan(channel_inputs=[CHANNEL_A], youtube_api_key="test", published_after=AFTER, published_before=BEFORE, max_pages_per_channel=1)
    assert result["discovery_complete"] is False
    assert "page cap" in result["channel_discovery"][0]["discovery_warning"]
    assert result["selected_count"] == 1


def test_batches_metadata_at_50_with_inclusive_lower_and_exclusive_upper_bound(monkeypatch):
    videos = [_video(number) for number in range(52)]
    videos[0] = _video(0, date=AFTER.isoformat(), duration=180)
    videos[1] = _video(1, date=BEFORE.isoformat())
    _, batches = _mock_api(monkeypatch, {CHANNEL_A: [[_item(video) for video in videos[:50]], [_item(video) for video in videos[50:]]]}, {video.video_id: video for video in videos})
    result = discover_collection_plan(channel_inputs=[CHANNEL_A], youtube_api_key="test", published_after=AFTER, published_before=BEFORE)
    assert [len(batch) for batch in batches] == [50, 1]
    assert result["selected_count"] == 51
    selected_ids = {item["video_id"] for item in result["items"]}
    assert videos[0].video_id in selected_ids
    assert videos[1].video_id not in selected_ids
    assert result["window_bounds"] == "[published_after, published_before)"


def test_duplicate_channels_do_not_repeat_discovery(monkeypatch):
    video = _video(1)
    calls, _ = _mock_api(monkeypatch, {CHANNEL_A: [[_item(video)]]}, {video.video_id: video})
    result = discover_collection_plan(channel_inputs=["@alpha", CHANNEL_A], youtube_api_key="test", published_after=AFTER, published_before=BEFORE)
    assert len(result["channels"]) == result["selected_count"] == 1
    assert len([call for call in calls if call["endpoint"].endswith("/playlistItems")]) == 1


def test_metadata_unknown_playlist_date_and_wrong_channel_are_handled(monkeypatch):
    wrong_channel = _video(1, channel=CHANNEL_B)
    valid = _video(2)
    invalid_date = _video(3, date="not-a-date")
    future = _video(4, date="2027-01-01T00:00:00Z")
    items = [_item(video) for video in [wrong_channel, valid, invalid_date, future]]
    for item in items:
        item["contentDetails"].pop("videoPublishedAt")
    _mock_api(monkeypatch, {CHANNEL_A: [items]}, {video.video_id: video for video in [wrong_channel, valid, invalid_date, future]})
    result = discover_collection_plan(channel_inputs=[CHANNEL_A], youtube_api_key="test", published_after=AFTER, published_before=BEFORE)
    assert [item["video_id"] for item in result["items"]] == [valid.video_id]
    skipped = result["channel_discovery"][0]["skipped_counts"]
    assert skipped["wrong_channel"] == skipped["unknown_publish_date"] == skipped["outside_window"] == 1
