from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import zip_longest
from typing import Any
from urllib.parse import unquote, urlparse

from learn_from.youtube import (
    YouTubeApiError,
    YouTubeChannelMetadata,
    _request_youtube_json,
    _resolve_thumbnail_url,
    _to_int,
    extract_video_id,
    fetch_videos_metadata,
)


_CHANNEL_ID = re.compile(r"^UC[A-Za-z0-9_-]{22}$")
_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com"}


def _channel_selector(value: str) -> tuple[str, str]:
    candidate = value.strip()
    if not candidate:
        raise ValueError("A channel ID, handle, or YouTube channel URL is required.")
    if candidate.startswith(("youtube.com/", "www.youtube.com/", "m.youtube.com/")):
        candidate = f"https://{candidate}"
    if "://" in candidate:
        parsed = urlparse(candidate)
        if parsed.scheme not in {"https", "http"} or parsed.hostname not in _YOUTUBE_HOSTS:
            raise ValueError("Channel URLs must use youtube.com.")
        parts = [unquote(part) for part in parsed.path.split("/") if part]
        if len(parts) >= 2 and parts[0] == "channel" and _CHANNEL_ID.fullmatch(parts[1]):
            return "id", parts[1]
        if parts and parts[0].startswith("@"):
            candidate = parts[0]
        else:
            raise ValueError("Use a YouTube @handle or /channel/UC... URL.")
    if _CHANNEL_ID.fullmatch(candidate):
        return "id", candidate
    handle = candidate.removeprefix("@")
    if not handle or any(character.isspace() or character in "/?#\\:" for character in handle):
        raise ValueError("Invalid YouTube channel handle.")
    return "forHandle", handle


def _resolve_channel(value: str, *, api_key: str) -> tuple[YouTubeChannelMetadata, str]:
    selector, identifier = _channel_selector(value)
    payload = _request_youtube_json(
        endpoint="https://www.googleapis.com/youtube/v3/channels",
        params={"part": "snippet,statistics,contentDetails", selector: identifier, "key": api_key},
        timeout_seconds=20.0,
    )
    items = payload.get("items") or []
    if len(items) != 1:
        raise YouTubeApiError(f"Could not resolve exactly one channel for {value!r}.")
    item = items[0]
    channel_id = str(item.get("id") or "")
    uploads = item.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
    if not _CHANNEL_ID.fullmatch(channel_id) or not uploads:
        raise YouTubeApiError(f"Channel {value!r} has no usable uploads playlist.")
    snippet = item.get("snippet") or {}
    statistics = item.get("statistics") or {}
    channel = YouTubeChannelMetadata(
        channel_id=channel_id,
        channel_name=str(snippet.get("title") or channel_id),
        channel_url=f"https://www.youtube.com/channel/{channel_id}",
        description=str(snippet.get("description") or ""),
        thumbnail_url=_resolve_thumbnail_url(snippet.get("thumbnails")),
        subscriber_count=_to_int(statistics.get("subscriberCount")),
    )
    return channel, str(uploads)


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _parse_date(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return _as_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        return None


def _channel_candidates(
    *,
    channel: YouTubeChannelMetadata,
    uploads_playlist: str,
    api_key: str,
    published_after: datetime,
    published_before: datetime,
    min_duration_seconds: int,
    max_pages: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    skipped = dict.fromkeys(
        ("outside_window", "below_min_duration", "unknown_duration", "missing_metadata",
         "invalid_video_id", "duplicate", "wrong_channel", "unknown_publish_date"),
        0,
    )
    candidate_ids: list[str] = []
    seen: set[str] = set()
    seen_tokens: set[str] = set()
    token: str | None = None
    pages_scanned = 0
    complete = False
    warning = None
    for _ in range(max_pages):
        params: dict[str, Any] = {
            "part": "contentDetails", "playlistId": uploads_playlist,
            "maxResults": 50, "key": api_key,
        }
        if token:
            params["pageToken"] = token
        payload = _request_youtube_json(
            endpoint="https://www.googleapis.com/youtube/v3/playlistItems",
            params=params,
            timeout_seconds=20.0,
        )
        pages_scanned += 1
        for item in payload.get("items") or []:
            details = item.get("contentDetails") or {}
            try:
                video_id = extract_video_id(str(details.get("videoId") or ""))
            except ValueError:
                skipped["invalid_video_id"] += 1
                continue
            if video_id in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(video_id)
            # This is the video's publication date, not the playlist insertion date.
            published = _parse_date(details.get("videoPublishedAt"))
            if published is not None and not published_after <= published < published_before:
                skipped["outside_window"] += 1
                continue
            candidate_ids.append(video_id)
        next_token = payload.get("nextPageToken")
        if not next_token:
            complete = True
            break
        token = str(next_token)
        if token in seen_tokens:
            warning = "YouTube returned a repeated pagination token; discovery is incomplete."
            break
        seen_tokens.add(token)
    if not complete and warning is None:
        warning = "The configured page cap was reached; discovery is incomplete."

    selected: list[dict[str, Any]] = []
    for start in range(0, len(candidate_ids), 50):
        batch = candidate_ids[start : start + 50]
        videos = fetch_videos_metadata(video_ids=batch, api_key=api_key)
        by_id = {video.video_id: video for video in videos if video.video_id in batch}
        for video_id in batch:
            video = by_id.get(video_id)
            if video is None:
                skipped["missing_metadata"] += 1
                continue
            if video.channel_id != channel.channel_id:
                skipped["wrong_channel"] += 1
                continue
            published = _parse_date(video.publish_date)
            if published is None:
                skipped["unknown_publish_date"] += 1
                continue
            if not published_after <= published < published_before:
                skipped["outside_window"] += 1
                continue
            if video.duration_seconds is None:
                skipped["unknown_duration"] += 1
                continue
            if video.duration_seconds < min_duration_seconds:
                skipped["below_min_duration"] += 1
                continue
            selected.append({
                "video_id": video_id,
                "channel_id": channel.channel_id,
                "title": video.title,
                "url": video.url,
                "published_at": published.isoformat(),
                "duration_seconds": video.duration_seconds,
            })
    selected.sort(key=lambda item: (item["published_at"], item["video_id"]), reverse=True)
    return selected, {
        "channel_id": channel.channel_id,
        "channel_name": channel.channel_name,
        "selected_count": len(selected),
        "pages_scanned": pages_scanned,
        "unique_video_ids_seen": len(seen),
        "skipped_counts": skipped,
        "discovery_complete": complete,
        "discovery_warning": warning,
    }


def discover_collection_plan(
    *,
    channel_inputs: Sequence[str],
    youtube_api_key: str,
    published_after: datetime,
    published_before: datetime | None = None,
    min_duration_seconds: int = 180,
    max_pages_per_channel: int = 100,
) -> dict[str, Any]:
    """List uploads in [published_after, published_before), without acquiring transcripts.

    The page cap bounds discovery calls. Only playlist exhaustion establishes complete
    enumeration; dates on one page are not assumed to prove the age of later pages.
    Duration is a selection rule, not a guarantee that an upload is a full episode.
    """
    if not youtube_api_key:
        raise ValueError("YOUTUBE_API_KEY is required for collection discovery.")
    if not channel_inputs or isinstance(channel_inputs, (str, bytes)):
        raise ValueError("Provide at least one channel in channel_inputs.")
    if isinstance(min_duration_seconds, bool) or min_duration_seconds < 0:
        raise ValueError("min_duration_seconds must be nonnegative.")
    if isinstance(max_pages_per_channel, bool) or max_pages_per_channel <= 0:
        raise ValueError("max_pages_per_channel must be greater than zero.")
    after = _as_utc(published_after)
    before = _as_utc(published_before or datetime.now(UTC))
    if after > before:
        raise ValueError("published_after cannot be later than published_before.")
    channels: list[dict[str, Any]] = []
    per_channel_items: list[list[dict[str, Any]]] = []
    discovery: list[dict[str, Any]] = []
    seen_channels: set[str] = set()
    for channel_input in channel_inputs:
        channel, uploads = _resolve_channel(channel_input, api_key=youtube_api_key)
        if channel.channel_id in seen_channels:
            continue
        seen_channels.add(channel.channel_id)
        items, status = _channel_candidates(
            channel=channel,
            uploads_playlist=uploads,
            api_key=youtube_api_key,
            published_after=after,
            published_before=before,
            min_duration_seconds=min_duration_seconds,
            max_pages=max_pages_per_channel,
        )
        channels.append(channel.to_dict())
        per_channel_items.append(items)
        discovery.append(status)
    interleaved = [item for row in zip_longest(*per_channel_items) for item in row if item is not None]
    complete = all(status["discovery_complete"] for status in discovery)
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "published_after": after.isoformat(),
        "published_before": before.isoformat(),
        "window_bounds": "[published_after, published_before)",
        "min_duration_seconds": min_duration_seconds,
        "max_pages_per_channel": max_pages_per_channel,
        "selection_order": "round_robin_channels_newest_first",
        "channels": channels,
        "items": interleaved,
        "selected_count": len(interleaved),
        "channel_discovery": discovery,
        "discovery_complete": complete,
        "discovery_warning": None if complete else "Some channels were not fully enumerated; see channel_discovery.",
    }
