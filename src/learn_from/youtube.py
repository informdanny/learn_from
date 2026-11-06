from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen


_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_ISO8601_DURATION_RE = re.compile(
    r"^P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)


class YouTubeApiError(RuntimeError):
    """Raised when the YouTube Data API request fails."""


class VideoNotFoundError(YouTubeApiError):
    """Raised when the requested video does not exist or is not accessible."""


@dataclass(frozen=True)
class YouTubeVideoMetadata:
    video_id: str
    title: str
    description: str
    channel_id: str
    channel_name: str
    url: str
    publish_date: str
    duration_seconds: int | None
    views: int | None
    likes: int | None
    comments: int | None
    thumbnail_url: str | None
    source_id: str = "youtube"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class YouTubeChannelMetadata:
    channel_id: str
    channel_name: str
    channel_url: str
    description: str
    thumbnail_url: str | None
    subscriber_count: int | None
    source_id: str = "youtube"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_video_id(value: str) -> str:
    """Extract a YouTube video id from a URL or return the id when already provided."""
    candidate = value.strip()
    if _VIDEO_ID_RE.fullmatch(candidate):
        return candidate

    parsed = urlparse(candidate)
    if parsed.netloc in {"youtu.be", "www.youtu.be"}:
        maybe_id = parsed.path.strip("/").split("/")[0]
        if _VIDEO_ID_RE.fullmatch(maybe_id):
            return maybe_id

    if "youtube.com" in parsed.netloc:
        if parsed.path == "/watch":
            query = parse_qs(parsed.query)
            maybe_id = query.get("v", [None])[0]
            if maybe_id and _VIDEO_ID_RE.fullmatch(maybe_id):
                return maybe_id

        for prefix in ("/shorts/", "/embed/", "/live/"):
            if parsed.path.startswith(prefix):
                maybe_id = parsed.path[len(prefix) :].split("/")[0]
                if _VIDEO_ID_RE.fullmatch(maybe_id):
                    return maybe_id

    raise ValueError(f"Unable to parse YouTube video id from input: {value}")


def parse_iso8601_duration_to_seconds(duration: str) -> int:
    """
    Convert an ISO-8601 duration string (YouTube contentDetails.duration) into seconds.
    """
    match = _ISO8601_DURATION_RE.fullmatch(duration)
    if not match:
        raise ValueError(f"Unsupported ISO-8601 duration value: {duration}")

    days = int(match.group("days") or 0)
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)
    return (days * 24 * 3600) + (hours * 3600) + (minutes * 60) + seconds


def fetch_video_metadata(
    *,
    video_id_or_url: str,
    api_key: str,
    timeout_seconds: float = 20.0,
) -> YouTubeVideoMetadata:
    """
    Fetch core video metadata from the YouTube Data API for one video id/url.
    """
    video_id = extract_video_id(video_id_or_url)
    payload = _request_videos_api([video_id], api_key=api_key, timeout_seconds=timeout_seconds)
    items = payload.get("items", [])
    if not items:
        raise VideoNotFoundError(f"No video found for id '{video_id}'.")

    return _parse_video_item(items[0])


def fetch_channel_metadata(
    *,
    channel_id: str,
    api_key: str,
    timeout_seconds: float = 20.0,
) -> YouTubeChannelMetadata:
    """Fetch basic channel metadata used by channel-backfill ingestion."""
    channel = channel_id.strip()
    if not channel:
        raise ValueError("Channel ID is required.")

    payload = _request_youtube_json(
        endpoint="https://www.googleapis.com/youtube/v3/channels",
        params={
            "part": "snippet,statistics",
            "id": channel,
            "key": api_key,
        },
        timeout_seconds=timeout_seconds,
    )
    items = payload.get("items", [])
    if not items:
        raise YouTubeApiError(f"No channel found for id '{channel}'.")
    item = items[0]
    snippet = item.get("snippet", {})
    statistics = item.get("statistics", {})

    return YouTubeChannelMetadata(
        channel_id=item.get("id", channel),
        channel_name=snippet.get("title", channel),
        channel_url=f"https://www.youtube.com/channel/{item.get('id', channel)}",
        description=snippet.get("description", ""),
        thumbnail_url=_resolve_thumbnail_url(snippet.get("thumbnails", {})),
        subscriber_count=_to_int(statistics.get("subscriberCount")),
    )


def list_channel_video_ids(
    *,
    channel_id: str,
    api_key: str,
    max_results: int = 30,
    timeout_seconds: float = 20.0,
) -> list[str]:
    """
    List recent YouTube video IDs for one channel, newest-first.
    """
    channel = channel_id.strip()
    if not channel:
        raise ValueError("Channel ID is required.")
    if max_results <= 0:
        raise ValueError("max_results must be greater than 0.")

    collected: list[str] = []
    seen: set[str] = set()
    page_token: str | None = None

    while len(collected) < max_results:
        page_size = min(50, max_results - len(collected))
        params: dict[str, Any] = {
            "part": "snippet",
            "channelId": channel,
            "maxResults": page_size,
            "order": "date",
            "type": "video",
            "key": api_key,
        }
        if page_token:
            params["pageToken"] = page_token

        payload = _request_youtube_json(
            endpoint="https://www.googleapis.com/youtube/v3/search",
            params=params,
            timeout_seconds=timeout_seconds,
        )
        items = payload.get("items", [])
        if not items:
            break

        for item in items:
            maybe_id = ""
            item_id = item.get("id")
            if isinstance(item_id, dict):
                maybe_id = str(item_id.get("videoId") or "")
            elif isinstance(item_id, str):
                maybe_id = item_id

            if not maybe_id or not _VIDEO_ID_RE.fullmatch(maybe_id) or maybe_id in seen:
                continue
            seen.add(maybe_id)
            collected.append(maybe_id)
            if len(collected) >= max_results:
                break

        page_token = payload.get("nextPageToken")
        if not page_token:
            break

    return collected


def fetch_videos_metadata(
    *,
    video_ids: Sequence[str],
    api_key: str,
    timeout_seconds: float = 20.0,
) -> list[YouTubeVideoMetadata]:
    """
    Fetch metadata for multiple video IDs (up to 50 per request).
    """
    normalized = [extract_video_id(v) for v in video_ids]
    if not normalized:
        return []

    payload = _request_videos_api(normalized, api_key=api_key, timeout_seconds=timeout_seconds)
    items = payload.get("items", [])
    return [_parse_video_item(item) for item in items]


def _request_videos_api(
    video_ids: Sequence[str], *, api_key: str, timeout_seconds: float
) -> dict[str, Any]:
    return _request_youtube_json(
        endpoint="https://www.googleapis.com/youtube/v3/videos",
        params={
            "part": "snippet,contentDetails,statistics",
            "id": ",".join(video_ids),
            "key": api_key,
        },
        timeout_seconds=timeout_seconds,
    )


def _parse_video_item(item: dict[str, Any]) -> YouTubeVideoMetadata:
    snippet = item.get("snippet", {})
    statistics = item.get("statistics", {})
    content_details = item.get("contentDetails", {})

    thumbnail_url = _resolve_thumbnail_url(snippet.get("thumbnails", {}))

    duration_value = content_details.get("duration")
    duration_seconds = (
        parse_iso8601_duration_to_seconds(duration_value) if isinstance(duration_value, str) else None
    )

    video_id = item.get("id") or ""
    return YouTubeVideoMetadata(
        video_id=video_id,
        title=snippet.get("title", ""),
        description=snippet.get("description", ""),
        channel_id=snippet.get("channelId", ""),
        channel_name=snippet.get("channelTitle", ""),
        url=f"https://www.youtube.com/watch?v={video_id}" if video_id else "",
        publish_date=snippet.get("publishedAt", ""),
        duration_seconds=duration_seconds,
        views=_to_int(statistics.get("viewCount")),
        likes=_to_int(statistics.get("likeCount")),
        comments=_to_int(statistics.get("commentCount")),
        thumbnail_url=thumbnail_url,
    )


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _request_youtube_json(
    *,
    endpoint: str,
    params: Mapping[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    query = urlencode(params)
    request = Request(
        url=f"{endpoint}?{query}",
        headers={"Accept": "application/json", "User-Agent": "learn-from/0.1.0"},
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
        raise YouTubeApiError(f"YouTube API HTTP error ({exc.code}): {details}") from exc
    except URLError as exc:
        raise YouTubeApiError(f"YouTube API connection error: {exc}") from exc

    try:
        parsed: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise YouTubeApiError("YouTube API returned invalid JSON.") from exc

    if "error" in parsed:
        raise YouTubeApiError(f"YouTube API error payload: {parsed['error']}")
    return parsed


def _resolve_thumbnail_url(thumbnails: Any) -> str | None:
    if not isinstance(thumbnails, dict):
        return None
    for quality in ("maxres", "high", "medium", "default"):
        item = thumbnails.get(quality)
        if isinstance(item, Mapping) and item.get("url"):
            return str(item["url"])
    return None
