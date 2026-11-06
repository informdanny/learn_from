from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from importlib import import_module
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from learn_from.youtube import extract_video_id


class TranscriptRetrievalError(RuntimeError):
    """Raised when transcript retrieval fails for a retriable/unknown reason."""


class TranscriptNotAvailableError(TranscriptRetrievalError):
    """Raised when a transcript is not available for the target video."""


@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    start_seconds: float
    duration_seconds: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class YouTubeTranscript:
    video_id: str
    language_code: str | None
    is_generated: bool | None
    segments: list[TranscriptSegment]

    @property
    def full_text(self) -> str:
        return "\n".join(segment.text for segment in self.segments if segment.text.strip())

    @property
    def timestamped_segments(self) -> list[dict[str, Any]]:
        return [segment.to_dict() for segment in self.segments]

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "language_code": self.language_code,
            "is_generated": self.is_generated,
            "full_text": self.full_text,
            "timestamped_segments": self.timestamped_segments,
        }


def fetch_transcript(
    *,
    video_id_or_url: str,
    languages: Sequence[str] = ("en",),
    preserve_formatting: bool = False,
    transcript_io_api_key: str | None = None,
    transcript_io_timeout_seconds: float = 30.0,
    proxy_http_url: str | None = None,
    proxy_https_url: str | None = None,
) -> YouTubeTranscript:
    """
    Fetch transcript segments for a YouTube video and normalize output for storage.
    """
    video_id = extract_video_id(video_id_or_url)
    if transcript_io_api_key:
        entries, language_code = _fetch_from_youtube_transcript_io(
            video_id=video_id,
            api_key=transcript_io_api_key,
            languages=languages,
            timeout_seconds=transcript_io_timeout_seconds,
        )
        is_generated: bool | None = None
    else:
        entries, language_code, is_generated = _fetch_raw_transcript(
            video_id=video_id,
            languages=languages,
            preserve_formatting=preserve_formatting,
            proxy_http_url=proxy_http_url,
            proxy_https_url=proxy_https_url,
        )

    segments: list[TranscriptSegment] = []
    for entry in entries:
        text_value, start_value, duration_value = _coerce_entry_fields(entry)
        segments.append(
            TranscriptSegment(
                text=text_value.strip(),
                start_seconds=float(start_value),
                duration_seconds=_to_float_or_none(duration_value),
            )
        )

    return YouTubeTranscript(
        video_id=video_id,
        language_code=language_code,
        is_generated=is_generated,
        segments=segments,
    )


def format_transcript_with_timestamps(transcript: YouTubeTranscript) -> str:
    """
    Render transcript segments as newline-delimited `[MM:SS] text` lines.
    """
    lines: list[str] = []
    for segment in transcript.segments:
        if not segment.text:
            continue
        lines.append(f"[{format_timestamp(segment.start_seconds)}] {segment.text}")
    return "\n".join(lines)


def format_timestamp(total_seconds: float) -> str:
    """
    Convert seconds to MM:SS or HH:MM:SS notation.
    """
    total = int(total_seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _fetch_raw_transcript(
    *,
    video_id: str,
    languages: Sequence[str],
    preserve_formatting: bool,
    proxy_http_url: str | None = None,
    proxy_https_url: str | None = None,
) -> tuple[list[dict[str, Any]], str | None, bool | None]:
    """
    Delegate transcript fetch to youtube-transcript-api while keeping imports lazy.
    """
    try:
        module = import_module("youtube_transcript_api")
    except ModuleNotFoundError as exc:
        raise TranscriptRetrievalError(
            "youtube-transcript-api is not installed. Run `poetry install` first."
        ) from exc

    api_cls = getattr(module, "YouTubeTranscriptApi")
    ytt_api = _build_api_client(
        api_cls=api_cls,
        proxy_http_url=proxy_http_url,
        proxy_https_url=proxy_https_url,
    )

    try:
        # Preferred API for youtube-transcript-api>=1.x
        fetched = ytt_api.fetch(
            video_id,
            languages=list(languages),
            preserve_formatting=preserve_formatting,
        )

        if hasattr(fetched, "to_raw_data"):
            entries = fetched.to_raw_data()
        else:
            entries = list(fetched)

        language_code = getattr(fetched, "language_code", None)
        is_generated = getattr(fetched, "is_generated", None)
        return list(entries), language_code, is_generated
    except TypeError:
        # Backward-compatible fallback for older pre-1.x API shapes.
        transcript_list = api_cls.list_transcripts(video_id)
        transcript_obj = transcript_list.find_transcript(list(languages))
        try:
            entries = transcript_obj.fetch(preserve_formatting=preserve_formatting)
        except TypeError:
            entries = transcript_obj.fetch()
        return list(entries), getattr(transcript_obj, "language_code", None), getattr(
            transcript_obj, "is_generated", None
        )
    except Exception as exc:
        name = exc.__class__.__name__
        if name in {
            "TranscriptsDisabled",
            "NoTranscriptFound",
            "VideoUnavailable",
            "NoTranscriptAvailable",
            "CouldNotRetrieveTranscript",
            "YouTubeRequestFailed",
            "IpBlocked",
            "RequestBlocked",
        }:
            raise TranscriptNotAvailableError(f"Transcript unavailable for video '{video_id}': {exc}") from exc
        raise TranscriptRetrievalError(f"Transcript retrieval failed for '{video_id}': {exc}") from exc


def _fetch_from_youtube_transcript_io(
    *,
    video_id: str,
    api_key: str,
    languages: Sequence[str],
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], str | None]:
    request = Request(
        url="https://www.youtube-transcript.io/api/transcripts",
        data=json.dumps({"ids": [video_id]}).encode("utf-8"),
        headers={
            "Authorization": f"Basic {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "learn-from/0.1.0",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
        if exc.code == 429:
            raise TranscriptRetrievalError(
                "youtube-transcript.io rate limit exceeded (HTTP 429). Retry after a short delay."
            ) from exc
        if exc.code in {401, 403}:
            raise TranscriptRetrievalError(
                "youtube-transcript.io authentication failed. Verify YOUTUBE_TRANSCRIPT_IO_API_KEY."
            ) from exc
        raise TranscriptRetrievalError(
            f"youtube-transcript.io request failed (HTTP {exc.code}): {details}"
        ) from exc
    except URLError as exc:
        raise TranscriptRetrievalError(f"youtube-transcript.io connection error: {exc}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TranscriptRetrievalError("youtube-transcript.io returned invalid JSON.") from exc

    if isinstance(payload, dict) and payload.get("error"):
        raise TranscriptRetrievalError(f"youtube-transcript.io error: {payload['error']}")

    if not isinstance(payload, list) or not payload:
        raise TranscriptNotAvailableError(
            f"No transcript payload returned by youtube-transcript.io for '{video_id}'."
        )

    chosen = None
    for item in payload:
        if isinstance(item, dict) and item.get("id") == video_id:
            chosen = item
            break
    if chosen is None:
        chosen = payload[0]

    tracks = chosen.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        text_value = str(chosen.get("text", "")).strip()
        if text_value:
            return [{"text": text_value, "start": 0, "dur": None}], None
        raise TranscriptNotAvailableError(
            f"No transcript track returned by youtube-transcript.io for '{video_id}'."
        )

    selected_track = _select_track(tracks=tracks, languages=languages)
    if selected_track is None:
        raise TranscriptNotAvailableError(
            f"No matching transcript track returned by youtube-transcript.io for '{video_id}'."
        )

    transcript_entries = selected_track.get("transcript")
    if not isinstance(transcript_entries, list) or not transcript_entries:
        raise TranscriptNotAvailableError(
            f"Transcript track was empty for '{video_id}' on youtube-transcript.io."
        )

    return transcript_entries, selected_track.get("language")


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _coerce_entry_fields(entry: Any) -> tuple[str, float, Any]:
    if isinstance(entry, dict):
        text_value = str(entry.get("text", ""))
        start_value = _to_float_or_none(entry.get("start", entry.get("start_seconds", 0.0))) or 0.0
        duration_value = entry.get("duration", entry.get("dur"))
        return text_value, start_value, duration_value

    text_value = str(getattr(entry, "text", ""))
    start_value = _to_float_or_none(getattr(entry, "start", 0.0)) or 0.0
    duration_value = getattr(entry, "duration", None)
    return text_value, start_value, duration_value


def _build_api_client(
    *,
    api_cls: Any,
    proxy_http_url: str | None,
    proxy_https_url: str | None,
) -> Any:
    if not proxy_http_url and not proxy_https_url:
        return api_cls()

    try:
        proxies_module = import_module("youtube_transcript_api.proxies")
        generic_proxy_cls = getattr(proxies_module, "GenericProxyConfig")
        proxy_config = generic_proxy_cls(
            http_url=proxy_http_url,
            https_url=proxy_https_url,
        )
        return api_cls(proxy_config=proxy_config)
    except Exception as exc:
        raise TranscriptRetrievalError(
            "Failed to initialize transcript proxy config. "
            "Verify TRANSCRIPT_PROXY_HTTP_URL / TRANSCRIPT_PROXY_HTTPS_URL."
        ) from exc


def _select_track(
    *, tracks: list[dict[str, Any]], languages: Sequence[str]
) -> dict[str, Any] | None:
    requested = [language.strip().lower() for language in languages if language.strip()]
    if not requested:
        requested = ["en"]

    for language in requested:
        for track in tracks:
            track_language = str(track.get("language", "")).strip().lower()
            if not track_language:
                continue
            if (
                track_language == language
                or track_language.startswith(language)
                or language.startswith(track_language)
            ):
                return track

    return tracks[0] if tracks else None
