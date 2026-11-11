"""OpenAI-managed transcript indexing and retrieval, with durable upload state.

Neon remains authoritative. This module does not chunk, embed, rerank, delete,
or fetch new transcripts. Provider mutations are never automatically retried.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4

from sqlalchemy import MetaData, Table, select, update
from sqlalchemy.engine import Engine

from learn_from.collections import _collection_lock


API_BASE = "https://api.openai.com/v1"
MAX_SOURCE_BYTES = 8 * 1024 * 1024
DEFAULT_INDEX_BYTES = 1024 ** 3
_FILE_STATES = {"in_progress", "completed", "cancelled", "failed"}
_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")


class HostedRetrievalError(RuntimeError):
    """A hosted operation failed or needs reconciliation before continuing."""


class HostedAPIError(HostedRetrievalError):
    def __init__(self, message: str, *, status_code: int | None = None, ambiguous: bool = True):
        super().__init__(message)
        self.status_code = status_code
        self.ambiguous = ambiguous


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _provider_id(value: Any, prefix: str) -> str:
    if not isinstance(value, str) or not value.startswith(prefix) or not _ID_PATTERN.fullmatch(value):
        raise HostedRetrievalError("OpenAI returned an invalid resource identifier.")
    return value


def _request(method: str, path: str, api_key: str, *, payload: dict[str, Any] | None = None,
             body: bytes | None = None, content_type: str = "application/json",
             raw: bool = False, timeout_seconds: float = 60.0) -> Any:
    if not api_key or not isinstance(api_key, str):
        raise ValueError("OPENAI_API_KEY is required.")
    if not path.startswith("/") or path.startswith("//"):
        raise ValueError("Invalid OpenAI API path.")
    data = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else body
    request = Request(API_BASE + path, data=data, method=method, headers={
        "Authorization": f"Bearer {api_key}", "Content-Type": content_type,
        "OpenAI-Beta": "assistants=v2",
    })
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            value = response.read(16 * 1024 * 1024 + 1)
        if len(value) > 16 * 1024 * 1024:
            raise ValueError("Response too large")
        result = value if raw else json.loads(value)
        if not raw and not isinstance(result, dict):
            raise ValueError("Invalid JSON object")
        return result
    except HTTPError as exc:
        # Error bodies can contain source text. Do not expose them or the key.
        raise HostedAPIError(f"OpenAI returned HTTP {exc.code}; no automatic retry was made.",
                             status_code=exc.code, ambiguous=exc.code >= 500 or exc.code in (408, 409)) from None
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        raise HostedAPIError(f"OpenAI response unavailable ({type(exc).__name__}); outcome may be unknown and was not retried.") from None


def _upload(api_key: str, filename: str, document: bytes) -> dict[str, Any]:
    boundary = "learnfrom" + uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nassistants\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
            "Content-Type: text/plain; charset=utf-8\r\n\r\n").encode()
    body += document + f"\r\n--{boundary}--\r\n".encode()
    return _request("POST", "/files", api_key, body=body, content_type=f"multipart/form-data; boundary={boundary}")


def _iso(value: Any) -> str:
    return value.isoformat() if isinstance(value, (date, datetime)) else str(value or "not stated")


def source_document(document: Mapping[str, Any]) -> tuple[bytes, dict[str, Any]]:
    """Render a complete original source and its deterministic indexing identity."""
    full_text = document.get("full_text")
    if not isinstance(full_text, str) or not full_text.strip():
        raise HostedRetrievalError("Saved transcript is empty.")
    video_id = document["media_id"]
    channel_id = document["channel_id"]
    if not isinstance(video_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise HostedRetrievalError("Saved source has an invalid YouTube video ID.")
    if not isinstance(channel_id, str) or not re.fullmatch(r"UC[A-Za-z0-9_-]{22}", channel_id):
        raise HostedRetrievalError("Saved source has an invalid channel ID.")
    segments = document.get("timestamped_segments")
    if isinstance(segments, str):
        try:
            segments = json.loads(segments)
        except json.JSONDecodeError:
            segments = None
    timed: list[str] = []
    words: list[str] = []
    last_start = -1.0
    valid_timing = isinstance(segments, list) and bool(segments)
    for segment in segments if isinstance(segments, list) else []:
        if not isinstance(segment, Mapping) or not isinstance(segment.get("text"), str):
            valid_timing = False
            break
        if not segment["text"].strip():
            continue
        try:
            start = float(segment.get("start_seconds"))
        except (TypeError, ValueError):
            valid_timing = False
            break
        if not math.isfinite(start) or start < 0 or start < last_start:
            valid_timing = False
            break
        last_start = start
        words.append(segment["text"])
        timed.append(f"[t={start:.3f}s] {segment['text']}")
    # Never silently discard full-text material not represented by segments.
    valid_timing = valid_timing and " ".join(" ".join(words).split()) == " ".join(full_text.split())
    transcript = "\n".join(timed) if valid_timing else full_text
    header = {
        "format_version": 1, "video_id": video_id, "content_id": int(document["content_id"]),
        "channel_id": channel_id, "channel_name": document.get("channel_name"),
        "title": document.get("title"), "published_at": _iso(document.get("publish_date")),
        "language": document.get("language") or "und",
        "source_url": f"https://www.youtube.com/watch?v={video_id}",
        "speaker_attribution": "Channel ownership does not identify the speaker; guests may express their own views.",
        "timing": "saved segment start times" if valid_timing else "unavailable or not aligned; do not infer timestamps",
    }
    rendered = (json.dumps(header, ensure_ascii=False, sort_keys=True, indent=2) + "\n\nORIGINAL TRANSCRIPT\n" + transcript + "\n").encode()
    if len(rendered) > MAX_SOURCE_BYTES:
        raise HostedRetrievalError("Transcript exceeds the supported 8 MiB source limit; it was not truncated or uploaded.")
    attrs = {"video_id": video_id, "channel_id": channel_id,
             "content_id": str(document["content_id"]), "language": str(header["language"])[:64],
             "source_hash": hashlib.sha256(rendered).hexdigest(),
             "published_at": header["published_at"][:64]}
    return rendered, attrs


class _IndexRepository:
    def __init__(self, engine: Engine):
        self.engine = engine
        metadata = MetaData()
        with engine.connect() as connection:
            self.jobs = Table("sync_jobs", metadata, autoload_with=connection)
            self.media = Table("media", metadata, autoload_with=connection)
            self.content = Table("doc_content", metadata, autoload_with=connection)
            self.channels = Table("channels", metadata, autoload_with=connection)

    def load(self, job_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(select(self.jobs.c.checkpoint_data).where(self.jobs.c.job_id == job_id)).first()
        if row is None or not isinstance(row[0], dict) or row[0].get("collection_version") != 1:
            raise HostedRetrievalError("Job does not contain a supported collection checkpoint.")
        return copy.deepcopy(row[0])

    def save(self, job_id: str, manifest: dict[str, Any]) -> None:
        # Merge only this subkey under a short row lock; provider I/O is outside
        # transactions. The collection advisory lock excludes acquisition workers.
        with self.engine.begin() as connection:
            row = connection.execute(select(self.jobs.c.checkpoint_data).where(self.jobs.c.job_id == job_id).with_for_update()).first()
            if row is None:
                raise HostedRetrievalError("Collection disappeared while indexing.")
            checkpoint = copy.deepcopy(row[0] or {})
            checkpoint["hosted_index"] = copy.deepcopy(manifest)
            connection.execute(update(self.jobs).where(self.jobs.c.job_id == job_id).values(checkpoint_data=checkpoint))

    def document(self, item: dict[str, Any]) -> dict[str, Any] | None:
        m, d, c = self.media, self.content, self.channels
        query = select(m.c.media_id, m.c.channel_id, m.c.title, m.c.publish_date, c.c.channel_name,
                       d.c.content_id, d.c.full_text, d.c.timestamped_segments, d.c.language).select_from(
            m.join(d, d.c.media_id == m.c.media_id).join(c, c.c.channel_id == m.c.channel_id)
        ).where(m.c.media_id == item["video_id"], m.c.channel_id == item["channel_id"],
                d.c.chunk_type == "complete")
        if item.get("content_id") is not None:
            query = query.where(d.c.content_id == int(item["content_id"]))
        query = query.order_by(d.c.content_id.desc()).limit(1)
        with self.engine.connect() as connection:
            row = connection.execute(query).mappings().first()
        return dict(row) if row else None


def get_collection_index(*, engine: Engine, job_id: str) -> dict[str, Any] | None:
    """Read the saved index manifest without any OpenAI request."""
    return _IndexRepository(engine).load(job_id).get("hosted_index")


def _recover_upload(api_key: str, filename: str, expected: bytes) -> str | None:
    """Reconcile an uncertain upload by unique filename AND complete byte hash.

    A bounded scan finding no match is not permission to upload again. Files
    listings can lag or contain older pages outside this bounded scan.
    """
    after = None
    matches: list[str] = []
    for _ in range(10):
        query: dict[str, Any] = {"purpose": "assistants", "limit": 100, "order": "desc"}
        if after:
            query["after"] = after
        page = _request("GET", "/files?" + urlencode(query), api_key)
        rows = page.get("data")
        if not isinstance(rows, list):
            raise HostedRetrievalError("OpenAI file listing was malformed; upload remains unresolved.")
        for row in rows:
            if isinstance(row, dict) and row.get("filename") == filename:
                matches.append(_provider_id(row.get("id"), "file"))
        if not page.get("has_more"):
            break
        after = page.get("last_id")
        if not isinstance(after, str):
            break
    if len(matches) != 1:
        return None
    body = _request("GET", f"/files/{matches[0]}/content", api_key, raw=True)
    if not isinstance(body, bytes) or hashlib.sha256(body).digest() != hashlib.sha256(expected).digest():
        raise HostedRetrievalError("Recovered upload does not match the saved source; reconciliation is required.")
    return matches[0]


def _store_usage(api_key: str, manifest: dict[str, Any]) -> None:
    store_id = _provider_id(manifest.get("vector_store_id"), "vs_")
    result = _request("GET", f"/vector_stores/{store_id}", api_key)
    if result.get("id") != store_id or result.get("status") not in ("completed", "in_progress", "expired"):
        raise HostedRetrievalError("OpenAI returned an invalid vector-store state.")
    usage = result.get("usage_bytes")
    if isinstance(usage, bool) or not isinstance(usage, int) or usage < 0:
        raise HostedRetrievalError("OpenAI storage usage is unavailable; new indexing is paused.")
    manifest.update(usage_bytes=usage, provider_status=result["status"], usage_checked_at=_now())
    if result["status"] == "expired":
        raise HostedRetrievalError("The hosted index has expired; sources are retained in Neon. No automatic replacement was created.")


def _poll_file(api_key: str, store_id: str, entry: dict[str, Any], attempts: int, interval: float) -> None:
    for attempt in range(attempts):
        try:
            result = _request("GET", f"/vector_stores/{store_id}/files/{entry['file_id']}", api_key)
        except HostedAPIError as exc:
            if exc.status_code == 404 and entry["status"] in ("attaching", "attachment_unknown"):
                entry["status"] = "attachment_unknown"
                return
            raise
        if result.get("id") != entry["file_id"] or result.get("vector_store_id") != store_id or result.get("status") not in _FILE_STATES:
            raise HostedRetrievalError("OpenAI returned an invalid file-index state.")
        entry["status"] = result["status"]
        usage = result.get("usage_bytes")
        entry["usage_bytes"] = usage if isinstance(usage, int) and not isinstance(usage, bool) and usage >= 0 else None
        entry["checked_at"] = _now()
        if entry["status"] != "in_progress":
            return
        if attempt + 1 < attempts:
            time.sleep(interval)


def sync_collection_index(*, engine: Engine, job_id: str, api_key: str,
                          max_files: int = 1000, max_index_bytes: int = DEFAULT_INDEX_BYTES,
                          poll_attempts: int = 30, poll_interval_seconds: float = 2.0,
                          on_progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Index saved acquired/reused items, at most ``max_files`` per invocation.

    ``max_index_bytes`` stops new files after measured usage reaches the limit;
    the next file's parsed size is unknown beforehand, so this is NOT a billing
    cap or proof of account-wide free storage. No expiry or deletion is applied.
    """
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required.")
    if not isinstance(job_id, str) or not job_id.strip() or len(job_id) > 255:
        raise ValueError("A valid collection job ID is required.")
    if isinstance(max_files, bool) or not isinstance(max_files, int) or not 1 <= max_files <= 10000:
        raise ValueError("max_files must be between 1 and 10,000.")
    if isinstance(max_index_bytes, bool) or not isinstance(max_index_bytes, int) or max_index_bytes <= 0:
        raise ValueError("max_index_bytes must be positive.")
    if (isinstance(poll_attempts, bool) or not isinstance(poll_attempts, int) or not 1 <= poll_attempts <= 30
            or isinstance(poll_interval_seconds, bool) or not isinstance(poll_interval_seconds, (int, float))
            or not math.isfinite(poll_interval_seconds) or not 0 <= poll_interval_seconds <= 2):
        raise ValueError("Invalid bounded polling settings.")
    repo = _IndexRepository(engine)
    with _collection_lock(engine, job_id) as check_lock:
        checkpoint = repo.load(job_id)
        items = [item for item in checkpoint.get("items", []) if item.get("status") in ("acquired", "reused")]
        manifest = checkpoint.get("hosted_index") or {"version": 1, "files": {}, "created_at": _now()}
        if manifest.get("version") != 1 or not isinstance(manifest.get("files"), dict):
            raise HostedRetrievalError("Unsupported hosted-index manifest.")
        manifest.update(eligible_count=len(items), max_files_per_run=max_files, max_index_bytes=max_index_bytes,
                        storage_scope="this vector store only; not account-wide free storage", expires_after=None)

        def save(status: str | None = None) -> None:
            check_lock()  # A dropped pinned connection also drops its lock.
            if status:
                manifest["status"] = status
            manifest["updated_at"] = _now()
            manifest["completed_count"] = sum(entry.get("status") == "completed" for entry in manifest["files"].values())
            repo.save(job_id, manifest)
            if on_progress:
                on_progress(copy.deepcopy(manifest))

        if not items:
            save("no_saved_transcripts")
            return copy.deepcopy(manifest)
        if not manifest.get("vector_store_id"):
            if manifest.get("store_state") in ("creating", "creation_unknown", "creation_failed"):
                save("reconciliation_required")
                raise HostedRetrievalError("A prior store-creation outcome is unresolved; no second store was created.")
            manifest["store_state"] = "creating"
            save("indexing")
            try:
                store = _request("POST", "/vector_stores", api_key, payload={
                    "name": f"Learn From collection {job_id}",
                    "metadata": {"collection_job_id": job_id[:512], "source": "learn_from_neon", "manifest_version": "1"},
                })
                manifest["vector_store_id"] = _provider_id(store.get("id"), "vs_")
                manifest["store_state"] = "created"
                save()
            except HostedAPIError as exc:
                manifest["store_state"] = "creation_unknown" if exc.ambiguous else "creation_failed"
                save("reconciliation_required")
                raise
        store_id = _provider_id(manifest["vector_store_id"], "vs_")
        handled = 0
        try:
            _store_usage(api_key, manifest)
            save("indexing")
            for item in items:
                video_id = item["video_id"]
                document = repo.document(item)
                if document is None:
                    manifest["halt_reason"] = "A completed collection item has no matching saved source."
                    save("source_unavailable")
                    break
                rendered, attrs = source_document(document)
                entry = manifest["files"].get(video_id)
                if entry and entry.get("source_hash") != attrs["source_hash"]:
                    manifest["halt_reason"] = "Saved source changed after indexing; reconcile the old index before replacement."
                    save("source_changed")
                    break
                if entry and entry.get("status") == "completed":
                    continue
                if handled >= max_files:
                    save("file_limit_reached")
                    break
                if manifest["usage_bytes"] >= max_index_bytes:
                    save("storage_limit_reached")
                    break
                handled += 1
                if entry is None:
                    filename = f"learnfrom-{hashlib.sha256(job_id.encode()).hexdigest()[:16]}-{video_id}-{attrs['source_hash']}.txt"
                    entry = {**attrs, "filename": filename, "source_bytes": len(rendered), "status": "pending"}
                    manifest["files"][video_id] = entry
                    save()
                if entry["status"] in ("uploading", "upload_unknown"):
                    recovered = _recover_upload(api_key, entry["filename"], rendered)
                    if recovered is None:
                        save("reconciliation_required")
                        break
                    entry.update(file_id=recovered, status="uploaded", recovered_at=_now())
                    save()
                if entry["status"] in ("upload_failed", "failed", "cancelled", "attachment_failed"):
                    save("file_failed")
                    break
                if entry["status"] == "pending":
                    entry["status"] = "uploading"
                    save()
                    try:
                        uploaded = _upload(api_key, entry["filename"], rendered)
                        entry.update(file_id=_provider_id(uploaded.get("id"), "file"), status="uploaded")
                        save()  # Persist provider ID BEFORE attachment.
                    except HostedAPIError as exc:
                        entry["status"] = "upload_unknown" if exc.ambiguous else "upload_failed"
                        save("reconciliation_required")
                        raise
                if entry["status"] == "uploaded":
                    entry["status"] = "attaching"
                    save()
                    try:
                        _request("POST", f"/vector_stores/{store_id}/files", api_key,
                                 payload={"file_id": entry["file_id"], "attributes": attrs})
                        entry["status"] = "in_progress"
                        save()
                    except HostedAPIError as exc:
                        entry["status"] = "attachment_unknown" if exc.ambiguous else "attachment_failed"
                        save("reconciliation_required")
                        raise
                _poll_file(api_key, store_id, entry, poll_attempts, poll_interval_seconds)
                _store_usage(api_key, manifest)
                save()
                if entry["status"] != "completed":
                    save("indexing_pending" if entry["status"] == "in_progress" else "reconciliation_required")
                    break  # Do not accumulate unknown storage from pending files.
                if manifest["usage_bytes"] >= max_index_bytes:
                    save("storage_limit_reached")
                    break
            else:
                manifest.pop("halt_reason", None)
                save("completed" if manifest["provider_status"] == "completed" else "indexing_pending")
        except HostedRetrievalError:
            save("reconciliation_required")
            raise
        return copy.deepcopy(manifest)


METADATA_FILTER_KEYS = frozenset({"industry", "business_model", "business_stage", "decision_area"})


def _metadata_filters(values: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    from .metadata_catalog import BUSINESS_FIELDS

    if values is None:
        return []
    if not isinstance(values, Mapping) or len(values) > len(METADATA_FILTER_KEYS):
        raise ValueError("Metadata filters must be a small field/value mapping.")
    filters = []
    for key, value in values.items():
        if (key not in METADATA_FILTER_KEYS or not isinstance(value, str)
                or not 0 < len(value) <= 512 or value != value.casefold()
                or not re.fullmatch(r"\w+", value, re.UNICODE)):
            raise ValueError("Choose a supported metadata field and a canonical lowercase value from the catalog.")
        definition = BUSINESS_FIELDS[key]
        if definition["policy"] == "fixed" and value not in definition["values"]:
            raise ValueError("Choose an accepted option for the fixed business_stage field.")
        filters.append({"type": "eq", "key": key, "value": value})
    return filters


def hosted_search(*, api_key: str, vector_store_id: str, query: str,
                  channel_ids: Sequence[str], max_results: int = 10,
                  metadata_filters: Mapping[str, str] | None = None,
                  video_ids: Sequence[str] | None = None,
                  rewrite_query: bool = False) -> dict[str, Any]:
    """Use OpenAI's managed retrieval with a mandatory hard channel filter."""
    store_id = _provider_id(vector_store_id, "vs_")
    if not isinstance(query, str) or not query.strip() or len(query) > 6000:
        raise ValueError("Query must contain 1 to 6,000 characters.")
    scope = list(dict.fromkeys(channel_ids))
    if not 1 <= len(scope) <= 10 or any(not isinstance(value, str) or not re.fullmatch(r"UC[A-Za-z0-9_-]{22}", value) for value in scope):
        raise ValueError("Select one to ten canonical YouTube channel IDs.")
    if isinstance(max_results, bool) or not isinstance(max_results, int) or not 1 <= max_results <= 50:
        raise ValueError("max_results must be between 1 and 50.")
    if not isinstance(rewrite_query, bool):
        raise ValueError("rewrite_query must be boolean.")
    facets = _metadata_filters(metadata_filters)
    selected_videos = None
    if video_ids is not None:
        if isinstance(video_ids, (str, bytes)):
            raise ValueError("video_ids must be a list of exact YouTube video IDs.")
        selected_videos = list(dict.fromkeys(video_ids))
        if len(selected_videos) > 1000 or any(not isinstance(v, str) or not re.fullmatch(r"[A-Za-z0-9_-]{11}", v) for v in selected_videos):
            raise ValueError("Select at most 1,000 exact YouTube video IDs.")
        if not selected_videos:
            return {"data": [], "search_performed": False, "reason": "No current indexed sources match the metadata selection."}
    channel_filter = {"type": "in", "key": "channel_id", "value": scope}
    constraints = [channel_filter, *facets]
    if selected_videos is not None:
        constraints.append({"type": "in", "key": "video_id", "value": selected_videos})
    payload = {"query": query.strip(), "filters": {"type": "and", "filters": constraints} if len(constraints) > 1 else channel_filter,
               "max_num_results": max_results}
    if rewrite_query:
        payload.update(rewrite_query=True, ranking_options={"ranker": "auto"})
    result = _request("POST", f"/vector_stores/{store_id}/search", api_key, payload=payload)
    rows = result.get("data")
    if not isinstance(rows, list) or len(rows) > max_results:
        raise HostedRetrievalError("OpenAI returned an invalid search page.")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("attributes"), dict) or row["attributes"].get("channel_id") not in scope:
            raise HostedRetrievalError("OpenAI returned an out-of-scope or unattributed search result.")
        if any(row["attributes"].get(facet["key"]) != facet["value"] for facet in facets):
            raise HostedRetrievalError("OpenAI returned a result outside the requested metadata filters.")
        if selected_videos is not None and row["attributes"].get("video_id") not in selected_videos:
            raise HostedRetrievalError("OpenAI returned a result outside the selected metadata source set.")
        _provider_id(row.get("file_id"), "file")
    return result


def _latest_index_metadata(engine: Engine, content_id: int) -> tuple[int, dict[str, Any]] | None:
    """Read explicit extraction output; legacy media defaults are not evidence of extraction."""
    with engine.connect() as connection:
        runs = Table("extraction_runs", MetaData(), autoload_with=connection)
        rows = connection.execute(select(runs.c.run_id, runs.c.resolved_output).where(
            runs.c.content_id == content_id, runs.c.status == "completed",
        ).order_by(runs.c.run_id.desc())).mappings()
        for row in rows:
            resolved = row["resolved_output"]
            metadata = resolved.get("index_metadata") if isinstance(resolved, dict) else None
            if isinstance(metadata, dict):
                return int(row["run_id"]), metadata
    return None


def sync_collection_metadata(*, engine: Engine, job_id: str, api_key: str,
                             video_ids: Sequence[str], max_files: int = 10) -> dict[str, Any]:
    """Refresh attributes on existing ready files, without reuploading or resuming acquisition.

    A GET reconciles an uncertain prior update before any further mutation. All
    source identity attributes are rebuilt and checked against the saved upload.
    The collection/index state is preserved, including unrelated stopped work.
    """
    selected = list(dict.fromkeys(video_ids))
    if not selected or len(selected) > 10 or any(not isinstance(v, str) or not re.fullmatch(r"[A-Za-z0-9_-]{11}", v) for v in selected):
        raise ValueError("Select one to ten exact video IDs for metadata synchronization.")
    if isinstance(max_files, bool) or not isinstance(max_files, int) or not 1 <= max_files <= 10:
        raise ValueError("Metadata sync max_files must be between one and ten.")
    repo = _IndexRepository(engine)
    outcomes = []
    with _collection_lock(engine, job_id) as check_lock:
        checkpoint = repo.load(job_id)
        manifest = checkpoint.get("hosted_index")
        if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
            raise HostedRetrievalError("No current hosted index exists for this collection.")
        store_id = _provider_id(manifest.get("vector_store_id"), "vs_")
        items = {item["video_id"]: item for item in checkpoint.get("items", [])}
        for video_id in selected[:max_files]:
            check_lock()
            entry = manifest["files"].get(video_id)
            item = items.get(video_id)
            if not item or not isinstance(entry, dict) or entry.get("status") != "completed":
                outcomes.append({"video_id": video_id, "status": "skipped", "reason": "Current index file is not ready."})
                continue
            document = repo.document(item)
            if document is None:
                raise HostedRetrievalError("Saved source is unavailable; metadata was not updated.")
            _, base = source_document(document)
            if entry.get("source_hash") != base["source_hash"] or str(entry.get("content_id")) != base["content_id"]:
                raise HostedRetrievalError("Source changed since indexing; metadata was not updated.")
            found = _latest_index_metadata(engine, int(document["content_id"]))
            if found is None:
                outcomes.append({"video_id": video_id, "status": "skipped", "reason": "No completed index metadata extraction."})
                continue
            run_id, extracted = found
            content_hash = hashlib.sha256(str(document["full_text"]).strip().encode()).hexdigest()
            if extracted.get("source_hash") != content_hash:
                raise HostedRetrievalError("Metadata describes a different source version; no update was made.")
            projected = extracted.get("openai_attributes")
            from learn_from.metadata_catalog import project_openai_metadata
            if not isinstance(projected, dict) or projected != project_openai_metadata(extracted):
                raise HostedRetrievalError("Metadata projection is missing.")
            _metadata_filters(projected)  # Same small scalar contract for writes and reads.
            attributes = {**base, **projected, "metadata_run_id": str(run_id),
                          "metadata_version": str(extracted.get("version") or "index-1.0")[:64]}
            if len(attributes) > 16:
                raise HostedRetrievalError("Metadata exceeds provider attribute limits.")
            digest = hashlib.sha256(json.dumps(attributes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            file_id = _provider_id(entry.get("file_id"), "file")
            path = f"/vector_stores/{store_id}/files/{file_id}"
            observed = _request("GET", path, api_key)
            if observed.get("id") != file_id or observed.get("vector_store_id") != store_id or observed.get("status") != "completed":
                raise HostedRetrievalError("Provider file is no longer ready; metadata was not updated.")
            already_matches = observed.get("attributes") == attributes
            state = {"status": "updating", "run_id": run_id, "attributes_hash": digest, "updated_at": _now()}
            entry["metadata"] = state
            repo.save(job_id, manifest)
            try:
                if not already_matches:
                    check_lock()
                    _request("POST", path, api_key, payload={"attributes": attributes})
                    # Attribute changes may become visible after the update response.
                    # Retry only reads; never repeat an uncertain mutation here.
                    for attempt in range(5):
                        check_lock()
                        observed = _request("GET", path, api_key)
                        if (observed.get("id") == file_id and observed.get("vector_store_id") == store_id
                                and observed.get("status") == "completed" and observed.get("attributes") == attributes):
                            break
                        if attempt < 4:
                            time.sleep(0.5)
                if observed.get("id") != file_id or observed.get("vector_store_id") != store_id or observed.get("status") != "completed" or observed.get("attributes") != attributes:
                    raise HostedRetrievalError("Provider metadata readback differs; reconciliation is required.")
                state.update(status="synced", updated_at=_now(), attributes=attributes)
                repo.save(job_id, manifest)
                outcomes.append({"video_id": video_id, "status": "unchanged" if already_matches else "synced", "run_id": run_id, "attributes": attributes})
            except HostedRetrievalError:
                state.update(status="reconciliation_required", updated_at=_now())
                repo.save(job_id, manifest)
                raise
    return {"job_id": job_id, "vector_store_id": store_id, "results": outcomes,
            "requested_count": len(selected), "deferred_count": max(0, len(selected) - max_files)}
