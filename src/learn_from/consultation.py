"""Bounded consultation using original passages from selected cloud-library channels.

Complete selected documents or externally retrieved evidence are supplied to a
bounded model call. No custom retrieval/ranking pipeline is required.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from importlib.resources import files
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine


MODEL = "gpt-6-luna"
RATE_DATE = "2026-11-06"
_INPUT_RATE = Decimal("0.10")
_OUTPUT_RATE = Decimal("0.50")
# No extended cache retention is requested. Reserve at the standard cache-write
# rate without a cache discount; double it when the byte-based token bound
# reaches the long-context tier. UTF-8 bytes plus overhead bound input tokens.
_RESERVATION_INPUT_RATE = Decimal("0.125")



class ConsultationError(RuntimeError):
    """Consultation cannot produce a validated, bounded result."""


class _Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str
    attribution: str
    evidence_ids: list[str]


class _Application(BaseModel):
    model_config = ConfigDict(extra="forbid")
    angle: str
    inference: str
    evidence_ids: list[str]


class _Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    framing: str
    published_views: list[_Claim]
    application_angles: list[_Application]
    counterarguments: list[_Application]
    questions_to_resolve: list[str]
    limitations: str


def _validate_scope(channel_ids: Sequence[str]) -> list[str]:
    scope = list(dict.fromkeys(channel_ids))
    if not scope or len(scope) > 10:
        raise ValueError("Select between one and ten channel IDs.")
    if any(not isinstance(value, str) or not re.fullmatch(r"UC[A-Za-z0-9_-]{22}", value) for value in scope):
        raise ValueError("Every channel must be a canonical YouTube channel ID.")
    return scope


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _normalize_segments(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    result = []
    for segment in value:
        if not isinstance(segment, Mapping):
            continue
        start = _number(segment.get("start_seconds"))
        content = segment.get("text")
        if start is None or not isinstance(content, str) or not content.strip():
            continue
        result.append({"text": content.strip(), "start_seconds": start, "duration_seconds": _number(segment.get("duration_seconds"))})
    return sorted(result, key=lambda item: item["start_seconds"])


def _load_full_documents(engine: Engine, scope: list[str], video_ids: list[str]) -> list[dict[str, Any]]:
    query = text("""
        SELECT m.media_id, m.title, m.url, m.channel_id, ch.channel_name,
               m.publish_date, dc.content_id, dc.full_text, dc.timestamped_segments
        FROM media m
        JOIN channels ch ON ch.channel_id = m.channel_id
        JOIN doc_content dc ON dc.media_id = m.media_id
        WHERE m.channel_id IN :channel_ids AND m.media_id IN :video_ids
          AND dc.chunk_type = 'complete'
          AND nullif(btrim(dc.full_text), '') IS NOT NULL
        ORDER BY dc.content_id DESC
    """).bindparams(bindparam("channel_ids", expanding=True), bindparam("video_ids", expanding=True))
    with engine.connect() as connection:
        return [dict(row) for row in connection.execute(query, {"channel_ids": scope, "video_ids": video_ids}).mappings()]


def read_consultation_documents(*, engine: Engine, question: str, channel_ids: Sequence[str], video_ids: Sequence[str], max_context_chars: int = 150000) -> dict[str, Any]:
    """Read complete selected transcripts into a model packet, without retrieval.

    All selected sources are included or an error is raised. Source text is
    split only to supply citation anchors; every segment stays in the packet.
    """
    scope = _validate_scope(channel_ids)
    selected = list(dict.fromkeys(video_ids))
    if not 1 <= len(selected) <= 10 or any(not re.fullmatch(r"[A-Za-z0-9_-]{11}", value) for value in selected):
        raise ValueError("Select one to ten canonical video IDs.")
    if not isinstance(question, str) or not question.strip() or len(question) > 6000:
        raise ValueError("Question must contain between 1 and 6,000 characters.")
    if not 1000 <= max_context_chars <= 300000:
        raise ValueError("Full-document context limit must be between 1,000 and 300,000 characters.")
    documents: dict[str, dict[str, Any]] = {}
    for document in _load_full_documents(engine, scope, selected):
        if document["channel_id"] not in scope or document["media_id"] not in selected:
            raise ConsultationError("Document loader returned an out-of-scope source.")
        documents.setdefault(document["media_id"], document)
    missing = [video_id for video_id in selected if video_id not in documents]
    if missing:
        raise ConsultationError(f"Selected transcripts are missing or outside the channel scope: {', '.join(missing)}. No partial full-document read was substituted.")
    chars = sum(len(str(documents[video_id]["full_text"])) for video_id in selected)
    if chars > max_context_chars:
        raise ConsultationError(f"Complete selected transcripts contain {chars:,} characters, exceeding the {max_context_chars:,} limit. Select fewer documents or use hosted retrieval; nothing was silently truncated.")
    evidence: list[dict[str, Any]] = []
    for video_id in selected:
        document = documents[video_id]
        full_text = str(document["full_text"]).strip()
        segments = _normalize_segments(document.get("timestamped_segments"))
        # Only use timed blocks when their combined text accounts for the full
        # stored transcript. Otherwise pass the complete original, untimed.
        aligned = bool(segments) and " ".join(full_text.split()) == " ".join(" ".join(segment["text"] for segment in segments).split())
        blocks: list[list[dict[str, Any]]] = []
        if aligned:
            block: list[dict[str, Any]] = []
            block_chars = 0
            for segment in segments:
                if block and block_chars + len(segment["text"]) > 3000:
                    blocks.append(block)
                    block, block_chars = [], 0
                block.append(segment)
                block_chars += len(segment["text"]) + 1
            if block:
                blocks.append(block)
        else:
            blocks = [[{"text": full_text, "start_seconds": None, "duration_seconds": None}]]
        published = document.get("publish_date")
        if isinstance(published, (date, datetime)):
            published = published.isoformat()
        for index, block in enumerate(blocks):
            content = "\n".join(segment["text"] for segment in block)
            start = block[0]["start_seconds"]
            last = block[-1]
            end = last["start_seconds"] + last["duration_seconds"] if last["start_seconds"] is not None and last["duration_seconds"] is not None else None
            url = f"https://www.youtube.com/watch?v={video_id}"
            evidence.append({
                "evidence_id": "E" + hashlib.sha256(f"{document['content_id']}:{index}:{content}".encode()).hexdigest()[:12],
                "video_id": video_id, "content_id": document["content_id"], "channel_id": document["channel_id"],
                "channel_name": document.get("channel_name"), "title": document.get("title"), "published_at": published,
                "url": url, "timestamp_url": f"{url}&t={int(start)}s" if start is not None else None,
                "start_seconds": start, "end_seconds": end, "text": content,
                "speaker": "not verified; channel ownership does not establish speaker identity",
                "context_is_partial": False, "block_index": index, "block_count": len(blocks),
            })
    return {"question": question.strip(), "evidence": evidence, "coverage": {
        "mode": "complete selected transcripts supplied directly to the model",
        "channel_ids": scope, "requested_video_ids": selected, "read_video_ids": selected,
        "complete_selected_documents": True, "selected_videos": len(selected), "selected_text_characters": chars,
        "selected_passages": len(evidence),
        "limitations": "All text of the selected saved transcripts is supplied, not the entire channel. Caption accuracy and speaker attribution are unverified. Timestamp anchors are omitted when stored segments do not account for the complete text.",
    }}


def _validate_packet(packet: dict[str, Any], question: str, scope: list[str]) -> dict[str, Any]:
    if not isinstance(packet, dict) or not isinstance(packet.get("evidence"), list) or not isinstance(packet.get("coverage"), dict):
        raise ValueError("Evidence packet requires evidence and coverage fields.")
    identifiers: set[str] = set()
    for source in packet["evidence"]:
        if not isinstance(source, dict) or source.get("channel_id") not in scope:
            raise ConsultationError("Evidence packet contains an out-of-scope source.")
        identifier = source.get("evidence_id")
        if not isinstance(identifier, str) or not re.fullmatch(r"E[A-Za-z0-9_-]+", identifier) or identifier in identifiers:
            raise ConsultationError("Evidence IDs must be valid and unique.")
        if not isinstance(source.get("text"), str) or not source["text"].strip():
            raise ConsultationError("Evidence source text is missing.")
        identifiers.add(identifier)
    return {"question": question.strip(), "evidence": packet["evidence"], "coverage": packet["coverage"]}


def _request_response(payload: dict[str, Any], api_key: str, timeout_seconds: float) -> dict[str, Any]:
    request = Request("https://api.openai.com/v1/responses", data=json.dumps(payload, ensure_ascii=False).encode(), method="POST", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode())
    except HTTPError as exc:
        # Provider bodies may echo request data; expose only safe status.
        raise ConsultationError(f"OpenAI returned HTTP {exc.code}; no automatic retry was made.") from None
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ConsultationError(f"OpenAI response unavailable ({type(exc).__name__}); the request may have incurred cost and was not retried.") from None
    if not isinstance(result, dict):
        raise ConsultationError("OpenAI returned an invalid response object.")
    return result


def _parse_answer(response: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    if response.get("status") != "completed":
        raise ConsultationError("OpenAI did not complete the consultation; partial output was not accepted.")
    blocks = [content for item in response.get("output", []) if isinstance(item, dict) for content in item.get("content", []) if isinstance(content, dict)]
    if any(block.get("type") == "refusal" for block in blocks):
        raise ConsultationError("OpenAI declined the consultation.")
    output = response.get("output_text") or "".join(str(block.get("text", "")) for block in blocks if block.get("type") == "output_text")
    try:
        answer = _Answer.model_validate_json(output).model_dump()
    except (ValidationError, TypeError) as exc:
        raise ConsultationError("OpenAI output did not match the consultation schema.") from None
    allowed = {item["evidence_id"] for item in evidence}
    for section in ("published_views", "application_angles", "counterarguments"):
        for item in answer[section]:
            citations = item["evidence_ids"]
            if section != "counterarguments" and not citations:
                raise ConsultationError(f"An uncited {section} item was rejected.")
            if any(citation not in allowed for citation in citations):
                raise ConsultationError("An unknown evidence citation was rejected.")
    return answer


def consult_published_views(*, engine: Engine, question: str, channel_ids: Sequence[str], openai_api_key: str, video_ids: Sequence[str] | None = None, evidence_packet: dict[str, Any] | None = None, max_cost_usd: float = 0.10, max_output_tokens: int = 2500, timeout_seconds: float = 90.0, max_context_chars: int = 150000) -> dict[str, Any]:
    """One bounded Luna call; returns evidence and an inference-labeled answer.

    The cap covers this call at the dated rates, not other jobs/account usage.
    No automatic retries, enrichment, acquisition, or external research occurs.
    """
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required.")
    scope = _validate_scope(channel_ids)
    if not isinstance(question, str) or not question.strip() or len(question) > 6000:
        raise ValueError("Question must contain between 1 and 6,000 characters.")
    if (video_ids is None) == (evidence_packet is None):
        raise ValueError("Provide selected video_ids for complete reading or an evidence_packet from hosted retrieval; lexical search is not automatic.")
    try:
        budget = Decimal(str(max_cost_usd))
    except Exception:
        raise ValueError("Invalid consultation budget.") from None
    if not budget.is_finite() or not Decimal("0") < budget <= Decimal("0.10"):
        raise ValueError("Consultation budget must be positive and at most $0.10.")
    if not 256 <= max_output_tokens <= 6000:
        raise ValueError("max_output_tokens must be between 256 and 6,000.")
    packet = (_validate_packet(evidence_packet, question, scope) if evidence_packet is not None else read_consultation_documents(engine=engine, question=question, channel_ids=scope, video_ids=video_ids or [], max_context_chars=max_context_chars))
    if not packet["evidence"]:
        return {"status": "no_evidence", "answer": None, **packet, "usage": None, "estimated_cost_usd": 0.0, "message": "No matching timestamped source passages were found; no model request was made."}
    registry = json.loads(files("learn_from").joinpath("prompts.json").read_text())
    contract = registry["consult_published_views"]
    model_input = contract["input_template"].format(question=packet["question"], coverage_json=json.dumps(packet["coverage"], ensure_ascii=False), evidence_json=json.dumps(packet["evidence"], ensure_ascii=False))
    payload = {
        "model": MODEL, "instructions": contract["instructions"],
        "input": model_input, "reasoning": {"effort": "none"},
        "service_tier": "default", "store": False,
        "truncation": "disabled",
        "max_output_tokens": max_output_tokens,
        "text": {"format": {"type": "json_schema", "name": "published_views_consultation", "strict": True, "schema": _Answer.model_json_schema()}},
    }
    input_token_upper_bound = len(json.dumps(payload, ensure_ascii=False).encode()) + 2048
    if input_token_upper_bound > 900000:
        raise ConsultationError("Input exceeds the supported context bound.")
    long_context = input_token_upper_bound > 272000
    reserved_input_rate = _RESERVATION_INPUT_RATE * (2 if long_context else 1)
    reserved_output_rate = _OUTPUT_RATE * (Decimal("1.5") if long_context else 1)
    reserve = (Decimal(input_token_upper_bound) * reserved_input_rate + Decimal(max_output_tokens) * reserved_output_rate) / 1_000_000
    if reserve > budget:
        raise ConsultationError(f"Conservative call estimate ${reserve:.6f} exceeds the ${budget:.6f} budget; no API request was made.")
    response = _request_response(payload, openai_api_key, timeout_seconds)
    answer = _parse_answer(response, packet["evidence"])
    usage = response.get("usage")
    estimate = None
    if isinstance(usage, dict) and isinstance(usage.get("input_tokens"), int) and isinstance(usage.get("output_tokens"), int):
        actual_long = usage["input_tokens"] > 272000
        estimate = float((Decimal(usage["input_tokens"]) * _INPUT_RATE * (2 if actual_long else 1) + Decimal(usage["output_tokens"]) * _OUTPUT_RATE * (Decimal("1.5") if actual_long else 1)) / 1_000_000)
    return {
        "status": "completed", "answer": answer, **packet,
        "model": MODEL, "prompt_version": registry["version"], "response_id": response.get("id"),
        "usage": usage, "estimated_cost_usd": estimate,
        "reserved_cost_usd": float(reserve), "max_cost_usd": float(budget), "rate_date": RATE_DATE,
        "cost_note": "Estimated standard token cost before cache adjustments, credits, and tax; not a provider invoice. Reservation uses conservative input bounds. No retry or tool calls.",
        "citation_check": "Evidence ID existence only; not independent verification that each claim is supported.",
    }
