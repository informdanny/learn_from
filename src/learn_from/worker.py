"""One-shot cloud job. Nothing is polled or scheduled after it finishes."""
from __future__ import annotations

import json
import os


def run_worker(*, settings, engine) -> int:
    from learn_from.collections import run_collection
    from learn_from.operations import _emit_index

    job_id = os.environ.get("LEARN_FROM_COLLECTION_JOB_ID", "").strip()
    if not job_id:
        raise ValueError("LEARN_FROM_COLLECTION_JOB_ID is required.")
    index_setting = os.environ.get("LEARN_FROM_INDEX_AFTER_COLLECTION", "true").strip().lower()
    if index_setting not in {"true", "false"}:
        raise ValueError("LEARN_FROM_INDEX_AFTER_COLLECTION must be true or false.")
    index_enabled = index_setting == "true"
    max_files = 1000
    if index_enabled:
        try:
            max_files = int(os.environ.get("LEARN_FROM_MAX_INDEX_FILES", "1000"))
        except ValueError:
            raise ValueError("LEARN_FROM_MAX_INDEX_FILES must be an integer from 1 to 10,000.") from None
        if not 1 <= max_files <= 10000:
            raise ValueError("LEARN_FROM_MAX_INDEX_FILES must be an integer from 1 to 10,000.")
    emit = lambda value: print(json.dumps(value, default=str, ensure_ascii=False), flush=True)
    result = run_collection(engine=engine, job_id=job_id, settings=settings, on_progress=emit)
    emit({"stage": "collection", "result": result})
    index_succeeded = True
    if index_enabled:
        if result.get("acquired_count", 0) + result.get("reused_count", 0) > 0:
            from learn_from.hosted_retrieval import sync_collection_index
            indexed = sync_collection_index(engine=engine, job_id=job_id, api_key=settings.openai_api_key,
                max_files=max_files, on_progress=_emit_index)
            _emit_index(indexed)
            index_succeeded = indexed.get("status") in {"completed", "file_limit_reached", "storage_limit_reached"}
        else:
            emit({"stage": "hosted_index", "result": {"status": "skipped", "reason": "No acquired or reused transcripts to index."}})
    # Reaching the user's request cap is expected; unresolved provider failures are not.
    return 0 if result.get("status") in {"completed", "budget_halted"} and index_succeeded else 1
