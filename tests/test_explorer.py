from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

from learn_from.explorer import ATLAS_KINDS, _v2_extraction_snapshot, make_handler


class _FakeRepository:
    def overview(self):
        return {"totals": {"documents": 2}}

    def facets(self):
        return {"categories": []}

    def documents(self, **kwargs):
        return {"documents": [], "query": kwargs}

    def document(self, media_id):
        return {"media_id": media_id} if media_id == "known" else None

    def atlas(self, **kwargs):
        return {"items": [], **kwargs}

    def atlas_documents(self, **kwargs):
        return {"documents": [], **kwargs}

    def quality(self, **kwargs):
        return {"coverage": [], "documents": [], **kwargs}

    def pipeline_prompt(self, **kwargs):
        return {"name": "Metadata extraction", "schema_version": "2.6", **kwargs}

    def extraction_v2_profiles(self):
        return {"available": True, "count": 0, "profiles": []}

    def profile_contexts(self):
        return {"profiles": [], "aliases": [], "learning_events": [], "counts": {}}

    def save_profile_instructions(self, payload):
        return {"status": "draft", **payload}

    def preview_profile_instructions(self, payload):
        return {"workers": [], "received": payload}

    def extraction_v2_runs(self, **kwargs):
        return {"available": True, "count": 0, "runs": [], **kwargs}


def _server_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(_FakeRepository()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_explorer_handler_serves_static_and_api_routes():
    server, url = _server_url()
    try:
        with urlopen(f"{url}/api/overview") as response:
            assert response.status == 200
            assert b'"documents": 2' in response.read()

        with urlopen(f"{url}/") as response:
            assert response.status == 200
            assert b"Corpus Lens" in response.read()

        with urlopen(Request(f"{url}/", method="HEAD")) as response:
            assert response.status == 200
            assert response.headers["Content-Type"] == "text/html"
    finally:
        server.shutdown()
        server.server_close()


def test_explorer_handler_returns_document_not_found():
    server, url = _server_url()
    try:
        try:
            urlopen(f"{url}/api/documents/missing")
        except Exception as exc:
            assert getattr(exc, "code", None) == 404
        else:
            raise AssertionError("Expected a 404 response")
    finally:
        server.shutdown()
        server.server_close()


def test_atlas_includes_collection_views():
    assert {"category", "channel"}.issubset(ATLAS_KINDS)


def test_explorer_handler_exposes_pipeline_prompt():
    server, url = _server_url()
    try:
        with urlopen(f"{url}/api/pipeline/prompt") as response:
            assert response.status == 200
            assert b'"schema_version": "2.6"' in response.read()

        with urlopen(f"{url}/api/pipeline/prompt?version=1.0") as response:
            assert json.load(response)["version"] == "1.0"
    finally:
        server.shutdown()
        server.server_close()


def test_explorer_handler_exposes_profile_contexts():
    server, url = _server_url()
    try:
        with urlopen(f"{url}/api/extraction/v2/contexts") as response:
            assert response.status == 200
            payload = json.load(response)
            assert payload["profiles"] == []
            assert payload["aliases"] == []
    finally:
        server.shutdown()
        server.server_close()


def test_explorer_handler_saves_profile_instruction_draft():
    server, url = _server_url()
    try:
        body = json.dumps({
            "scope": "category",
            "category_id": 7,
            "prompt_instructions": "Capture every cited source.",
        }).encode()
        request = Request(
            f"{url}/api/extraction/v2/profile-instructions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            assert response.status == 201
            payload = json.load(response)
            assert payload["status"] == "draft"
            assert payload["scope"] == "category"
            assert payload["prompt_instructions"] == "Capture every cited source."
    finally:
        server.shutdown()
        server.server_close()


def test_explorer_handler_previews_effective_profile_instructions():
    server, url = _server_url()
    try:
        body = json.dumps({"config": {"prompt_instructions": "Scoped"}}).encode()
        request = Request(
            f"{url}/api/extraction/v2/instruction-preview",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            assert response.status == 200
            payload = json.load(response)
            assert payload["received"]["config"]["prompt_instructions"] == "Scoped"
    finally:
        server.shutdown()
        server.server_close()


def test_explorer_handler_forwards_universal_extraction_version():
    server, url = _server_url()
    try:
        with urlopen(f"{url}/api/documents?version=1.0") as response:
            payload = json.load(response)
            assert payload["query"]["version"] == "1.0"

        with urlopen(f"{url}/api/atlas?kind=keyword&version=2.0") as response:
            payload = json.load(response)
            assert payload["version"] == "2.0"

        with urlopen(f"{url}/api/quality?version=1.0") as response:
            payload = json.load(response)
            assert payload["version"] == "1.0"
    finally:
        server.shutdown()
        server.server_close()


def test_v2_extraction_snapshot_normalizes_inspector_contract():
    snapshot = _v2_extraction_snapshot(
        {
            "run_id": 7,
            "release_version": "2.0",
            "run_mode": "bootstrap",
            "extractor_model": "gpt-5.6-terra",
            "completed_at": "2026-07-11T12:00:00",
            "manifest": {"profile_version_ids": [1, 2, 5]},
            "raw_output": {
                "category": "LDS Faith",
                "topics": ["fallen angels"],
                "keywords": ["watchers", "Luke 10:18"],
            },
            "resolved_output": {
                "participants": [
                    {
                        "raw_name": "Dan Mlullen",
                        "canonical_name": "Dan McClellan",
                        "role": "host",
                        "evidence": "I'm Dan Mlullen",
                        "confidence": 0.95,
                        "resolution_status": "trusted",
                        "resolution_basis": "normalized",
                    }
                ],
                "entities": [],
                "references": [
                    {
                        "raw_mention": "Luke 10:18",
                        "canonical_label": "Luke 10:18",
                        "type": "scripture",
                        "usage": "evidence",
                        "evidence": "Luke chapter ten verse eighteen",
                        "confidence": 0.99,
                        "resolution_status": "trusted",
                        "resolution_basis": "explicit",
                        "scripture": {"canonical_locator": "scripture:bible:luke:10:18"},
                    }
                ],
            },
        }
    )

    assert snapshot["key"] == "v2.0-run-7"
    assert snapshot["profile_version_ids"] == [1, 2, 5]
    assert snapshot["data"]["participants"][0]["name"] == "Dan McClellan"
    assert snapshot["data"]["references"][0]["label"] == "Luke 10:18"
    assert snapshot["data"]["keywords"] == ["watchers"]


def test_v2_snapshot_reclassifies_previous_guest_as_entity():
    snapshot = _v2_extraction_snapshot(
        {
            "run_id": 8,
            "release_version": "2.0",
            "run_mode": "stable",
            "extractor_model": "gpt-5.6-luna",
            "manifest": {},
            "raw_output": {"topics": [], "keywords": []},
            "resolved_output": {
                "participants": [
                    {
                        "raw_name": "April of Joy",
                        "canonical_name": "April of Joy",
                        "role": "previous guest mentioned in discussion",
                        "evidence": "The hosts recall having April of Joy on the show.",
                        "confidence": 0.98,
                        "resolution_status": "trusted",
                        "resolution_basis": "explicit",
                    }
                ],
                "entities": [],
                "references": [],
            },
        }
    )

    assert snapshot["data"]["participants"] == []
    assert snapshot["data"]["entities"][0]["label"] == "April of Joy"
