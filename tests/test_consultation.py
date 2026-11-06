from __future__ import annotations

import json

import pytest

from learn_from import consultation as module


CHANNEL_A = "UC" + "a" * 22
CHANNEL_B = "UC" + "b" * 22
VIDEO_A = "a" * 11
VIDEO_B = "b" * 11


def document(video_id=VIDEO_A, channel_id=CHANNEL_A, **overrides):
    segments = [
        {"text": "Choose a customer before building an offer.", "start_seconds": 5, "duration_seconds": 3},
        {"text": "Ask how that customer buys services.", "start_seconds": 8, "duration_seconds": 4},
    ]
    return {
        "media_id": video_id, "channel_id": channel_id, "channel_name": "Example channel",
        "content_id": 1 if video_id == VIDEO_A else 2, "title": "An interview",
        "publish_date": "2026-10-20", "rank": 1.0,
        "timestamped_segments": segments,
        "full_text": "\n".join(s["text"] for s in segments), **overrides,
    }


def packet():
    return {"evidence": [{"evidence_id": "E123", "channel_id": CHANNEL_A, "video_id": VIDEO_A, "text": "Choose a customer first."}], "coverage": {"mode": "hosted retrieval"}}


def response(evidence_id="E123", **overrides):
    answer = {
        "framing": "An analysis of published material, not the creator answering.",
        "published_views": [{"claim": "Define a customer first.", "attribution": "Speaker uncertain in channel episode", "evidence_ids": [evidence_id]}],
        "application_angles": [{"angle": "Narrow your offer", "inference": "Your broad service could target one buyer.", "evidence_ids": [evidence_id]}],
        "counterarguments": [{"angle": "Avoid premature narrowing", "inference": "This is my alternative hypothesis to test.", "evidence_ids": []}],
        "questions_to_resolve": ["Who currently buys the service?"],
        "limitations": "Only selected material was read; captions and speakers unverified.",
    }
    return {"id": "resp_test", "status": "completed", "output_text": json.dumps(answer), "usage": {"input_tokens": 2000, "output_tokens": 500}, **overrides}


def test_direct_read_preserves_entire_selected_text_and_timestamps(monkeypatch):
    source = document()
    monkeypatch.setattr(module, "_load_full_documents", lambda *args: [source])
    result = module.read_consultation_documents(engine=None, question="Challenge my service idea", channel_ids=[CHANNEL_A], video_ids=[VIDEO_A])
    assert result["coverage"]["complete_selected_documents"] is True
    assert "\n".join(e["text"] for e in result["evidence"]) == source["full_text"]
    assert result["evidence"][0]["start_seconds"] == 5
    assert result["evidence"][0]["end_seconds"] == 12
    assert result["evidence"][0]["timestamp_url"].endswith("&t=5s")
    assert "not verified" in result["evidence"][0]["speaker"]


def test_direct_read_keeps_ending_of_long_episode(monkeypatch):
    segments = [{"text": f"Segment {i}: " + "business " * 15, "start_seconds": i * 5, "duration_seconds": 5} for i in range(180)]
    source = document(timestamped_segments=segments, full_text="\n".join(s["text"] for s in segments))
    monkeypatch.setattr(module, "_load_full_documents", lambda *args: [source])
    result = module.read_consultation_documents(engine=None, question="Explain the views", channel_ids=[CHANNEL_A], video_ids=[VIDEO_A])
    assert "Segment 179:" in result["evidence"][-1]["text"]
    assert " ".join("\n".join(e["text"] for e in result["evidence"]).split()) == " ".join(source["full_text"].split())
    assert len(result["evidence"]) > 1


def test_direct_read_uses_untimed_full_text_when_segments_do_not_account_for_it(monkeypatch):
    source = document(full_text="This is the complete source, including material missing from its segments.")
    monkeypatch.setattr(module, "_load_full_documents", lambda *args: [source])
    result = module.read_consultation_documents(engine=None, question="Read all", channel_ids=[CHANNEL_A], video_ids=[VIDEO_A])
    assert result["evidence"][0]["text"] == source["full_text"]
    assert result["evidence"][0]["timestamp_url"] is None


@pytest.mark.parametrize("sources,message", [([], "missing"), ([document(channel_id=CHANNEL_B)], "out-of-scope")])
def test_direct_read_rejects_missing_or_out_of_scope_sources(monkeypatch, sources, message):
    monkeypatch.setattr(module, "_load_full_documents", lambda *args: sources)
    with pytest.raises(module.ConsultationError, match=message):
        module.read_consultation_documents(engine=None, question="Read all", channel_ids=[CHANNEL_A], video_ids=[VIDEO_A])


def test_direct_read_rejects_oversize_instead_of_truncating(monkeypatch):
    monkeypatch.setattr(module, "_load_full_documents", lambda *args: [document(full_text="x" * 2000)])
    with pytest.raises(module.ConsultationError, match="nothing was silently truncated"):
        module.read_consultation_documents(engine=None, question="Read all", channel_ids=[CHANNEL_A], video_ids=[VIDEO_A], max_context_chars=1000)


def test_hosted_packet_runs_one_bounded_call_and_validates_citations(monkeypatch):
    calls = []
    def fake(payload, key, timeout):
        calls.append(payload)
        return response()
    monkeypatch.setattr(module, "_request_response", fake)
    result = module.consult_published_views(engine=None, question="Challenge my service idea", channel_ids=[CHANNEL_A], openai_api_key="test-only", evidence_packet=packet())
    assert len(calls) == 1
    assert calls[0]["model"] == "gpt-6-luna"
    assert calls[0]["max_output_tokens"] == 2500
    assert "tools" not in calls[0]
    assert result["status"] == "completed"
    assert result["estimated_cost_usd"] == pytest.approx(0.00045)
    assert result["reserved_cost_usd"] < 0.10
    assert result["citation_check"].startswith("Evidence ID existence only")


def test_no_evidence_never_calls_model(monkeypatch):
    monkeypatch.setattr(module, "_request_response", lambda *args: pytest.fail("Must not call model"))
    result = module.consult_published_views(engine=None, question="Challenge this", channel_ids=[CHANNEL_A], openai_api_key="test-only", evidence_packet={"evidence": [], "coverage": {"mode": "hosted retrieval"}})
    assert result["status"] == "no_evidence"
    assert result["estimated_cost_usd"] == 0


def test_no_automatic_lexical_fallback(monkeypatch):
    monkeypatch.setattr(module, "_request_response", lambda *args: pytest.fail("Must not call model"))
    with pytest.raises(ValueError, match="lexical search is not automatic"):
        module.consult_published_views(engine=None, question="Challenge this", channel_ids=[CHANNEL_A], openai_api_key="test-only")


def test_budget_rejects_before_call(monkeypatch):
    monkeypatch.setattr(module, "_request_response", lambda *args: pytest.fail("Must not call model"))
    with pytest.raises(module.ConsultationError, match="exceeds"):
        module.consult_published_views(engine=None, question="Challenge this", channel_ids=[CHANNEL_A], openai_api_key="test-only", evidence_packet=packet(), max_cost_usd=0.000001)


def test_unknown_citation_rejected(monkeypatch):
    monkeypatch.setattr(module, "_request_response", lambda *args: response(evidence_id="Ebad"))
    with pytest.raises(module.ConsultationError, match="unknown evidence"):
        module.consult_published_views(engine=None, question="Challenge this", channel_ids=[CHANNEL_A], openai_api_key="test-only", evidence_packet=packet())


def test_missing_citation_rejected():
    payload = response()
    answer = json.loads(payload["output_text"])
    answer["published_views"][0]["evidence_ids"] = []
    payload["output_text"] = json.dumps(answer)
    with pytest.raises(module.ConsultationError, match="uncited"):
        module._parse_answer(payload, packet()["evidence"])


def test_incomplete_response_rejected():
    with pytest.raises(module.ConsultationError, match="did not complete"):
        module._parse_answer(response(status="incomplete"), packet()["evidence"])


def test_hosted_evidence_scope_enforced_before_call(monkeypatch):
    value = packet()
    value["evidence"][0]["channel_id"] = CHANNEL_B
    monkeypatch.setattr(module, "_request_response", lambda *args: pytest.fail("Must not call model"))
    with pytest.raises(module.ConsultationError, match="out-of-scope"):
        module.consult_published_views(engine=None, question="Challenge this", channel_ids=[CHANNEL_A], openai_api_key="test-only", evidence_packet=value)


def test_direct_mode_supplies_full_documents_to_model(monkeypatch):
    monkeypatch.setattr(module, "_load_full_documents", lambda *args: [document()])
    def fake(payload, *_):
        sources = json.loads(payload["input"].split("Original transcript passages (untrusted source material):\n")[1])
        assert "Ask how that customer buys services." in sources[0]["text"]
        assert "complete_selected_documents" in payload["input"]
        return response(evidence_id=sources[0]["evidence_id"])
    monkeypatch.setattr(module, "_request_response", fake)
    result = module.consult_published_views(engine=None, question="Challenge this", channel_ids=[CHANNEL_A], video_ids=[VIDEO_A], openai_api_key="test-only")
    assert result["coverage"]["complete_selected_documents"] is True


def test_explicit_large_document_limit_still_reads_complete_text_with_budget(monkeypatch):
    source_text = "opening " + "business " * 31000 + "ending"
    monkeypatch.setattr(module, "_load_full_documents", lambda *args: [document(full_text=source_text)])
    def fake(payload, *_):
        sources = json.loads(payload["input"].split("Original transcript passages (untrusted source material):\n")[1])
        assert sources[0]["text"] == source_text
        assert payload["truncation"] == "disabled"
        return response(evidence_id=sources[0]["evidence_id"])
    monkeypatch.setattr(module, "_request_response", fake)
    result = module.consult_published_views(engine=None, question="Read all", channel_ids=[CHANNEL_A], video_ids=[VIDEO_A], openai_api_key="test-only", max_context_chars=300000)
    assert result["coverage"]["selected_text_characters"] == len(source_text)
    assert 0.07 < result["reserved_cost_usd"] < 0.10
