from __future__ import annotations

import pytest
from sqlalchemy import JSON, Boolean, Column, Integer, MetaData, String, Table, create_engine, insert, select

from learn_from.metadata_v2 import (
    ContentContext,
    EffectiveProfile,
    ModelRoute,
    build_release_manifest,
    build_transcript_chunks,
    default_profile_config,
    determine_model_route,
    get_metadata_v2_prompt_inspection,
    light_normalize_text,
    preview_profile_instruction_contracts,
    run_extractor_worker,
    run_resolver_worker,
    save_scoped_instruction_draft,
    summarize_run_usage,
)


def _context(*, source_type: str = "channel", document_count: int = 1) -> ContentContext:
    return ContentContext(
        content_id=1,
        media_id="video-1",
        source_id="youtube",
        source_type=source_type,
        channel_id="channel-1" if source_type == "channel" else None,
        channel_name="Dan McClellan" if source_type == "channel" else None,
        category_id=7,
        category_name="Christian Faith",
        title="Angels in chains",
        description=None,
        transcript_text="The speaker compares Isaiah 14 with Luke 10:18.",
        timestamped_segments=[],
        channel_document_count=document_count,
    )


def _profile(*, scripture: bool = True, participants: list[str] | None = None) -> EffectiveProfile:
    config = default_profile_config()
    if scripture:
        config["reference_modules"] = ["scripture"]
    if participants is not None:
        config["known_values"] = {"participants": participants}
    return EffectiveProfile(
        profile_version_id=1,
        profile_ids=(1,),
        version_ids=(1,),
        learning_mode="autopilot",
        config=config,
    )


def test_v2_prompt_inspection_uses_runtime_builders_and_placeholders() -> None:
    inspection = get_metadata_v2_prompt_inspection()

    assert inspection["name"] == "Metadata Extraction V2"
    assert "You are Worker 1" in inspection["prompt_contracts"]["extractor"]["instructions"]
    assert "{{ chunk.text }}" in inspection["prompt_contracts"]["extractor"]["input_template"]
    assert "You are Worker 2" in inspection["prompt_contracts"]["resolver"]["instructions"]
    assert "{{ deterministic_resolution.unresolved | json }}" in inspection["prompt_contracts"]["resolver"]["input_template"]
    assert "You are Worker 3" in inspection["prompt_contracts"]["learner"]["instructions"]
    assert inspection["prompt_contracts"]["learner"]["json_schema"]["additionalProperties"] is False


def test_profile_instruction_preview_uses_effective_runtime_builders() -> None:
    preview = preview_profile_instruction_contracts({
        "reference_modules": ["scripture"],
        "prompt_instructions": "Prefer canonical scripture labels.",
    })

    extractor = next(worker for worker in preview["workers"] if worker["key"] == "extractor")
    resolver = next(worker for worker in preview["workers"] if worker["key"] == "resolver")
    learner = next(worker for worker in preview["workers"] if worker["key"] == "learner")
    assert "Prefer canonical scripture labels." not in extractor["base_instructions"]
    assert "Prefer canonical scripture labels." in extractor["effective_instructions"]
    assert "Enabled modules: scripture" in extractor["effective_instructions"]
    assert "Enabled specialized modules: scripture" in resolver["effective_instructions"]
    assert learner["base_instructions"] == learner["effective_instructions"]
    assert preview["prompt_instruction_behavior"]["merge"] == "most_specific_replaces"


def test_scoped_instruction_save_updates_one_draft_and_preserves_active_config() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    metadata = MetaData()
    profiles = Table(
        "extraction_profiles", metadata,
        Column("profile_id", Integer, primary_key=True, autoincrement=True),
        Column("profile_name", String, nullable=False),
        Column("scope", String, nullable=False),
        Column("source_id", String), Column("channel_id", String), Column("category_id", Integer),
        Column("learning_mode", String, nullable=False), Column("is_active", Boolean, nullable=False),
    )
    versions = Table(
        "extraction_profile_versions", metadata,
        Column("profile_version_id", Integer, primary_key=True, autoincrement=True),
        Column("profile_id", Integer, nullable=False), Column("version", Integer, nullable=False),
        Column("status", String, nullable=False), Column("config", JSON, nullable=False),
    )
    categories = Table(
        "doc_categories", metadata,
        Column("category_id", Integer, primary_key=True), Column("category_name", String, nullable=False),
    )
    Table(
        "channels", metadata,
        Column("channel_id", String, primary_key=True), Column("channel_name", String, nullable=False),
        Column("source_id", String, nullable=False),
    )
    metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(insert(categories).values(category_id=7, category_name="Science"))
        profile_id = connection.execute(insert(profiles).values(
            profile_name="Science Metadata V2", scope="category", category_id=7,
            learning_mode="autopilot", is_active=True,
        )).inserted_primary_key[0]
        connection.execute(insert(versions).values(
            profile_id=profile_id, version=1, status="active",
            config={"prompt_instructions": "Original", "known_values": {"participants": ["Ada"]}},
        ))

    first = save_scoped_instruction_draft(
        engine=engine, scope="category", category_id=7, prompt_instructions="Revised instructions",
    )
    second = save_scoped_instruction_draft(
        engine=engine, scope="category", category_id=7, prompt_instructions="Final draft",
    )

    assert first["profile_version_id"] == second["profile_version_id"]
    with engine.connect() as connection:
        rows = connection.execute(select(versions).order_by(versions.c.version)).mappings().all()
    assert len(rows) == 2
    assert rows[0]["status"] == "active" and rows[0]["config"]["prompt_instructions"] == "Original"
    assert rows[1]["status"] == "draft" and rows[1]["config"]["prompt_instructions"] == "Final draft"
    assert rows[1]["config"]["known_values"] == {"participants": ["Ada"]}
    with pytest.raises(ValueError, match="limited to category and channel"):
        save_scoped_instruction_draft(engine=engine, scope="global", prompt_instructions="Nope")


def test_light_normalization_collapses_repeated_fillers_without_erasing_uncertainty() -> None:
    value = light_normalize_text("Um, um, I think it is Isaiah, but I am not sure.")

    assert value.lower().count("um") == 1
    assert "not sure" in value


def test_transcript_chunking_covers_text_beyond_v1_limit() -> None:
    transcript = " ".join(f"word-{index}" for index in range(7000))
    analysis_text, _, chunks = build_transcript_chunks(
        transcript_text=transcript,
        max_chars=4000,
        overlap_segments=0,
    )

    assert len(analysis_text) > 12_000
    assert len(chunks) > 3
    assert "word-6999" in chunks[-1].text


def test_model_routing_uses_bootstrap_stable_audit_and_ad_hoc_modes() -> None:
    profile = _profile()

    bootstrap = determine_model_route(context=_context(document_count=2), profile=profile)
    stable = determine_model_route(context=_context(document_count=3), profile=profile)
    audit = determine_model_route(context=_context(document_count=9), profile=profile)
    ad_hoc = determine_model_route(context=_context(source_type="ad_hoc"), profile=profile)

    assert (bootstrap.run_mode, bootstrap.extractor_model) == ("bootstrap", "gpt-5.6-terra")
    assert (stable.run_mode, stable.extractor_model) == ("stable", "gpt-5.6-luna")
    assert (audit.run_mode, audit.extractor_model) == ("audit", "gpt-5.6-terra")
    assert (ad_hoc.run_mode, ad_hoc.extractor_model) == ("ad_hoc", "gpt-5.6-luna")


def test_extractor_supplements_missing_scripture_with_deterministic_full_text_detection() -> None:
    context = _context()
    route = determine_model_route(context=context, profile=_profile())

    def requester(**kwargs):
        return {
            "category": "Christian Faith",
            "topics": ["fallen angels"],
            "keywords": ["angels"],
            "participants": [],
            "entities": [],
            "references": [],
            "category_attributes": [],
        }

    extraction, _, _, _ = run_extractor_worker(
        context=context,
        profile=_profile(),
        route=route,
        openai_api_key="test-key",
        requester=requester,
    )

    assert {item["suggested_label"] for item in extraction["references"]} == {
        "Isaiah 14",
        "Luke 10:18",
    }


def test_extractor_removes_scripture_citations_from_keywords() -> None:
    context = _context()
    route = determine_model_route(context=context, profile=_profile())

    def requester(**kwargs):
        return {
            "category": "Christian Faith",
            "topics": ["rules of war"],
            "keywords": ["deuteronomy 20:10-14", "forced labor"],
            "participants": [],
            "entities": [],
            "references": [],
            "category_attributes": [],
        }

    extraction, _, _, _ = run_extractor_worker(
        context=context,
        profile=_profile(),
        route=route,
        openai_api_key="test-key",
        requester=requester,
    )

    assert extraction["keywords"] == ["forced labor"]


def test_resolver_rejects_generic_roles_and_maps_channel_name_typo() -> None:
    context = _context()
    profile = _profile()
    route = ModelRoute(
        run_mode="bootstrap",
        extractor_model="gpt-5.6-terra",
        extractor_reasoning="medium",
        resolver_model="gpt-5.6-terra",
        learner_model="gpt-5.6-terra",
    )
    extraction = {
        "participants": [
            {"raw_name": "speaker", "role": None, "evidence": "speaker says", "confidence": 0.99},
            {
                "raw_name": "Dan Mlelen",
                "role": "host",
                "evidence": "Dan Mlelen explains",
                "confidence": 0.90,
            },
        ],
        "entities": [],
        "references": [],
    }

    resolution = run_resolver_worker(
        context=context,
        profile=profile,
        route=route,
        extraction=extraction,
        aliases={},
        openai_api_key="test-key",
        use_model=False,
    )

    assert resolution["participants"][0]["resolution_status"] == "rejected"
    assert resolution["participants"][0]["canonical_name"] is None
    assert resolution["participants"][1]["canonical_name"] == "Dan McClellan"
    assert resolution["participants"][1]["resolution_status"] == "trusted"


def test_resolver_uses_profile_participant_hints_when_channel_is_a_brand() -> None:
    context = ContentContext(
        **{
            **_context().__dict__,
            "channel_name": "Data Over Dogma",
        }
    )
    profile = _profile(participants=["Dan McClellan", "Dan Beecher"])
    route = determine_model_route(context=context, profile=profile)
    extraction = {
        "participants": [
            {
                "raw_name": "Dan Mlullen",
                "role": "host",
                "evidence": "I'm Dan Mlullen",
                "confidence": 0.94,
            }
        ],
        "entities": [],
        "references": [],
    }

    resolution = run_resolver_worker(
        context=context,
        profile=profile,
        route=route,
        extraction=extraction,
        aliases={},
        openai_api_key="test-key",
        use_model=False,
    )

    participant = resolution["participants"][0]
    assert participant["canonical_name"] == "Dan McClellan"
    assert participant["resolution_status"] == "trusted"
    assert participant["resolution_basis"] == "normalized"


def test_resolver_keeps_shared_first_name_ambiguous() -> None:
    context = _context()
    profile = _profile(participants=["Dan McClellan", "Dan Beecher"])
    route = determine_model_route(context=context, profile=profile)
    extraction = {
        "participants": [
            {
                "raw_name": "Dan",
                "role": None,
                "evidence": "The other speaker addresses Dan.",
                "confidence": 0.99,
            }
        ],
        "entities": [],
        "references": [],
    }

    resolution = run_resolver_worker(
        context=context,
        profile=profile,
        route=route,
        extraction=extraction,
        aliases={},
        openai_api_key="test-key",
        use_model=False,
    )

    participant = resolution["participants"][0]
    assert participant["canonical_name"] is None
    assert participant["resolution_status"] == "provisional"


def test_resolver_approved_alias_overrides_shared_first_name_ambiguity() -> None:
    context = _context()
    profile = _profile(participants=["Dan McClellan", "Dan Beecher"])
    route = determine_model_route(context=context, profile=profile)
    extraction = {
        "participants": [
            {"raw_name": "Dan", "role": None, "evidence": "Dan explains the passage.", "confidence": 0.99},
            {"raw_name": "Dan McClellan", "role": "host", "evidence": "Dan McClellan explains.", "confidence": 0.98},
        ],
        "entities": [],
        "references": [],
    }

    resolution = run_resolver_worker(
        context=context,
        profile=profile,
        route=route,
        extraction=extraction,
        aliases={"participant": {"dan": "Dan McClellan"}},
        openai_api_key="test-key",
        use_model=False,
    )

    assert len(resolution["participants"]) == 1
    assert resolution["participants"][0]["canonical_name"] == "Dan McClellan"
    assert resolution["participants"][0]["resolution_basis"] == "alias"


def test_resolver_moves_previous_guest_from_participant_to_entity() -> None:
    context = _context()
    profile = _profile()
    route = determine_model_route(context=context, profile=profile)
    extraction = {
        "participants": [
            {
                "raw_name": "April of Joy",
                "role": "previous guest mentioned in discussion",
                "evidence": "The hosts recall having April of Joy on the show.",
                "confidence": 0.98,
            }
        ],
        "entities": [],
        "references": [],
    }

    resolution = run_resolver_worker(
        context=context,
        profile=profile,
        route=route,
        extraction=extraction,
        aliases={},
        openai_api_key="test-key",
        use_model=False,
    )

    assert resolution["participants"][0]["resolution_status"] == "rejected"
    assert resolution["entities"][0]["canonical_label"] == "April of Joy"
    assert resolution["entities"][0]["usage"] == "background"


def test_inferred_scripture_remains_provisional() -> None:
    context = _context()
    profile = _profile()
    route = determine_model_route(context=context, profile=profile)
    extraction = {
        "participants": [],
        "entities": [],
        "references": [
            {
                "type": "scripture",
                "raw_mention": "the passage about the fallen morning star",
                "suggested_label": "Isaiah 14",
                "context": None,
                "usage": "background",
                "evidence": "the passage about the fallen morning star",
                "confidence": 0.97,
                "resolution_basis": "inferred",
            }
        ],
    }

    resolution = run_resolver_worker(
        context=context,
        profile=profile,
        route=route,
        extraction=extraction,
        aliases={},
        openai_api_key="test-key",
        use_model=False,
    )

    reference = resolution["references"][0]
    assert reference["canonical_label"] == "Isaiah 14"
    assert reference["resolution_basis"] == "inferred"
    assert reference["resolution_status"] == "provisional"


def test_release_manifest_is_stable_and_complete() -> None:
    profile = _profile()
    route = determine_model_route(context=_context(), profile=profile)

    first = build_release_manifest(profile=profile, route=route)
    second = build_release_manifest(profile=profile, route=route)

    assert first["manifest_hash"] == second["manifest_hash"]
    assert first["pipeline_version"] == "2.0"
    assert first["normalizer_version"] == "light-1"
    assert first["reference_modules"] == ["scripture"]
    assert first["extractor"]["invoked"] is True
    assert first["resolver"]["invoked"] is True
    assert first["learner"]["invoked"] is True


def test_release_manifest_records_skipped_model_workers() -> None:
    profile = _profile()
    route = determine_model_route(context=_context(), profile=profile)

    manifest = build_release_manifest(
        profile=profile,
        route=route,
        resolver_model_enabled=False,
        learner_model_enabled=False,
    )

    assert manifest["extractor"]["invoked"] is True
    assert manifest["resolver"]["invoked"] is False
    assert manifest["learner"]["invoked"] is False


def test_run_usage_summary_rolls_worker_tokens_and_costs_up() -> None:
    profile = _profile()
    route = determine_model_route(context=_context(), profile=profile)
    manifest = build_release_manifest(profile=profile, route=route)
    usage = summarize_run_usage(
        events=[
            {
                "step": "extractor",
                "model": "gpt-5.6-terra",
                "input_tokens": 1_000,
                "cached_input_tokens": 100,
                "output_tokens": 200,
                "reasoning_tokens": 50,
                "total_tokens": 1_200,
                "estimated_cost": 0.005275,
                "service_tier": "default",
            },
            {
                "step": "resolver",
                "model": "gpt-5.6-terra",
                "input_tokens": 300,
                "cached_input_tokens": 0,
                "output_tokens": 100,
                "reasoning_tokens": 20,
                "total_tokens": 400,
                "estimated_cost": 0.00225,
                "service_tier": "flex",
            },
        ],
        manifest=manifest,
        chunk_count=1,
        durations={"extractor": 2.5, "resolver": 1.0, "learner": 0.2},
    )

    assert usage["tracking_status"] == "recorded"
    assert usage["total_tokens"] == 1_600
    assert usage["input_tokens"] == 1_300
    assert usage["output_tokens"] == 300
    assert usage["estimated_cost"] == 0.007525
    assert [step["step"] for step in usage["steps"]] == [
        "normalization",
        "extractor",
        "resolver",
        "learner",
        "persistence",
    ]


def test_luna_cost_separates_cached_reads_writes_and_uncached_input() -> None:
    from learn_from.metadata_v2 import _estimate_token_cost

    event = {"model": "gpt-6-luna", "input_tokens": 1000, "cached_input_tokens": 100,
             "cache_write_tokens": 700, "output_tokens": 100, "service_tier": "default", "cache_ttl": "30m"}
    assert _estimate_token_cost(event) == pytest.approx(0.0001585)
    assert _estimate_token_cost({**event, "service_tier": "flex"}) == pytest.approx(0.00007925)
    assert _estimate_token_cost({**event, "cache_ttl": "12h"}) == pytest.approx(0.000246)
    assert _estimate_token_cost({"model": "gpt-6-luna", "input_tokens": 300000,
                                 "cache_write_tokens": 300000, "output_tokens": 1000, "cache_ttl": "30m"}) == pytest.approx(0.07575)


def test_segment_overlap_releases_space_for_a_large_following_segment() -> None:
    segments = [{"text": "a " * 300, "start_seconds": 0, "duration_seconds": 1},
                {"text": "b " * 300, "start_seconds": 1, "duration_seconds": 1}]
    text, _, chunks = build_transcript_chunks(transcript_text="", timestamped_segments=segments,
                                              max_chars=1000, overlap_segments=3)
    assert len(chunks) == 2
    assert "a " in chunks[0].text and "b " in chunks[-1].text
    assert text.count('a') == 300 and text.count('b') == 300


def test_cache_write_ttl_split_overrides_deprecated_retention_and_unknown_ttl_is_not_priced() -> None:
    from learn_from.metadata_v2 import _estimate_token_cost

    event = {'model': 'gpt-6-luna', 'input_tokens': 12000, 'cached_input_tokens': 4000,
             'cache_write_tokens': 7000, 'cache_write_30m_tokens': 3000,
             'cache_write_12h_tokens': 4000, 'output_tokens': 100, 'cache_retention': '24h'}
    assert _estimate_token_cost(event) == pytest.approx(0.001565)
    unknown = {key: value for key, value in event.items() if key not in {'cache_write_30m_tokens', 'cache_write_12h_tokens'}}
    assert _estimate_token_cost(unknown) is None
    assert _estimate_token_cost({**unknown, 'cache_ttl': 'unknown'}) is None
    assert _estimate_token_cost({**unknown, 'cache_ttl': '30m'}) == pytest.approx(0.001065)
    assert _estimate_token_cost({**event, 'cache_write_12h_tokens': 3999}) is None


def test_usage_capture_records_split_lifetimes_and_allowlisted_failure_diagnostics() -> None:
    from learn_from.metadata_v2 import _USAGE_COLLECTOR, _capture_response_usage

    events = []
    token = _USAGE_COLLECTOR.set(events)
    try:
        _capture_response_usage(response={'status': 'incomplete', 'incomplete_details': {'reason': 'max_output_tokens'},
            'prompt_cache_retention': '24h', 'prompt_cache_options': {'ttl': '12h'},
            'usage': {'input_tokens': 12000, 'output_tokens': 100,
                'input_tokens_details': {'cached_tokens': 4000, 'cache_write_tokens': 7000,
                    'cache_write_tokens_details': {'ttl_30m_tokens': 3000, 'ttl_12h_tokens': 4000}}}},
            model='gpt-6-luna', requested_service_tier='default', requested_cache_ttl='30m')
        _capture_response_usage(response={'status': 'incomplete', 'incomplete_details': {'reason': 'secret-body-value'}},
            model='gpt-6-luna', requested_service_tier='default')
    finally:
        _USAGE_COLLECTOR.reset(token)
    assert events[0]['cache_ttl'] == '12h'
    assert events[0]['estimated_cost'] == pytest.approx(0.001565)
    assert events[0]['incomplete_reason'] == 'max_output_tokens'
    assert events[1]['incomplete_reason'] == 'other'
    assert events[1]['estimated_cost'] is None and events[1]['total_tokens'] is None
    assert 'secret-body-value' not in str(events)


def test_saved_receipts_can_be_repriced_without_changing_original_or_guessing_ttl() -> None:
    from learn_from.metadata_v2 import recalculate_metadata_usage

    usage = {'events': [{'model': 'gpt-6-luna', 'step': 'extractor', 'input_tokens': 7335,
        'cache_write_tokens': 7332, 'cached_input_tokens': 0, 'output_tokens': 1145,
        'total_tokens': 8480, 'cache_retention': '24h', 'estimated_cost': 0.0014893}],
        'steps': [{'step': 'extractor', 'kind': 'model', 'estimated_cost': 0.0014893}],
        'estimated_cost': 0.0014893, 'unreported_requests': 1}
    assert recalculate_metadata_usage(usage)['estimated_cost'] is None
    updated = recalculate_metadata_usage(usage, assumed_cache_ttl='30m')
    assert updated['estimated_cost'] == usage['estimated_cost']
    assert updated['steps'][0]['estimated_cost'] == usage['estimated_cost']
    assert updated['unreported_requests'] == 1
    assert 'cache_ttl' not in usage['events'][0]


def test_responses_parser_ignores_reasoning_tool_arguments_and_commentary() -> None:
    from learn_from.metadata_v2 import _parse_v2_response

    response = {'output': [
        {'type': 'reasoning', 'content': [{'type': 'reasoning_text', 'text': 'not JSON'}]},
        {'type': 'function_call', 'arguments': '{"wrong":true}'},
        {'type': 'message', 'role': 'assistant', 'phase': 'commentary', 'content': [{'type': 'output_text', 'text': '{"wrong":true}\nExplanation after JSON.'}]},
        {'type': 'message', 'role': 'assistant', 'phase': 'final_answer', 'status': 'completed',
         'content': [{'type': 'output_text', 'text': '{"right":'}, {'type': 'output_text', 'text': 'true}'}]},
    ]}
    assert _parse_v2_response(response) == {'right': True}
    with pytest.raises(Exception, match='missing_assistant_output'):
        _parse_v2_response({'output': [response['output'][2]]})
    with pytest.raises(Exception, match='missing_assistant_output'):
        _parse_v2_response({'output': response['output'][:2]})
    with pytest.raises(Exception, match='refusal'):
        _parse_v2_response({'output': [{'type': 'message', 'role': 'assistant', 'content': [{'type': 'refusal', 'refusal': 'private details'}]}]})
    with pytest.raises(Exception, match='invalid_json') as raised:
        _parse_v2_response({'output_text': 'private source text'})
    assert 'private source text' not in str(raised.value)


def test_index_timestamp_matches_full_quote_across_segments_and_rejects_ambiguity() -> None:
    from learn_from.metadata_v2 import _find_index_evidence_time

    segments = [
        {'text': 'Yeah.', 'start_seconds': 2, 'duration_seconds': 1},
        {'text': 'A great ecosystem to sell', 'start_seconds': 50, 'duration_seconds': 3},
        {'text': 'and be paid for the work. Yeah.', 'start_seconds': 53, 'duration_seconds': 4},
    ]
    assert _find_index_evidence_time('A great ecosystem to sell and be paid for the work. Yeah.', segments) == (50, 57)
    assert _find_index_evidence_time('Yeah.', segments) == (None, None)
    assert _find_index_evidence_time('not in transcript', segments) == (None, None)
