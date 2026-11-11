from __future__ import annotations

import copy
import hashlib

import pytest
from sqlalchemy import JSON, Column, DateTime, Float, Integer, MetaData, String, Table, Text, create_engine, select

from learn_from import metadata_catalog as catalog_module
from learn_from import metadata_v2 as v2


@pytest.fixture
def setup_runner(monkeypatch):
    engine = create_engine('sqlite+pysqlite:///:memory:')
    db = MetaData()
    runs = Table('extraction_runs', db,
        Column('run_id', Integer, primary_key=True), Column('content_id', Integer), Column('media_id', String),
        Column('profile_version_id', Integer), Column('release_version', String), Column('run_mode', String),
        Column('status', String), Column('extractor_model', String), Column('resolver_model', String),
        Column('learner_model', String), Column('manifest', JSON), Column('raw_output', JSON), Column('resolved_output', JSON),
        Column('input_characters', Integer), Column('chunk_count', Integer), Column('error', Text), Column('completed_at', DateTime))
    Table('transcript_analysis_versions', db, Column('id', Integer, primary_key=True), Column('content_id', Integer),
        Column('run_id', Integer), Column('normalizer_version', String), Column('analysis_text', Text), Column('segment_map', JSON))
    Table('extraction_observations', db, Column('id', Integer, primary_key=True), Column('run_id', Integer),
        Column('content_id', Integer), Column('dimension', String), Column('raw_value', String), Column('normalized_value', String),
        Column('canonical_value', String), Column('evidence_text', Text), Column('start_seconds', Float), Column('end_seconds', Float),
        Column('confidence', Float), Column('resolution_status', String), Column('resolution_basis', String), Column('details', JSON))
    Table('doc_reference_mentions', db, Column('id', Integer, primary_key=True))
    db.create_all(engine)
    text = 'We sell a productized service to local plumbers. Start by finding five customers.'
    context = v2.ContentContext(1, 'video-1', 'youtube', 'channel', 'channel-1', 'Example channel', None, None,
        'Useful business', None, text, [{'text': text, 'start_seconds': 12.0, 'duration_seconds': 10.0}], 0)
    config = v2.default_profile_config()
    config['metadata_catalog'] = {'category_id': 10}
    profile = v2.EffectiveProfile(1, (1,), (1,), 'manual', config)
    field = {'schema_id': 7, 'key': 'business_model', 'type': 'string', 'description': 'How the business earns revenue',
        'policy': 'expandable', 'known_values': [{'value_id': 1, 'value': 'productized_service', 'label': 'Productized service'}],
        'provisional_values': [], 'aliases': [], 'counts': {'known': 1, 'provisional': 0}, 'has_more': False,
        'known_has_more': False, 'value_offset': 0, 'openai': {'eligible': True, 'key': 'business_model'}}
    catalog = {'version': 'catalog-1', 'fields': [field]}
    state = {'context': context, 'profile': profile, 'catalog': catalog, 'candidate_calls': []}
    monkeypatch.setattr(v2, 'load_content_context', lambda **kw: state['context'])
    monkeypatch.setattr(v2, 'load_effective_profile', lambda **kw: state['profile'])
    monkeypatch.setattr(v2, 'load_value_aliases', lambda **kw: {})
    monkeypatch.setattr(catalog_module, 'get_metadata_schema', lambda **kw: state['catalog'])
    monkeypatch.setattr(catalog_module, 'persist_metadata_candidates', lambda **kw: state['candidate_calls'].append(kw))
    monkeypatch.setattr(v2, '_persist_learning_output', lambda **kw: pytest.fail('No automatic learner persistence'))
    monkeypatch.setattr(v2, '_apply_corroborated_aliases', lambda **kw: pytest.fail('No automatic alias promotion'))
    def row(run_id):
        with engine.connect() as connection:
            return connection.execute(select(runs).where(runs.c.run_id == run_id)).mappings().one()
    state.update(engine=engine, runs=runs, row=row)
    return state


def extraction():
    return {'category': None, 'topics': [], 'keywords': [], 'participants': [], 'entities': [], 'references': [],
        'category_attributes': [{'key': 'business_model', 'value': 'productized service',
            'evidence': 'We sell a productized service to local plumbers.', 'confidence': 0.95}]}


def metered_request(calls):
    def request(**kwargs):
        calls.append(kwargs)
        v2._capture_response_usage(response={'usage': {'input_tokens': 1000,
            'input_tokens_details': {'cached_tokens': 100, 'cache_write_tokens': 700},
            'output_tokens': 100, 'total_tokens': 1100}}, model=kwargs['model'], requested_service_tier=kwargs['service_tier'])
        return extraction()
    return request


def run(state, **kwargs):
    return v2.run_index_metadata_for_content(engine=state['engine'], content_id=1, openai_api_key='test-key', **kwargs)


def test_new_runner_persists_normalized_evidence_and_reuses_without_model_charge(setup_runner):
    state = setup_runner
    calls = []
    result = run(state, requester=metered_request(calls))
    assert result['reused'] is False
    assert calls[0]['model'] == 'gpt-6-luna' and calls[0]['reasoning_effort'] == 'none'
    assert calls[0]['store'] is False and calls[0]['max_output_tokens'] == 5000
    assert result['index_metadata']['openai_attributes'] == {'business_model': 'productized_service'}
    assert result['field_results'][0]['status'] == 'trusted'
    assert result['field_results'][0]['start_seconds'] == 12.0
    assert result['index_metadata']['source_hash'] == hashlib.sha256(state['context'].transcript_text.encode()).hexdigest()
    row = state['row'](result['run_id'])
    assert row['status'] == 'completed'
    assert row['resolved_output']['index_metadata'] == result['index_metadata']
    assert row['manifest']['resolver']['invoked'] is False and row['manifest']['learner']['invoked'] is False
    assert row['manifest']['usage']['cache_write_tokens'] == 700
    # Proposals and volatile counters do not invalidate a completed extraction.
    state['catalog']['fields'][0].update(counts={'known': 1, 'provisional': 2000}, has_more=True,
        provisional_values=[{'value': 'a_new_value'}])
    cached = run(state, requester=lambda **kw: pytest.fail('Cache must not call the model'))
    assert cached['reused'] and cached['run_id'] == result['run_id']
    assert cached['usage']['estimated_cost'] == 0
    assert cached['original_usage']['estimated_cost'] > 0
    assert len(calls) == 1


def test_source_profile_prompt_and_catalog_changes_invalidate_cache(setup_runner, monkeypatch):
    state = setup_runner
    calls = []
    request = metered_request(calls)
    first = run(state, requester=request)
    state['context'] = v2.replace(state['context'], transcript_text=state['context'].transcript_text + ' More evidence.')
    second = run(state, requester=request)
    assert second['run_id'] != first['run_id']
    state['profile'].config['prompt_instructions'] = 'Additional constraint.'
    third = run(state, requester=request)
    assert third['run_id'] != second['run_id']
    state['catalog']['version'] = 'catalog-2'
    fourth = run(state, requester=request)
    assert fourth['run_id'] != third['run_id']
    original = v2._v2_prompt_contract
    monkeypatch.setattr(v2, '_v2_prompt_contract', lambda: {**original(), 'version': 'new-prompt-version'})
    fifth = run(state, requester=request)
    assert fifth['run_id'] != fourth['run_id']
    assert len(calls) == 5


def test_full_source_is_read_when_saved_segments_are_incomplete(setup_runner):
    state = setup_runner
    state['context'] = v2.replace(state['context'], transcript_text=state['context'].transcript_text + ' ' + 'important evidence ' * 2000 + ' FINAL_SOURCE_MARKER')
    calls = []
    result = run(state, requester=metered_request(calls))
    assert len(calls) > 1
    assert 'FINAL_SOURCE_MARKER' in calls[-1]['input_text']
    assert state['row'](result['run_id'])['manifest']['index_metadata']['coverage']['timing_available'] is False
    assert result['field_results'][0]['start_seconds'] is None


def test_whole_run_budget_and_catalog_bounds_fail_before_any_provider_call(setup_runner):
    state = setup_runner
    def forbidden(**kw):
        pytest.fail('Preflight must reject before model call')
    with pytest.raises(v2.MetadataV2Error, match='complete-run estimate'):
        run(state, requester=forbidden, max_cost_usd=0.0000001)
    state['catalog']['fields'][0]['known_has_more'] = True
    with pytest.raises(v2.MetadataV2Error, match='catalog is incomplete'):
        run(state, requester=forbidden)
    with state['engine'].connect() as connection:
        assert connection.execute(select(state['runs'])).first() is None


def test_failed_run_retains_usage_and_blocks_duplicate_charge_unless_forced(setup_runner):
    state = setup_runner
    def malformed(**kwargs):
        metered_request([])(**kwargs)
        return {'private_data': 'secret-test-key'}
    with pytest.raises(v2.MetadataV2Error) as raised:
        run(state, requester=malformed)
    row = state['row'](raised.value.run_id)
    assert row['status'] == 'failed'
    assert row['manifest']['usage']['estimated_cost'] > 0
    assert row['manifest']['usage']['requests_started'] == 1
    assert row['manifest']['usage']['unreported_requests'] == 0
    assert 'secret-test-key' not in row['error'] and 'private_data' not in row['error']
    with pytest.raises(v2.MetadataV2Error, match='no model request made'):
        run(state, requester=lambda **kw: pytest.fail('No retry without force'))
    result = run(state, requester=metered_request([]), force=True)
    assert result['run_id'] != row['run_id'] and not result['reused']


def test_unknown_provider_outcome_keeps_reservation_without_fabricating_zero_cost(setup_runner):
    state = setup_runner
    def unavailable(**kwargs):
        raise TimeoutError('secret-test-key source text must not be logged')
    with pytest.raises(v2.MetadataV2Error) as raised:
        run(state, requester=unavailable)
    usage = state['row'](raised.value.run_id)['manifest']['usage']
    assert usage['estimated_cost'] is None
    assert usage['unreported_requests'] == 1
    assert usage['reserved_cost_usd'] > 0


def test_request_payload_bounds_and_incomplete_response_usage_are_preserved(monkeypatch):
    payloads = []
    def post(**kwargs):
        payloads.append(kwargs['payload'])
        return {'status': 'incomplete', 'incomplete_details': {'reason': 'max_output_tokens'}, 'usage': {'input_tokens': 50, 'output_tokens': 100, 'total_tokens': 150}}
    monkeypatch.setattr(v2, '_post_json', post)
    events = []
    token = v2._USAGE_COLLECTOR.set(events)
    try:
        with pytest.raises(Exception, match='did not complete'):
            v2._request_openai_structured(api_key='test', model='gpt-6-luna', reasoning_effort='none', instructions='i',
                input_text='source', schema_name='test', schema={'type': 'object'}, timeout_seconds=10,
                service_tier='default', max_output_tokens=1000, store=False, prompt_cache_ttl='30m')
    finally:
        v2._USAGE_COLLECTOR.reset(token)
    assert payloads[0]['store'] is False and payloads[0]['truncation'] == 'disabled'
    assert payloads[0]['max_output_tokens'] == 1000
    assert payloads[0]['prompt_cache_options'] == {'ttl': '30m'}
    assert events[0]['response_status'] == 'incomplete'
    assert events[0]['incomplete_reason'] == 'max_output_tokens'
    assert events[0]['total_tokens'] == 150


def test_incomplete_failure_diagnostics_and_metering_are_durable(setup_runner, monkeypatch):
    state = setup_runner
    def incomplete(**kwargs):
        assert kwargs['payload']['max_output_tokens'] == 5000
        assert kwargs['payload']['prompt_cache_options'] == {'ttl': '30m'}
        return {'status': 'incomplete', 'incomplete_details': {'reason': 'max_output_tokens'},
                'usage': {'input_tokens': 500, 'output_tokens': 5000, 'total_tokens': 5500}}
    monkeypatch.setattr(v2, '_post_json', incomplete)
    with pytest.raises(v2.MetadataV2Error) as raised:
        run(state)
    event = state['row'](raised.value.run_id)['manifest']['usage']['events'][0]
    assert event['response_status'] == 'incomplete'
    assert event['incomplete_reason'] == 'max_output_tokens'
    assert event['estimated_cost'] > 0


def test_http_failure_status_survives_wrapping_without_body_or_message(setup_runner):
    from io import BytesIO
    from urllib.error import HTTPError
    state = setup_runner
    def unavailable(**kwargs):
        try:
            raise HTTPError('https://api.openai.com/v1/responses', 503, 'secret-upstream-message', {}, BytesIO(b'secret-response-body'))
        except HTTPError as cause:
            raise v2.MetadataExtractionError('secret-wrapped-provider-body') from cause
    with pytest.raises(v2.MetadataV2Error) as raised:
        run(state, requester=unavailable)
    row = state['row'](raised.value.run_id)
    assert row['manifest']['index_metadata']['failure'] == {'kind': 'http_error', 'http_status': 503}
    assert 'HTTP 503' in row['error']
    assert 'secret-' not in str(row['manifest']) + row['error']
    assert row['manifest']['usage']['unreported_requests'] == 1


def test_transport_failure_classification_does_not_copy_url_error_reason():
    from urllib.error import URLError
    assert v2._safe_metadata_failure(URLError(TimeoutError('private endpoint')))['kind'] == 'timeout'
    result = v2._safe_metadata_failure(URLError('secret URL and credentials'))
    assert result == {'kind': 'network_error', 'http_status': None}
    assert 'secret' not in str(result)
