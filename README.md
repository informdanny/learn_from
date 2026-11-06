# Learn From

A Python CLI for collecting selected YouTube transcripts, indexing saved sources with OpenAI managed retrieval, and consulting published material with source references. Original transcripts and durable job checkpoints remain in PostgreSQL.

## Cloud worker

This snapshot runs an existing collection job once, optionally continues its hosted index, and exits. It does not schedule monitoring. The Dockerfile installs the package and starts `learn-from collection-worker`; `railway.toml` requests one replica with no automatic restart.

Configure these variables through the deployment platform's secret manager:

- `DATABASE_URL`: an existing PostgreSQL database with the application's schema.
- `YOUTUBE_API_KEY`: YouTube Data API access.
- `YOUTUBE_TRANSCRIPT_IO_API_KEY`: transcript provider access.
- `OPENAI_API_KEY`: required when hosted indexing is enabled.
- `LEARN_FROM_COLLECTION_JOB_ID`: an existing saved collection job.
- `LEARN_FROM_INDEX_AFTER_COLLECTION`: `true` or `false`; defaults to `true`.
- `LEARN_FROM_INDEX_BEFORE_COLLECTION`: defaults to `false`; enable to index saved sources before collecting more. Pending or unresolved indexing stops the worker before acquisition.
- `LEARN_FROM_MAX_INDEX_FILES`: maximum files considered in this invocation; defaults to `1000`.

The saved collection determines the selected videos and provider-request cap. Reusable transcripts avoid acquisition calls. Reservations and uncertain outcomes survive process interruptions. Hosted index storage can incur ongoing charges independently of worker execution.

This deployment is a worker, not a public web application. Do not expose the local Corpus Lens explorer without an appropriate authentication layer. Database bootstrap and migrations are managed separately from this deployment snapshot.

## Local checks

Install the package and test dependency in a virtual environment, then run:

```bash
pip install . pytest
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
learn-from --help
```

The `consult --prepare-only` option exports complete selected source evidence for review without making a Responses API model call. Hosted search may still be used when sources are selected by a collection job.

Hosted consultations with multiple `--channel` values search each channel and reserve one complete transcript per channel. Missing evidence is reported without generating a comparison. Optional extra sources may be omitted to fit the full-document context limit; those omissions are recorded. The configured OpenAI key must have access to both the saved vector store and the answer model.

Tests use isolated fixtures and mocked external calls. No credentials are included. The standalone transcript skill, private planning notes, provider payloads, outputs, and historical migration tests are outside this runtime snapshot.
