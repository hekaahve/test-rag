# Roadmap

## Documents and indexing

- Support PDF, DOCX, TXT, and Markdown files.
- Store file metadata: source, version, uploader, and upload timestamp.
- Add OCR support for scanned PDFs.
- Detect duplicates with a content hash.
- Re-index only changed sections on document update.
- Add configurable chunking and metadata filtering.
- Add hybrid vector and keyword retrieval, relevance thresholds, and reranking.

## Answer quality and evaluation

- Include source citations and quoted snippets in responses.
- Explicitly state uncertainty when the answer is not in the sources.
- Standardize responses as short answer, rationale, and sources.
- Support Finnish and English queries with confidence gating.
- Store an evaluation question bank with expected answers.
- Measure accuracy, coverage, and source grounding; publish reports and prevent regressions.
- Log incorrect answers for continuous improvement.

## Security and operations

- Use private Blob access and RBAC; keep secrets in Azure Key Vault.
- Detect and mask PII before indexing.
- Add audit logs and support deletion requests.
- Add document status and re-index administration features.
- Add response latency and source-document reporting.
- Add centralized logs, metrics, Azure-call retries, and circuit breakers.
- Define scalability targets and release acceptance criteria.
- Add CI tests, linting, evaluation, unit coverage targets, integration tests, and versioned indexes.
