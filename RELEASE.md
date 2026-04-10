# Release Notes

## 0.2.0

### Added
- Split GGUF shard support for HuggingFace downloads.
- Logical repo-file grouping so shard sets appear as one model option in the downloader.
- Backend shard metadata for repo-file responses, including shard count, primary filename,
  aggregate size, completeness, and missing-shard information.
- Idempotent split-GGUF downloads that skip complete shards, fetch missing shards, redownload
  wrong-sized shards, and support forced redownloads.
- Graceful split-GGUF database upgrade behavior for legacy shard-named rows.
- Downloader UI metadata for split GGUF models.

### Changed
- `ModelManager.download_model()` now returns a `ModelDownloadResult` carrying the canonical model
  name, primary path, aggregate size, and shard metadata.
- Admin download responses now look up the canonical registered model name returned by the backend.
- CLI GGUF downloads now report the resolved local model path from `ModelDownloadResult`.

### Verification
- `uv run pytest` passed with 94 tests.
- Scoped ruff checks on changed implementation and tests passed.
- `uv run mypy src/cyber_inference/services/model_manager.py --ignore-missing-imports` passed.
