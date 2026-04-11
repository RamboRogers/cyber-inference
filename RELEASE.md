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
- Admin-adjustable model load timeout, defaulting to 300 seconds.
- Admin-configurable pre-model load command for Thor/DGX Spark cache clearing, defaulting to
  `sudo sysctl -w vm.drop_caches=3` and disabled by default.

### Changed
- `ModelManager.download_model()` now returns a `ModelDownloadResult` carrying the canonical model
  name, primary path, aggregate size, and shard metadata.
- Admin download responses now look up the canonical registered model name returned by the backend.
- CLI GGUF downloads now report the resolved local model path from `ModelDownloadResult`.
- Backend startup waits now use the configured model load timeout across llama.cpp, whisper.cpp, and
  transformers servers.
- Model startup can now run the configured pre-load command before launching the backend server when
  enabled from admin-initiated loads. Public `/v1` lazy-loads skip the hook in this release.

### Fixed
- Timed-out backend startups now terminate the launched process, release the allocated port, and
  avoid leaving stale process tracking behind.

### Verification
- `uv run pytest` passed with 103 tests.
- Scoped ruff checks on changed implementation and tests passed.
- `uv run mypy src/cyber_inference/services/model_manager.py --ignore-missing-imports` passed.
