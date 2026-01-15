# Codex SDK Python Changelog

This file tracks SDK-level changes. Keep the newest changes at the top.

## 0.85.0 (2026-01-15)

### Added
- `web_search_mode` thread option (`disabled`, `cached`, `live`) mapped to `--config web_search=...`.

### Updated
- Legacy `web_search_enabled`/`web_search_cached_enabled` now map to `web_search` for CLI
  compatibility.
- Bundled Codex CLI vendor binaries updated to 0.85.0 via `scripts/setup_binary.py`.
- README updated for web search mode configuration and safety defaults.
- SDK version set to 0.85.0 to match Codex CLI release.

### Notes
- Codex 0.85.0 app-server emits collaboration tool calls as item events, with richer agent
  controls (`spawn_agent` role presets and optional interrupt on `send_input`).
- `/models` metadata now includes upgrade migration markdown.
- Linux sandbox falls back to Landlock-only when user namespaces are unavailable.
- `codex resume --last` now respects the current working directory.
- Stdin prompt decoding handles BOMs/UTF-16 with clearer errors.

## 0.81.0 (2026-01-15)

### Added
- App-server helper for `config/mcpServer/reload` to refresh MCP server config.

### Updated
- Bundled Codex CLI vendor binaries updated to 0.81.0 via `scripts/setup_binary.py`.
- README updated with the new `mcp_server_refresh` convenience method.
- SDK version set to 0.81.0 to match Codex CLI release.

### Deprecated
- Python 3.8 and 3.9 support are deprecated and will be removed in a future release.

### Notes
- Codex 0.81.0 default model is now `gpt-5.2-codex`.
- Headless runs automatically switch to device-code login.
- Linux sandbox supports read-only bind mounts; app-server now emits `configWarning`
  notifications for config/rules parse errors.

## 0.80.0 (2026-01-11)

### Added
- App-server JSON-RPC client with initialize handshake, notifications, and request handling.
- App-server helpers for `thread/start`, `thread/resume`, `thread/fork`, `thread/loaded/list`,
  `thread/list`, `thread/archive`, `thread/rollback`, `config/read`, `config/value/write`,
  `config/batchWrite`, `skills/list`, `model/list`, `command/exec`, `review/start`,
  MCP auth/status, account endpoints, feedback upload, `configRequirements/read`,
  `turn/start`, and `turn/interrupt`.
- App-server turn session wrapper with `ApprovalDecisions` for auto-responding to approvals.
- Config override helpers to pass `--config key=value` to Codex CLI runs.
- `ThreadHooks` + `Thread.run_with_hooks()` for event callbacks during streamed turns.
- New examples for app-server usage (basic, fork, requirements, skill input, approvals),
  turn sessions, hooks, notify hooks, and config overrides.
- App-server error type (`CodexAppServerError`).
- Pytest `conftest.py` to make `uv run pytest` work without extra PYTHONPATH setup.
- Added `UPGRADE_CHECKLIST.md` to guide future release updates.
- GitHub Actions release workflow that creates GitHub releases from `CHANGELOG_SDK.md`.

### Updated
- Bundled Codex CLI vendor binaries updated to 0.80.0 via `scripts/setup_binary.py`.
- README updated with app-server usage, notify/OTEL notes, and config override examples.
- SDK version set to 0.80.0 to match Codex CLI release.
- Dev dependencies now include `pydantic` and `pydantic-ai` so integration tests run in `uv run pytest`.
- CI workflow now installs dev deps and enforces coverage in `pytest --cov=codex_sdk`.

### Deprecated
- `ThreadOptions.skills_enabled` is deprecated; skills are always enabled in Codex 0.80+ and
  the SDK no longer sends `features.skills`.

### Notes
- PydanticAI integrations were reviewed; no code changes required for 0.80.0.
