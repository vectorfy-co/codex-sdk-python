# Codex SDK Python Changelog

This file tracks SDK-level changes. Keep the newest changes at the top.

## [0.104.0] - 2026-02-18

### Added
- `AppServerClient.thread_list(...)` now accepts `cwd` for scoped thread queries.
- `AppServerClient.model_list(...)` now accepts `include_hidden` to request hidden models.
- `AppServerClient.skills_remote_read(...)` now supports explicit `cwds`, `enabled`,
  `hazelnut_scope`, and `product_surface` filters and adds typed request helpers.
- `skills/remote/write` typed payload helper to keep optional legacy
  `is_preload` compatibility while supporting current protocol usage.

### Updated
- Bundled Codex CLI vendor binaries updated to `0.104.0` via `scripts/setup_binary.py`.
- SDK package version bumped to `0.104.0` (`pyproject.toml`, `codex_sdk.__version__`,
  README release badge).
- App-server method coverage aligned with upstream `rust-v0.104.0` schema additions
  (including thread list/model list/remote skills parameters).

### Notes
- Upstream `rust-v0.104.0` introduces thread archived/unarchived notifications and
  command approval IDs; the SDK remains compatible because app-server notifications
  and approval payloads are passed through as structured dictionaries.

## [0.101.0] - 2026-02-15

### Added
- New app-server wrappers for recently exposed endpoints, including
  `thread_unarchive`, `thread_name_set`, `thread_compact_start`,
  `thread_background_terminals_clean`, `turn_steer`, `experimental_feature_list`,
  `account_chatgpt_auth_tokens_refresh`, `skills_config_write`,
  `skills_remote_read`/`skills_remote_write`, `item_tool_call`,
  `item_tool_request_user_input`, `item_command_execution_request_approval`,
  `item_file_change_request_approval`, and `mock_experimental_method`.
- New `tool_envelope` module for shared tool-call envelope parsing, normalization,
  and schema validation utilities used by model-driven tool loops.
- Pydantic-AI model provider hardening for request/usage/streaming compatibility
  across API variants, with expanded integration coverage for robust tool-call flows.

### Updated
- Bundled Codex CLI vendor binaries updated to 0.101.0 via `scripts/setup_binary.py`.
- SDK version set to 0.101.0 to match the latest published Codex CLI stable release.
- `scripts/setup_binary.py` now supports the current npm packaging model that ships
  platform-specific `@openai/codex@<version>-<target>` artifacts.

### Notes
- Upstream `0.103.0` was not available on public GitHub/npm release channels during this
  update run; the sync is pinned to the latest published stable (`0.101.0`).

## [0.91.0] - 2026-01-27

### Added
- Thread option `connectors_enabled` to toggle `features.connectors`.
- App-server helper `app_list` for `app/list`.

### Updated
- Enforced `max_threads` cap of 6 to match Codex 0.91.0 sub-agent limits.
- Bundled Codex CLI vendor binaries updated to 0.91.0 via `scripts/setup_binary.py`.
- SDK version set to 0.91.0 to match Codex CLI release.

### Notes
- Codex 0.91.0 reduces the maximum number of sub-agents to 6 and adds app listings
  in the app-server protocol.

## [0.89.0] - 2026-01-22

### Added
- App-server helper for `thread_read` (supports `include_turns`).
- `thread_list` now accepts an `archived` filter.
- `config_read` accepts an optional `cwd` for layered config resolution.

### Updated
- Bundled Codex CLI vendor binaries updated to 0.89.0 via `scripts/setup_binary.py`.
- SDK version set to 0.89.0 to match Codex CLI release.

### Notes
- Codex 0.89.0 adds `/permissions`, skill enable/disable UI, and app-server support for
  `thread/read` and layered `config/read`.

## [0.88.0] - 2026-01-22

### Added
- Thread options for `model_instructions_file`, `model_personality`, `max_threads`,
  `collaboration_modes_enabled`, and `responses_websockets_enabled`.
- App-server helper for `collaborationMode/list` to fetch collaboration mode presets.

### Updated
- Bundled Codex CLI vendor binaries updated to 0.88.0 via `scripts/setup_binary.py`.
- SDK version set to 0.88.0 to match Codex CLI release.
- README updated with new ThreadOptions mappings and collaboration mode list helper.

### Notes
- Codex 0.88.0 adds device-code auth as a headless fallback and tightens config loading to
  trusted folders (including symlink resolution).
- Collaboration modes/presets, request-user-input tooling, and model personality/instruction
  file config landed in the CLI/core stack.

## [0.87.0] - 2026-01-17

### Added
- App-server input normalization now accepts `text_elements`/`byte_range` for text items and
  converts them to camelCase (`textElements`/`byteRange`) for the JSON-RPC protocol.

### Updated
- Bundled Codex CLI vendor binaries updated to 0.87.0 via `scripts/setup_binary.py`.
- SDK version set to 0.87.0 to match Codex CLI release.
- README updated with app-server text element input normalization notes.

### Notes
- Codex 0.87.0 round-trips user message text element metadata (byte ranges) through the
  protocol/app-server/core stack.
- MCP `CallToolResult` now includes `threadId` in both `content` and `structuredContent`.
- Collaboration wait calls can block on multiple receiver IDs.
- Piped non-PTY commands no longer hang waiting on stdin; shell commands run under user snapshots.

## [0.86.0] - 2026-01-16

### Added
- App-server `skills_list` now returns optional `interface` metadata when provided by
  `SKILL.toml` (display name, icons, brand color, default prompt).

### Updated
- Bundled Codex CLI vendor binaries updated to 0.86.0 via `scripts/setup_binary.py`.
- SDK version set to 0.86.0 to match Codex CLI release.

### Notes
- Codex 0.86.0 can explicitly disable web search and advertises eligibility via a header.
- MCP elicitation accept now sends an empty JSON payload instead of null for stricter servers.
- Unified exec cleans up background processes to avoid late End events after listeners stop.

## [0.85.0] - 2026-01-15

### Added
- `web_search_mode` thread option (`disabled`, `cached`, `live`) mapped to `--config web_search=...`.
- PydanticAI model provider now supports streamed responses and includes tool metadata in the
  prompt (kind/strict/timeout/metadata).

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

## [0.81.0] - 2026-01-15

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

## [0.80.0] - 2026-01-11

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
