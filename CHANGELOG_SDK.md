# Codex SDK Python Changelog

This file tracks SDK-level changes. Keep the newest changes at the top.

## [0.117.0] - 2026-03-28

### Added
- App-server helpers for newly exposed upstream RPCs:
  `thread_shell_command`, `thread_background_terminals_clean`,
  `experimental_feature_enablement_set`, `fs_watch`, and `fs_unwatch`.
- `CodexModel(..., hooks=ThreadHooks(...))` now dispatches typed SDK thread hooks for
  app-server-backed PydanticAI requests, including thread start, turn start/completion,
  turn failure, and item lifecycle events.

### Updated
- `AppServerOptions` now supports `opt_out_notification_methods`, wiring
  `initialize.capabilities.optOutNotificationMethods` for app-server clients that
  want to suppress selected notification streams.
- The PydanticAI model-provider path now runs through upstream
  `Model.prepare_request()`, keeping request customization, output-mode handling,
  builtin-tool validation, and thinking resolution aligned with PydanticAI `1.73.0`.
- `CodexModel` now maps PydanticAI `model_settings["thinking"]` onto Codex
  `model_reasoning_effort` when no explicit thread-level reasoning effort is set.
- PydanticAI dependency support updated to `>=1.73.0,<2`, with the dev dependency
  pinned to `1.73.0` for reproducible test coverage against the latest release.
- `thread_list` now supports the upstream `cwd` and `search_term` filters.
- `plugin_list` now supports `force_remote_sync`, `config_batch_write` now supports
  `reload_user_config`, and `windows_sandbox_setup_start` now accepts an optional
  workspace `cwd`.
- README and PydanticAI examples updated to show typed output models, runtime hooks,
  and per-run thinking configuration.

### Notes
- Codex 0.116.0 and 0.117.0 introduced app-server support for thread-scoped `!`
  shell commands, filesystem watch subscriptions, richer plugin sync flows, and
  per-connection notification suppression.

## [0.115.1] - 2026-03-17

### Updated
- Refreshed the locked optional `pydantic-ai` dependency graph to upgrade the
  transitive `PyJWT` dependency to `2.12.1`, addressing the open GitHub
  security advisory on `uv.lock` without changing the SDK API surface.
- Added explicit least-privilege `GITHUB_TOKEN` permissions to the CI workflow
  so the open CodeQL workflow-permissions alerts no longer rely on repository
  defaults.

## [0.115.0] - 2026-03-17

### Added
- App-server helpers for newly exposed 0.115.0 filesystem RPCs:
  `fs_copy`, `fs_create_directory`, `fs_get_metadata`, `fs_read_directory`,
  `fs_read_file`, `fs_remove`, and `fs_write_file`.
- App-server helper `plugin_read` for fetching plugin metadata from a marketplace
  before install.

### Updated
- `CollabToolCallItem` now reflects the upstream collaboration payload by exposing
  optional `model` and `reasoning_effort` fields for spawned agents.
- Collaboration item parsing now accepts the renamed `wait_agent` tool and the
  new `interrupted` collaboration agent status while keeping legacy `wait`
  compatibility for older threads.
- The typed SDK surface now exposes `model_reasoning_effort="none"`,
  `approval_policy="granular"`, and `approvals_reviewer` so the CLI-backed
  thread API and `CodexModel` match the upstream 0.115.0 thread-start contract.
- README app-server method coverage and release version updated to 0.115.0.

### Notes
- Codex 0.115.0 adds experimental filesystem app-server RPCs, `plugin/read`,
  guardian approval review flows, granular approval routing, and the `wait_agent`
  collaboration tool rename.

## [0.114.1] - 2026-03-13

### Updated
- Refreshed the PydanticAI integration docs/examples to use `gpt-5.4` for the
  general OpenAI model examples, matching the current official model guidance.
- Added a matching regression assertion in the PydanticAI provider test coverage.
- `scripts/setup_binary.py` now falls back to the latest published
  compatible `@openai/codex-sdk` package from the same major/minor release line
  when a Python-only patch release version has not been published to npm yet,
  which fixes CI for SDK-only patch releases without silently jumping to a newer
  release line.
- Added a repo-managed pre-push hook installer and CI-check runner so local
  pushes can execute the same validation flow as GitHub Actions, using a
  non-mutating vendor verification step instead of rewriting checked-in binaries.

## [0.114.0] - 2026-03-13

### Added
- App-server helpers for newly exposed protocol methods:
  `thread_metadata_update`, `plugin_list`, `plugin_install`, `plugin_uninstall`,
  `command_exec_write`, `command_exec_resize`, and `command_exec_terminate`.
- Root-level collaboration exports for the `collab_tool_call` item family:
  `CollabToolCallItem`, `CollabToolCallStatus`, `CollabTool`, `CollabAgentStatus`,
  and `CollabAgentState`.

### Updated
- `command_exec` now accepts the current 0.114.0 app-server parameters for interactive
  sessions and output streaming: `disable_output_cap`, `disable_timeout`, `env`,
  `output_bytes_cap`, `process_id`, `size`, `stream_stdin`, `stream_stdout_stderr`,
  and `tty`.
- `ApprovalDecisions`/`AppServerTurnSession` can now auto-handle
  `item/permissions/requestApproval` requests via `permissions_request`.
- `CodexModel` now streams live PydanticAI text/tool updates from app-server notifications when
  Codex emits valid incremental envelope state, while keeping `streamed.get()` aligned with the
  canonical final turn response.
- The PydanticAI integration now targets the current release line only
  (`pydantic-ai>=1.68.0,<2`) and uses the modern streaming API surface directly.
- README app-server method coverage, collaboration export docs, and SDK version updated to 0.114.0.

### Notes
- Codex 0.114.0 adds plugin marketplace endpoints, thread metadata updates,
  streaming `command/exec` control methods, hook lifecycle notifications, and
  improved permissions handling in the app-server protocol.

## [0.107.0] - 2026-03-04

### Added
- App-server helpers for newly exposed protocol methods:
  `thread_unsubscribe`, `turn_steer`, `experimental_feature_list`,
  `external_agent_config_detect`, `external_agent_config_import`,
  `windows_sandbox_setup_start`.
- App/server wrappers for the renamed remote-skill endpoints:
  `skills_remote_list` and `skills_remote_export`.

### Updated
- `skills_remote_read` now maps to `skills/remote/list` and
  `skills_remote_write` maps to `skills/remote/export` (kept as backward-compatible aliases).
- `model_list` now accepts `include_hidden`.
- `app_list` now accepts `force_refetch` and `thread_id` to match upstream gating/cache controls.
- SDK version set to 0.107.0 to match Codex CLI release.

### Notes
- Codex 0.107.0 adds thread forking UX improvements, richer model/app availability metadata,
  configurable memories, and resume-state fixes for pending approval/input requests.

## [0.98.0] - 2026-02-05

### Added
- Exec event parsing for `collab_tool_call` items (collaboration tool calls).
- Exec event parsing for `web_search` `action` payloads.
- Thread option `model_personality="none"` (mirrors app-server personality support).
- App-server option `experimental_api_enabled` to opt into experimental methods/fields via `initialize.capabilities.experimentalApi=true`.
- App-server helpers: `thread_name_set`, `thread_unarchive`, `thread_compact_start`,
  `skills_remote_read`, `skills_remote_write`, and `skills_config_write`.
- `thread_list` now accepts `sort_key` and `source_kinds`.

### Updated
- Fixed `codex exec` argument ordering when resuming a thread with `--image` attachments
  (resume args now precede image args to avoid greedy flag parsing).
- `max_threads` validation now only enforces `>= 1` (Codex defaults to 6; this is not a hard cap).
- PydanticAI integration updated for `pydantic-ai` 0.6.x.
- Fixed `logfire` optional dependency to avoid shadowing Pydantic's Logfire packages in CI,
  which prevented `pydantic_ai` from importing and caused coverage failures.
- `scripts/setup_binary.py` now pins the npm download to `@openai/codex-sdk@<pyproject version>`
  so vendor binaries match the SDK version.
- Bundled Codex CLI vendor binaries updated to 0.98.0 via `scripts/setup_binary.py`.
- SDK version set to 0.98.0 to match Codex CLI release.

### Notes
- Codex 0.98.0 introduces GPT-5.3-Codex (model availability is controlled by your Codex provider).

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
