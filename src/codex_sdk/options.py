"""Configuration options for the Codex SDK."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Mapping, Optional, Union

from .abort import AbortSignal

ApprovalMode = Literal["never", "on-request", "on-failure", "untrusted"]

SandboxMode = Literal["read-only", "workspace-write", "danger-full-access"]

ModelReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh"]
WebSearchMode = Literal["disabled", "cached", "live"]
ModelPersonality = Literal["friendly", "pragmatic", "none"]


@dataclass
class CodexOptions:
    """Options for configuring the Codex client.

    Attributes:
        codex_path_override: Override the path to the codex binary.
        base_url: Base URL for the API.
        api_key: API key for authentication (sets CODEX_API_KEY for the child process).
        env: Environment variables passed to the Codex CLI process. When provided, the SDK
            will not inherit variables from os.environ.
        config_overrides: Optional config overrides passed as `--config key=value` to the Codex
            CLI. Values are encoded as TOML literals; for complex overrides, use strings.
    """

    # Override the path to the codex binary
    codex_path_override: Optional[str] = None

    # Base URL for the API
    base_url: Optional[str] = None

    # API key for authentication
    api_key: Optional[str] = None

    # Environment variables passed to the Codex CLI process.
    env: Optional[Mapping[str, str]] = None

    # Optional config overrides passed to the Codex CLI process.
    config_overrides: Optional[Mapping[str, Any]] = None


@dataclass
class ThreadOptions:
    """Options for configuring a thread.

    Attributes:
        model: Model to use for the thread.
        sandbox_mode: Sandbox mode for the thread.
        working_directory: Working directory for the thread.
        skip_git_repo_check: Skip Git repository safety check.
        model_reasoning_effort: Model reasoning effort preset.
        model_instructions_file: Optional path to a file containing model instructions.
        model_personality: Optional personality preset for the model.
        max_threads: Maximum number of concurrent threads allowed by the agent
            (defaults to the CLI's configured value; the CLI currently defaults to 6).
        network_access_enabled: Enable/disable network access in workspace-write sandbox.
        web_search_mode: Web search mode ("disabled", "cached", or "live").
        web_search_enabled: Enable/disable web search feature.
        web_search_cached_enabled: Enable/disable cached web search feature.
        skills_enabled: (Deprecated) Skills are always enabled in Codex 0.80+.
        shell_snapshot_enabled: Enable/disable shell snapshotting.
        background_terminals_enabled: Enable/disable background terminals (unified exec).
        apply_patch_freeform_enabled: Enable/disable freeform apply_patch tool.
        exec_policy_enabled: Enable/disable exec policy enforcement.
        remote_models_enabled: Enable/disable remote model list refresh.
        collaboration_modes_enabled: Enable/disable collaboration mode tooling.
        connectors_enabled: Enable/disable connector tooling.
        responses_websockets_enabled: Enable/disable responses websocket transport.
        request_compression_enabled: Enable/disable request body compression.
        feature_overrides: Arbitrary feature flag overrides (key -> bool).
        approval_policy: Approval policy for tool execution.
        additional_directories: Additional directories to add to the sandbox.
        config_overrides: Optional config overrides passed as `--config key=value` for this
            thread's invocations. Values are encoded as TOML literals.
    """

    # Model to use for the thread
    model: Optional[str] = None

    # Sandbox mode for the thread
    sandbox_mode: Optional[SandboxMode] = None

    # Working directory for the thread
    working_directory: Optional[str] = None

    # Skip Git repository check
    skip_git_repo_check: Optional[bool] = None

    # Model reasoning effort preset
    model_reasoning_effort: Optional[ModelReasoningEffort] = None

    # Optional path to a file containing model instructions
    model_instructions_file: Optional[Union[str, Path]] = None

    # Optional personality preset for the model
    model_personality: Optional[ModelPersonality] = None

    # Maximum number of concurrent threads allowed by the agent
    max_threads: Optional[int] = None

    # Enable/disable network access in workspace-write sandbox
    network_access_enabled: Optional[bool] = None

    # Web search mode ("disabled", "cached", or "live")
    web_search_mode: Optional[WebSearchMode] = None

    # Enable/disable web search feature
    web_search_enabled: Optional[bool] = None

    # Enable/disable cached web search feature
    web_search_cached_enabled: Optional[bool] = None

    # (Deprecated) Skills are always enabled in Codex 0.80+
    skills_enabled: Optional[bool] = None

    # Enable/disable shell snapshotting
    shell_snapshot_enabled: Optional[bool] = None

    # Enable/disable background terminals (unified exec)
    background_terminals_enabled: Optional[bool] = None

    # Enable/disable freeform apply_patch tool
    apply_patch_freeform_enabled: Optional[bool] = None

    # Enable/disable exec policy enforcement
    exec_policy_enabled: Optional[bool] = None

    # Enable/disable remote model list refresh
    remote_models_enabled: Optional[bool] = None

    # Enable/disable collaboration mode tooling
    collaboration_modes_enabled: Optional[bool] = None

    # Enable/disable connector tooling
    connectors_enabled: Optional[bool] = None

    # Enable/disable responses websocket transport
    responses_websockets_enabled: Optional[bool] = None

    # Enable/disable request body compression
    request_compression_enabled: Optional[bool] = None

    # Arbitrary feature flag overrides (feature key -> bool)
    feature_overrides: Optional[Mapping[str, bool]] = None

    # Approval policy for tool execution
    approval_policy: Optional[ApprovalMode] = None

    # Additional directories to add to the sandbox
    additional_directories: Optional[List[str]] = None

    # Optional config overrides passed to the Codex CLI process.
    config_overrides: Optional[Mapping[str, Any]] = None


@dataclass
class TurnOptions:
    """Options for configuring a turn.

    Attributes:
        output_schema: JSON schema describing the expected agent output as a JSON object.
        signal: Abort signal to cancel the turn.
    """

    # JSON schema describing the expected agent output
    output_schema: Optional[Any] = None

    # Abort signal to cancel the turn
    signal: Optional[AbortSignal] = None
