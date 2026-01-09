"""Configuration options for the Codex SDK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Literal, Mapping, Optional

from .abort import AbortSignal

ApprovalMode = Literal["never", "on-request", "on-failure", "untrusted"]

SandboxMode = Literal["read-only", "workspace-write", "danger-full-access"]

ModelReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh"]


@dataclass
class CodexOptions:
    """Options for configuring the Codex client.

    Attributes:
        codex_path_override: Override the path to the codex binary.
        base_url: Base URL for the API.
        api_key: API key for authentication (sets CODEX_API_KEY for the child process).
        env: Environment variables passed to the Codex CLI process. When provided, the SDK
            will not inherit variables from os.environ.
    """

    # Override the path to the codex binary
    codex_path_override: Optional[str] = None

    # Base URL for the API
    base_url: Optional[str] = None

    # API key for authentication
    api_key: Optional[str] = None

    # Environment variables passed to the Codex CLI process.
    env: Optional[Mapping[str, str]] = None


@dataclass
class ThreadOptions:
    """Options for configuring a thread.

    Attributes:
        model: Model to use for the thread.
        sandbox_mode: Sandbox mode for the thread.
        working_directory: Working directory for the thread.
        skip_git_repo_check: Skip Git repository safety check.
        model_reasoning_effort: Model reasoning effort preset.
        network_access_enabled: Enable/disable network access in workspace-write sandbox.
        web_search_enabled: Enable/disable web search feature.
        approval_policy: Approval policy for tool execution.
        additional_directories: Additional directories to add to the sandbox.
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

    # Enable/disable network access in workspace-write sandbox
    network_access_enabled: Optional[bool] = None

    # Enable/disable web search feature
    web_search_enabled: Optional[bool] = None

    # Approval policy for tool execution
    approval_policy: Optional[ApprovalMode] = None

    # Additional directories to add to the sandbox
    additional_directories: Optional[List[str]] = None


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
