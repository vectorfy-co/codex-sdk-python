"""Custom exceptions surfaced by the Codex Python SDK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class CodexError(Exception):
    """Base exception for Codex SDK errors."""


@dataclass
class CodexCLIError(CodexError):
    """Raised when the Codex CLI exits with a non-zero status code."""

    exit_code: int
    stderr: str

    def __post_init__(self) -> None:
        message = f"Codex CLI exited with code {self.exit_code}"
        if self.stderr:
            message = f"{message}: {self.stderr}"
        super().__init__(message)


class CodexParseError(CodexError):
    """Raised when the SDK cannot parse CLI output."""


class CodexAbortError(CodexError):
    """Raised when an operation is aborted via an AbortSignal."""


class TurnFailedError(CodexError):
    """Raised when a turn ends with a `turn.failed` event."""

    def __init__(self, message: str, *, error: Optional[object] = None) -> None:
        super().__init__(message)
        self.error = error


@dataclass
class CodexAppServerError(CodexError):
    """Raised when the Codex app-server returns a JSON-RPC error response."""

    code: int
    message: str
    data: Optional[object] = None

    def __post_init__(self) -> None:
        detail = f"Codex app-server error {self.code}: {self.message}"
        if self.data is not None:
            detail = f"{detail} ({self.data})"
        super().__init__(detail)
