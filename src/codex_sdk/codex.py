"""Main Codex class for interacting with the Codex agent."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

from .exceptions import CodexError
from .exec import CodexExec
from .options import CodexOptions, ThreadOptions
from .thread import Thread


class Codex:
    """
    Codex is the main class for interacting with the Codex agent.

    Use start_thread() to start a new thread or resume_thread() to resume a previously
    started thread.
    """

    def __init__(self, options: Optional[CodexOptions] = None):
        """
        Create a Codex client.

        Args:
            options: Optional configuration for the Codex CLI invocation.
        """
        if options is None:
            options = CodexOptions()

        self._exec = CodexExec(options.codex_path_override, env=options.env)
        self._options = options

    def start_thread(self, options: Optional[ThreadOptions] = None) -> Thread:
        """
        Starts a new conversation with an agent.

        Args:
            options: Optional thread configuration options.

        Returns:
            A new thread instance configured with the provided options.
        """
        if options is None:
            options = ThreadOptions()

        return Thread(self._exec, self._options, options)

    def resume_thread(
        self, thread_id: str, options: Optional[ThreadOptions] = None
    ) -> Thread:
        """
        Resumes a conversation with an agent based on the thread id.
        Threads are persisted in ~/.codex/sessions.

        Args:
            thread_id: The id of the thread to resume.
            options: Optional thread configuration options.

        Returns:
            A thread instance bound to the existing session.
        """
        if options is None:
            options = ThreadOptions()

        return Thread(self._exec, self._options, options, thread_id)

    def resume_last_thread(
        self,
        options: Optional[ThreadOptions] = None,
        *,
        codex_home: Optional[str] = None,
    ) -> Thread:
        """
        Resume the most recently modified thread from the Codex sessions directory.
        """
        if options is None:
            options = ThreadOptions()

        sessions_root = _resolve_codex_home(codex_home) / "sessions"
        rollout = _find_latest_rollout(sessions_root)
        if rollout is None:
            raise CodexError(f"No Codex sessions found under {sessions_root}")

        thread_id = _extract_thread_id_from_rollout(rollout)
        if thread_id is None:
            raise CodexError(f"Failed to determine thread id from {rollout}")

        return self.resume_thread(thread_id, options)


UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)


def _resolve_codex_home(override: Optional[str]) -> Path:
    if override:
        return Path(override).expanduser()
    env_value = os.environ.get("CODEX_HOME")
    if env_value:
        return Path(env_value).expanduser()
    return Path.home() / ".codex"


def _find_latest_rollout(sessions_root: Path) -> Optional[Path]:
    if not sessions_root.exists():
        return None

    latest: Optional[Path] = None
    latest_mtime = -1.0
    for path in sessions_root.rglob("rollout-*.jsonl"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime > latest_mtime:
            latest = path
            latest_mtime = mtime
    return latest


def _extract_thread_id_from_rollout(rollout_path: Path) -> Optional[str]:
    try:
        with rollout_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError:
                    break
                if isinstance(obj, dict) and obj.get("type") == "thread.started":
                    thread_id = obj.get("thread_id")
                    if isinstance(thread_id, str) and thread_id:
                        return thread_id
                break
    except OSError:
        return None

    match = UUID_RE.search(rollout_path.name)
    return match.group(0) if match else None
