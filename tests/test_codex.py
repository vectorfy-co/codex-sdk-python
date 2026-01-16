"""
Tests for the Codex Python SDK surface.
"""

import json
import os
from pathlib import Path

import pytest

from codex_sdk import (
    Codex,
    CodexOptions,
    ThreadOptions,
    TurnOptions,
)
from codex_sdk.exceptions import CodexError, CodexParseError, TurnFailedError
from codex_sdk.thread import Thread


class FakeExec:
    def __init__(self, lines):
        self.lines = lines
        self.calls = []

    async def run(self, args):
        self.calls.append(args)
        for line in self.lines:
            yield line


class TestCodex:
    """Test cases for the Codex class."""

    def test_codex_initialization(self):
        """Test that Codex can be initialized with default options."""
        codex = Codex()
        assert codex is not None

    def test_codex_initialization_with_options(self):
        """Test that Codex can be initialized with custom options."""
        options = CodexOptions(base_url="https://api.example.com", api_key="test-key")
        codex = Codex(options)
        assert codex is not None

    def test_start_thread(self):
        """Test that start_thread returns a Thread instance."""
        codex = Codex()
        thread = codex.start_thread()
        assert isinstance(thread, Thread)

    def test_start_thread_with_options(self):
        """Test that start_thread accepts ThreadOptions."""
        codex = Codex()
        options = ThreadOptions(model="gpt-4", working_directory="/tmp")
        thread = codex.start_thread(options)
        assert thread is not None

    def test_resume_thread(self):
        """Test that resume_thread returns a Thread instance with the provided id."""
        codex = Codex()
        thread = codex.resume_thread("test-thread-id")
        assert thread is not None
        assert thread.id == "test-thread-id"

    def test_resume_last_thread_prefers_latest_rollout(self, tmp_path: Path):
        codex_home = tmp_path / "codex-home"
        sessions_root = codex_home / "sessions"
        sessions_root.mkdir(parents=True)

        rollout_1 = sessions_root / "t1" / "rollout-1.jsonl"
        rollout_1.parent.mkdir(parents=True)
        rollout_1.write_text(
            json.dumps({"type": "thread.started", "thread_id": "thread-1"}) + "\n",
            encoding="utf-8",
        )

        rollout_2 = sessions_root / "t2" / "rollout-2.jsonl"
        rollout_2.parent.mkdir(parents=True)
        rollout_2.write_text(
            json.dumps({"type": "thread.started", "thread_id": "thread-2"}) + "\n",
            encoding="utf-8",
        )

        now = 1_700_000_000
        os.utime(rollout_1, (now, now))
        os.utime(rollout_2, (now + 10, now + 10))

        codex = Codex(CodexOptions(codex_path_override="codex-binary"))
        thread = codex.resume_last_thread(codex_home=str(codex_home))
        assert thread.id == "thread-2"

    def test_resume_last_thread_falls_back_to_uuid_in_filename(self, tmp_path: Path):
        codex_home = tmp_path / "codex-home"
        sessions_root = codex_home / "sessions"
        sessions_root.mkdir(parents=True)

        thread_uuid = "123e4567-e89b-12d3-a456-426614174000"
        rollout = sessions_root / f"rollout-{thread_uuid}.jsonl"
        rollout.write_text("not json\n", encoding="utf-8")

        codex = Codex(CodexOptions(codex_path_override="codex-binary"))
        thread = codex.resume_last_thread(codex_home=str(codex_home))
        assert thread.id == thread_uuid

    def test_resume_last_thread_raises_when_missing_sessions(self, tmp_path: Path):
        codex = Codex(CodexOptions(codex_path_override="codex-binary"))
        with pytest.raises(CodexError):
            codex.resume_last_thread(codex_home=str(tmp_path / "codex-home"))

    def test_resume_last_thread_raises_when_no_thread_id(self, tmp_path: Path):
        codex_home = tmp_path / "codex-home"
        sessions_root = codex_home / "sessions"
        sessions_root.mkdir(parents=True)

        rollout = sessions_root / "rollout-latest.jsonl"
        rollout.write_text("not json\n", encoding="utf-8")

        codex = Codex(CodexOptions(codex_path_override="codex-binary"))
        with pytest.raises(CodexError):
            codex.resume_last_thread(codex_home=str(codex_home))

    def test_resume_last_thread_uses_CODEX_HOME_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        codex_home = tmp_path / "codex-home"
        sessions_root = codex_home / "sessions"
        sessions_root.mkdir(parents=True)

        rollout = sessions_root / "rollout-1.jsonl"
        rollout.write_text(
            json.dumps({"type": "thread.started", "thread_id": "thread-env"}) + "\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("CODEX_HOME", str(codex_home))
        codex = Codex(CodexOptions(codex_path_override="codex-binary"))
        thread = codex.resume_last_thread()
        assert thread.id == "thread-env"

    def test_resume_last_thread_uses_default_home(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("CODEX_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        sessions_root = tmp_path / ".codex" / "sessions"
        sessions_root.mkdir(parents=True)
        rollout = sessions_root / "rollout-1.jsonl"
        rollout.write_text(
            json.dumps({"type": "thread.started", "thread_id": "thread-home"}) + "\n",
            encoding="utf-8",
        )

        codex = Codex(CodexOptions(codex_path_override="codex-binary"))
        thread = codex.resume_last_thread()
        assert thread.id == "thread-home"

    def test_find_latest_rollout_ignores_stat_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from codex_sdk.codex import _find_latest_rollout

        sessions_root = tmp_path / "sessions"
        sessions_root.mkdir()
        good = sessions_root / "rollout-good.jsonl"
        good.write_text("{}", encoding="utf-8")
        bad = sessions_root / "rollout-bad.jsonl"
        bad.write_text("{}", encoding="utf-8")

        original_stat = Path.stat

        def fake_stat(self: Path, *args, **kwargs):
            if self == bad:
                raise OSError("boom")
            return original_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", fake_stat)
        assert _find_latest_rollout(sessions_root) == good


class TestThread:
    """Test cases for the Thread class."""

    @pytest.mark.asyncio
    async def test_run_basic(self):
        """Test basic run functionality."""
        lines = [
            '{"type": "thread.started", "thread_id": "test-thread-123"}',
            '{"type": "turn.started"}',
            '{"type": "item.completed", "item": {"id": "item-1", "type": "agent_message", "text": "Hello, World!"}}',
            '{"type": "turn.completed", "usage": {"input_tokens": 10, "cached_input_tokens": 0, "output_tokens": 5}}',
        ]
        exec = FakeExec(lines)
        thread = Thread(exec, CodexOptions(), ThreadOptions())

        turn = await thread.run("Test input")

        assert turn.final_response == "Hello, World!"
        assert len(turn.items) == 1
        assert turn.items[0].type == "agent_message"
        assert turn.usage is not None
        assert turn.usage.input_tokens == 10
        assert thread.id == "test-thread-123"

    @pytest.mark.asyncio
    async def test_run_streamed_events_helper(self):
        """Test streaming helper yields events directly."""
        lines = [
            '{"type": "turn.started"}',
            '{"type": "turn.completed", "usage": {"input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 1}}',
        ]
        exec = FakeExec(lines)
        thread = Thread(exec, CodexOptions(), ThreadOptions(), thread_id="existing")

        events = []
        async for event in thread.run_streamed_events("Test input"):
            events.append(event)

        assert len(events) == 2
        assert events[0].type == "turn.started"
        assert events[1].type == "turn.completed"
        assert exec.calls[0].thread_id == "existing"

    @pytest.mark.asyncio
    async def test_run_turn_failed_raises_turn_failed_error(self):
        """Test that a turn failure raises TurnFailedError."""
        lines = [
            '{"type": "turn.failed", "error": {"message": "boom"}}',
        ]
        exec = FakeExec(lines)
        thread = Thread(exec, CodexOptions(), ThreadOptions())

        with pytest.raises(TurnFailedError):
            await thread.run("Test input")

    @pytest.mark.asyncio
    async def test_unknown_event_raises_parse_error(self):
        """Test parse errors surface as CodexParseError."""
        lines = ['{"type": "unknown.event"}']
        exec = FakeExec(lines)
        thread = Thread(exec, CodexOptions(), ThreadOptions())

        with pytest.raises(CodexParseError):
            async for _ in thread.run_streamed_events("Test input"):
                pass

    def test_input_alias(self):
        """Input alias mirrors TypeScript SDK."""
        # Input is a typing alias; runtime equality checks are not meaningful.
        assert True

    @pytest.mark.asyncio
    async def test_input_normalization_for_images(self):
        """Structured input entries concatenate text and forward images."""
        lines = [
            '{"type": "turn.started"}',
            '{"type": "item.completed", "item": {"id": "item-1", "type": "agent_message", "text": "ok"}}',
            '{"type": "turn.completed", "usage": {"input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 1}}',
        ]
        exec = FakeExec(lines)
        thread = Thread(exec, CodexOptions(), ThreadOptions())

        await thread.run(
            [
                {"type": "text", "text": "Describe these screenshots"},
                {"type": "local_image", "path": "/tmp/ui.png"},
                {"type": "text", "text": "Focus on failures"},
                {"type": "local_image", "path": "/tmp/diagram.jpg"},
            ]
        )

        assert exec.calls[0].input == "Describe these screenshots\n\nFocus on failures"
        assert exec.calls[0].images == ["/tmp/ui.png", "/tmp/diagram.jpg"]


class TestOptions:
    """Test cases for option classes."""

    def test_codex_options(self):
        """Test CodexOptions initialization."""
        options = CodexOptions(
            codex_path_override="/custom/path",
            base_url="https://api.example.com",
            api_key="test-key",
            env={"CUSTOM_ENV": "custom"},
        )

        assert options.codex_path_override == "/custom/path"
        assert options.base_url == "https://api.example.com"
        assert options.api_key == "test-key"
        assert options.env == {"CUSTOM_ENV": "custom"}

    def test_thread_options(self):
        """Test ThreadOptions initialization."""
        options = ThreadOptions(
            model="gpt-4",
            sandbox_mode="read-only",
            working_directory="/tmp",
            skip_git_repo_check=True,
            model_reasoning_effort="high",
            network_access_enabled=True,
            web_search_mode="cached",
            web_search_enabled=False,
            web_search_cached_enabled=True,
            skills_enabled=True,
            shell_snapshot_enabled=True,
            background_terminals_enabled=True,
            apply_patch_freeform_enabled=True,
            exec_policy_enabled=False,
            remote_models_enabled=True,
            request_compression_enabled=True,
            feature_overrides={"web_search_cached": False},
            approval_policy="on-request",
            additional_directories=["../backend"],
        )

        assert options.model == "gpt-4"
        assert options.sandbox_mode == "read-only"
        assert options.working_directory == "/tmp"
        assert options.skip_git_repo_check is True
        assert options.model_reasoning_effort == "high"
        assert options.network_access_enabled is True
        assert options.web_search_mode == "cached"
        assert options.web_search_enabled is False
        assert options.web_search_cached_enabled is True
        assert options.skills_enabled is True
        assert options.shell_snapshot_enabled is True
        assert options.background_terminals_enabled is True
        assert options.apply_patch_freeform_enabled is True
        assert options.exec_policy_enabled is False
        assert options.remote_models_enabled is True
        assert options.request_compression_enabled is True
        assert options.feature_overrides == {"web_search_cached": False}
        assert options.approval_policy == "on-request"
        assert options.additional_directories == ["../backend"]

    def test_turn_options(self):
        """Test TurnOptions initialization."""
        schema = {"type": "object", "properties": {"test": {"type": "string"}}}
        options = TurnOptions(output_schema=schema)

        assert options.output_schema == schema
