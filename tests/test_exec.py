import asyncio
import inspect
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest

from codex_sdk.abort import AbortController
from codex_sdk.exceptions import CodexCLIError, CodexError
from codex_sdk.exec import (
    INTERNAL_ORIGINATOR_ENV,
    PYTHON_SDK_ORIGINATOR,
    CodexAbortError,
    CodexExec,
    CodexExecArgs,
    create_output_schema_file,
)


class FakeStream:
    def __init__(self, lines: List[str]):
        self._lines = [line.encode("utf-8") for line in lines]

    async def readline(self) -> bytes:
        if not self._lines:
            return b""
        return self._lines.pop(0)


class FakeStdin:
    def __init__(self) -> None:
        self.writes: List[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class FakeProcess:
    def __init__(
        self, stdout_lines: List[str], stderr_lines: List[str], return_code: int
    ):
        self.stdin = FakeStdin()
        self.stdout = FakeStream(stdout_lines)
        self.stderr = FakeStream(stderr_lines)
        self._return_code = return_code
        self.returncode = None

    async def wait(self) -> int:
        self.returncode = self._return_code
        return self._return_code

    def terminate(self) -> None:
        self.returncode = self._return_code

    def kill(self) -> None:
        self.returncode = self._return_code


class FakeBlockingStream:
    def __init__(self):
        self._event = asyncio.Event()

    async def readline(self) -> bytes:
        await self._event.wait()
        return b""


class FakeProcessNoStdin(FakeProcess):
    def __init__(self):
        super().__init__([], [], 0)
        self.stdin = None


class FakeProcessNoStdout(FakeProcess):
    def __init__(self):
        super().__init__([], [], 0)
        self.stdout = None


class FakeProcessWithBlockingStderr(FakeProcessNoStdin):
    def __init__(self):
        super().__init__()
        self.stderr = FakeBlockingStream()


class FakeProcessSlow(FakeProcess):
    def __init__(self):
        super().__init__([], [], 0)
        self._wait_event = asyncio.Event()
        self.terminated = False
        self.killed = False

    async def wait(self) -> int:
        await self._wait_event.wait()
        self.returncode = self._return_code
        return self._return_code

    def terminate(self) -> None:
        self.terminated = True
        self._wait_event.set()
        super().terminate()

    def kill(self) -> None:
        self.killed = True
        self._wait_event.set()
        super().kill()


@pytest.mark.asyncio
async def test_exec_passes_args_and_respects_env(monkeypatch: pytest.MonkeyPatch):
    captured: Dict[str, Any] = {}

    async def fake_spawn(*cmd: Any, **kwargs: Any) -> FakeProcess:
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env", {})
        return FakeProcess([], [], 0)

    monkeypatch.setenv(INTERNAL_ORIGINATOR_ENV, "custom-origin")
    monkeypatch.setenv("CODEX_API_KEY", "existing-key")
    monkeypatch.setenv("CUSTOM_VAR", "preserved")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    exec = CodexExec("codex-binary")
    args = CodexExecArgs(
        input="hello",
        base_url="https://api.example.com",
        api_key=None,
        thread_id="thread-123",
        model="gpt-4",
        sandbox_mode="read-only",
        working_directory="/tmp/work",
        skip_git_repo_check=True,
        output_schema_file="/tmp/schema.json",
        config_overrides={
            "analytics.enabled": True,
            "notify": ["python3", "/tmp/notify.py"],
        },
    )

    async for _ in exec.run(args):
        pass

    cmd_list = list(captured["cmd"])
    assert cmd_list[:2] == ["codex-binary", "exec"]
    assert "--model" in cmd_list and "gpt-4" in cmd_list
    assert "--sandbox" in cmd_list and "read-only" in cmd_list
    assert "--cd" in cmd_list and "/tmp/work" in cmd_list
    assert "--skip-git-repo-check" in cmd_list
    assert "--output-schema" in cmd_list and "/tmp/schema.json" in cmd_list
    assert "--config" in cmd_list and "analytics.enabled=true" in cmd_list
    assert "--config" in cmd_list and 'notify=["python3", "/tmp/notify.py"]' in cmd_list
    assert cmd_list[-2:] == ["resume", "thread-123"]

    env = captured["env"]
    assert env["OPENAI_BASE_URL"] == "https://api.example.com"
    assert env["CODEX_API_KEY"] == "existing-key"
    assert env[INTERNAL_ORIGINATOR_ENV] == "custom-origin"
    assert env["CUSTOM_VAR"] == "preserved"


@pytest.mark.asyncio
async def test_exec_sets_originator_when_absent(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(INTERNAL_ORIGINATOR_ENV, raising=False)
    monkeypatch.setenv("CODEX_API_KEY", "token")

    captured: Dict[str, Any] = {}

    async def fake_spawn(*cmd: Any, **kwargs: Any) -> FakeProcess:
        captured["env"] = kwargs.get("env", {})
        return FakeProcess([], [], 0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    exec = CodexExec("codex-binary")
    args = CodexExecArgs(input="hello")

    async for _ in exec.run(args):
        pass

    assert captured["env"][INTERNAL_ORIGINATOR_ENV] == PYTHON_SDK_ORIGINATOR


@pytest.mark.asyncio
async def test_exec_surfaces_cli_errors_with_stderr(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CODEX_API_KEY", "token")

    async def fake_spawn(*cmd: Any, **kwargs: Any) -> FakeProcess:
        return FakeProcess([], ["boom\n"], 9)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    exec = CodexExec("codex-binary")
    args = CodexExecArgs(input="hello")

    with pytest.raises(CodexCLIError) as exc:
        async for _ in exec.run(args):
            pass

    assert exc.value.exit_code == 9
    assert "boom" in exc.value.stderr


@pytest.mark.asyncio
async def test_exec_does_not_require_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CODEX_API_KEY", raising=False)

    async def fake_spawn(*cmd: Any, **kwargs: Any) -> FakeProcess:
        return FakeProcess([], [], 0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    exec = CodexExec("codex-binary")
    args = CodexExecArgs(input="hello")

    async for _ in exec.run(args):
        pass


@pytest.mark.asyncio
async def test_exec_allows_overriding_env(monkeypatch: pytest.MonkeyPatch):
    captured: Dict[str, Any] = {}

    async def fake_spawn(*cmd: Any, **kwargs: Any) -> FakeProcess:
        captured["env"] = kwargs.get("env", {})
        return FakeProcess([], [], 0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)
    monkeypatch.setenv("CODEX_ENV_SHOULD_NOT_LEAK", "leak")

    exec = CodexExec("codex-binary", env={"CUSTOM_ENV": "custom"})
    args = CodexExecArgs(
        input="hello", base_url="https://api.example.com", api_key="test"
    )

    async for _ in exec.run(args):
        pass

    env = captured["env"]
    assert env["CUSTOM_ENV"] == "custom"
    assert env.get("CODEX_ENV_SHOULD_NOT_LEAK") is None
    assert env["OPENAI_BASE_URL"] == "https://api.example.com"
    assert env["CODEX_API_KEY"] == "test"
    assert env[INTERNAL_ORIGINATOR_ENV] == PYTHON_SDK_ORIGINATOR


@pytest.mark.asyncio
async def test_exec_aborts_when_signal_already_aborted(monkeypatch: pytest.MonkeyPatch):
    async def fake_spawn(*cmd: Any, **kwargs: Any) -> FakeProcess:
        raise AssertionError("process should not spawn when already aborted")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    controller = AbortController()
    controller.abort("test abort")

    exec = CodexExec("codex-binary")
    args = CodexExecArgs(input="hello", signal=controller.signal)

    with pytest.raises(CodexAbortError):
        async for _ in exec.run(args):
            pass


@pytest.mark.asyncio
async def test_exec_passes_config_and_repeated_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("CODEX_API_KEY", "token")
    captured: Dict[str, Any] = {}

    async def fake_spawn(*cmd: Any, **kwargs: Any) -> FakeProcess:
        captured["cmd"] = list(cmd)
        return FakeProcess([], [], 0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    exec = CodexExec("codex-binary")
    instructions_path = tmp_path / "instructions.md"
    args = CodexExecArgs(
        input="hello",
        model_reasoning_effort="high",
        model_instructions_file=instructions_path,
        model_personality="pragmatic",
        max_threads=5,
        network_access_enabled=True,
        web_search_enabled=False,
        web_search_cached_enabled=True,
        shell_snapshot_enabled=True,
        background_terminals_enabled=False,
        apply_patch_freeform_enabled=True,
        exec_policy_enabled=False,
        remote_models_enabled=True,
        collaboration_modes_enabled=True,
        connectors_enabled=True,
        responses_websockets_enabled=False,
        request_compression_enabled=False,
        approval_policy="on-request",
        additional_directories=["../backend", "/tmp/shared"],
        images=["/tmp/one.png", "/tmp/two.jpg"],
    )

    async for _ in exec.run(args):
        pass

    cmd_list = captured["cmd"]
    assert cmd_list[:2] == ["codex-binary", "exec"]
    assert "--config" in cmd_list
    assert 'model_reasoning_effort="high"' in cmd_list
    assert f'model_instructions_file="{instructions_path}"' in cmd_list
    assert 'model_personality="pragmatic"' in cmd_list
    assert "agents.max_threads=5" in cmd_list
    assert "sandbox_workspace_write.network_access=true" in cmd_list
    assert 'web_search="cached"' in cmd_list
    assert "features.shell_snapshot=true" in cmd_list
    assert "features.unified_exec=false" in cmd_list
    assert "features.apply_patch_freeform=true" in cmd_list
    assert "features.exec_policy=false" in cmd_list
    assert "features.remote_models=true" in cmd_list
    assert "features.collaboration_modes=true" in cmd_list
    assert "features.connectors=true" in cmd_list
    assert "features.responses_websockets=false" in cmd_list
    assert "features.enable_request_compression=false" in cmd_list
    assert 'approval_policy="on-request"' in cmd_list

    add_dir_values = [
        cmd_list[i + 1] for i, v in enumerate(cmd_list[:-1]) if v == "--add-dir"
    ]
    assert add_dir_values == ["../backend", "/tmp/shared"]

    image_values = [
        cmd_list[i + 1] for i, v in enumerate(cmd_list[:-1]) if v == "--image"
    ]
    assert image_values == ["/tmp/one.png", "/tmp/two.jpg"]


@pytest.mark.asyncio
async def test_exec_prefers_web_search_mode(monkeypatch: pytest.MonkeyPatch):
    captured: Dict[str, Any] = {}

    async def fake_spawn(*cmd: Any, **kwargs: Any) -> FakeProcess:
        captured["cmd"] = list(cmd)
        return FakeProcess([], [], 0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    exec = CodexExec("codex-binary")
    args = CodexExecArgs(
        input="hello",
        web_search_mode="live",
        web_search_enabled=False,
        web_search_cached_enabled=True,
    )

    async for _ in exec.run(args):
        pass

    cmd_list = captured["cmd"]
    assert 'web_search="live"' in cmd_list


@pytest.mark.asyncio
async def test_exec_omits_optional_feature_flags_when_unset(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: Dict[str, Any] = {}

    async def fake_spawn(*cmd: Any, **kwargs: Any) -> FakeProcess:
        captured["cmd"] = list(cmd)
        return FakeProcess([], [], 0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    exec = CodexExec("codex-binary")
    args = CodexExecArgs(input="hello")

    async for _ in exec.run(args):
        pass

    cmd_list = captured["cmd"]
    assert all(
        "features.skills" not in arg
        and "features.shell_snapshot" not in arg
        and "features.unified_exec" not in arg
        and "web_search=" not in arg
        and "features.apply_patch_freeform" not in arg
        and "features.exec_policy" not in arg
        and "features.remote_models" not in arg
        and "features.collaboration_modes" not in arg
        and "features.connectors" not in arg
        and "features.responses_websockets" not in arg
        and "features.enable_request_compression" not in arg
        and "model_instructions_file" not in arg
        and "model_personality" not in arg
        and "agents.max_threads" not in arg
        for arg in cmd_list
        if isinstance(arg, str)
    )


@pytest.mark.asyncio
async def test_exec_rejects_max_threads_over_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_spawn(*_cmd: Any, **_kwargs: Any) -> FakeProcess:
        raise AssertionError("process should not spawn when max_threads is invalid")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    exec = CodexExec("codex-binary")
    args = CodexExecArgs(input="hello", max_threads=7)

    with pytest.raises(CodexError):
        async for _ in exec.run(args):
            pass


@pytest.mark.asyncio
async def test_exec_applies_feature_overrides(monkeypatch: pytest.MonkeyPatch):
    captured: Dict[str, Any] = {}

    async def fake_spawn(*cmd: Any, **kwargs: Any) -> FakeProcess:
        captured["cmd"] = list(cmd)
        return FakeProcess([], [], 0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    exec = CodexExec("codex-binary")
    args = CodexExecArgs(
        input="hello",
        feature_overrides={"remote_models": True, "powershell_utf8": True},
        remote_models_enabled=False,
    )

    async for _ in exec.run(args):
        pass

    cmd_list = captured["cmd"]
    override_key = "features.remote_models=true"
    explicit_key = "features.remote_models=false"
    assert "features.powershell_utf8=true" in cmd_list
    assert override_key in cmd_list
    assert explicit_key in cmd_list
    assert cmd_list.index(override_key) < cmd_list.index(explicit_key)


@pytest.mark.asyncio
async def test_create_output_schema_file_creates_and_cleans() -> None:
    schema = {"type": "object", "properties": {"field": {"type": "string"}}}
    path, cleanup = await create_output_schema_file(schema)
    assert path is not None
    assert os.path.exists(path)
    cleanup_result = cleanup()
    if inspect.isawaitable(cleanup_result):
        await cleanup_result
    assert not os.path.exists(os.path.dirname(path))


@pytest.mark.asyncio
async def test_create_output_schema_file_validates_schema() -> None:
    with pytest.raises(CodexError):
        await create_output_schema_file(["not", "an", "object"])


@pytest.mark.parametrize(
    ("system", "machine", "expected_triple", "expected_binary_name"),
    [
        ("linux", "x86_64", "x86_64-unknown-linux-musl", "codex"),
        ("linux", "aarch64", "aarch64-unknown-linux-musl", "codex"),
        ("darwin", "x86_64", "x86_64-apple-darwin", "codex"),
        ("darwin", "arm64", "aarch64-apple-darwin", "codex"),
        ("windows", "x86_64", "x86_64-pc-windows-msvc", "codex.exe"),
        ("windows", "arm64", "aarch64-pc-windows-msvc", "codex.exe"),
    ],
)
def test_find_codex_path_builds_vendor_path(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    machine: str,
    expected_triple: str,
    expected_binary_name: str,
):
    import codex_sdk.exec as exec_module

    exec_obj = CodexExec.__new__(CodexExec)
    monkeypatch.setattr(exec_module.platform, "system", lambda: system)
    monkeypatch.setattr(exec_module.platform, "machine", lambda: machine)
    path = exec_obj._find_codex_path()
    assert expected_triple in path
    assert path.endswith(expected_binary_name)


def test_find_codex_path_raises_for_unsupported_platform(
    monkeypatch: pytest.MonkeyPatch,
):
    import codex_sdk.exec as exec_module

    exec_obj = CodexExec.__new__(CodexExec)
    monkeypatch.setattr(exec_module.platform, "system", lambda: "solaris")
    monkeypatch.setattr(exec_module.platform, "machine", lambda: "sparc")
    with pytest.raises(CodexError):
        exec_obj._find_codex_path()


def test_ensure_executable_is_noop_on_windows(monkeypatch: pytest.MonkeyPatch):
    import codex_sdk.exec as exec_module

    exec_obj = CodexExec.__new__(CodexExec)
    exec_obj.executable_path = "missing"
    monkeypatch.setattr(exec_module.platform, "system", lambda: "Windows")
    exec_obj._ensure_executable()


def test_ensure_executable_sets_exec_bit(tmp_path: Path):
    binary_path = tmp_path / "codex"
    binary_path.write_text("x", encoding="utf-8")
    os.chmod(binary_path, 0o644)

    CodexExec(str(binary_path))
    mode = os.stat(binary_path).st_mode
    assert mode & 0o111


def test_ensure_executable_raises_on_stat_error(monkeypatch: pytest.MonkeyPatch):
    import codex_sdk.exec as exec_module

    def bad_stat(_: Any):
        raise OSError("boom")

    monkeypatch.setattr(exec_module.os, "stat", bad_stat)
    with pytest.raises(CodexError):
        CodexExec("codex-binary")


def test_ensure_executable_raises_on_chmod_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import codex_sdk.exec as exec_module

    binary_path = tmp_path / "codex"
    binary_path.write_text("x", encoding="utf-8")
    os.chmod(binary_path, 0o644)

    monkeypatch.setattr(
        exec_module.os,
        "chmod",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no")),
    )
    with pytest.raises(CodexError):
        CodexExec(str(binary_path))


@pytest.mark.asyncio
async def test_exec_yields_stdout_lines(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CODEX_API_KEY", "token")

    async def fake_spawn(*cmd: Any, **kwargs: Any) -> FakeProcess:
        return FakeProcess(["first\n", "second\r\n"], [], 0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    exec = CodexExec("codex-binary")
    args = CodexExecArgs(input="hello")

    lines = [line async for line in exec.run(args)]
    assert lines == ["first", "second"]


@pytest.mark.asyncio
async def test_exec_raises_when_process_stdin_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CODEX_API_KEY", "token")

    async def fake_spawn(*cmd: Any, **kwargs: Any) -> FakeProcessNoStdin:
        return FakeProcessNoStdin()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    exec = CodexExec("codex-binary")
    args = CodexExecArgs(input="hello")

    with pytest.raises(CodexError, match="did not expose stdin"):
        async for _ in exec.run(args):
            pass


@pytest.mark.asyncio
async def test_exec_raises_when_process_stdout_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CODEX_API_KEY", "token")

    async def fake_spawn(*cmd: Any, **kwargs: Any) -> FakeProcessNoStdout:
        return FakeProcessNoStdout()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    exec = CodexExec("codex-binary")
    args = CodexExecArgs(input="hello")

    with pytest.raises(CodexError, match="did not expose stdout"):
        async for _ in exec.run(args):
            pass


@pytest.mark.asyncio
async def test_exec_aborts_after_start(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CODEX_API_KEY", "token")
    controller = AbortController()
    process = FakeProcessSlow()

    async def fake_spawn(*cmd: Any, **kwargs: Any) -> FakeProcessSlow:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    exec = CodexExec("codex-binary")
    args = CodexExecArgs(input="hello", signal=controller.signal)

    async def consume() -> None:
        async for _ in exec.run(args):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    controller.abort("stop")

    with pytest.raises(CodexAbortError):
        await task

    assert process.terminated is True


@pytest.mark.asyncio
async def test_exec_cancels_abort_task_on_completion(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CODEX_API_KEY", "token")
    controller = AbortController()

    async def fake_spawn(*cmd: Any, **kwargs: Any) -> FakeProcess:
        return FakeProcess([], [], 0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    exec = CodexExec("codex-binary")
    args = CodexExecArgs(input="hello", signal=controller.signal)

    async for _ in exec.run(args):
        pass


@pytest.mark.asyncio
async def test_exec_cancels_stderr_task_when_exiting_early(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("CODEX_API_KEY", "token")

    async def fake_spawn(*cmd: Any, **kwargs: Any) -> FakeProcessWithBlockingStderr:
        return FakeProcessWithBlockingStderr()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    exec = CodexExec("codex-binary")
    args = CodexExecArgs(input="hello")
    with pytest.raises(CodexError):
        async for _ in exec.run(args):
            pass


@pytest.mark.asyncio
async def test_create_output_schema_file_cleans_on_write_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import codex_sdk.exec as exec_module

    created: Dict[str, Any] = {}

    def fake_mkdtemp(prefix: str) -> str:
        schema_dir = tmp_path / "schema"
        schema_dir.mkdir()
        created["dir"] = schema_dir
        return str(schema_dir)

    monkeypatch.setattr(exec_module.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(
        exec_module,
        "_write_schema_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError):
        await create_output_schema_file({"type": "object"})

    assert not (tmp_path / "schema").exists()
