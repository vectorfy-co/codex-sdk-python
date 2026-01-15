"""Execution module for spawning and communicating with the Codex CLI."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import platform
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from .abort import AbortSignal
from .config_overrides import encode_config_overrides
from .exceptions import CodexAbortError, CodexCLIError, CodexError
from .options import ApprovalMode, ModelReasoningEffort, SandboxMode
from .telemetry import span

INTERNAL_ORIGINATOR_ENV = "CODEX_INTERNAL_ORIGINATOR_OVERRIDE"
PYTHON_SDK_ORIGINATOR = "codex_sdk_python"
CleanupCallable = Union[Callable[[], Awaitable[None]], Callable[[], None]]


@dataclass
class CodexExecArgs:
    """Typed arguments passed to the Codex CLI."""

    input: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    thread_id: Optional[str] = None
    images: Optional[List[str]] = None
    model: Optional[str] = None
    sandbox_mode: Optional[SandboxMode] = None
    working_directory: Optional[str] = None
    additional_directories: Optional[List[str]] = None
    skip_git_repo_check: Optional[bool] = None
    output_schema_file: Optional[str] = None
    model_reasoning_effort: Optional[ModelReasoningEffort] = None
    network_access_enabled: Optional[bool] = None
    web_search_enabled: Optional[bool] = None
    web_search_cached_enabled: Optional[bool] = None
    skills_enabled: Optional[bool] = None
    shell_snapshot_enabled: Optional[bool] = None
    background_terminals_enabled: Optional[bool] = None
    apply_patch_freeform_enabled: Optional[bool] = None
    exec_policy_enabled: Optional[bool] = None
    remote_models_enabled: Optional[bool] = None
    request_compression_enabled: Optional[bool] = None
    feature_overrides: Optional[Mapping[str, bool]] = None
    approval_policy: Optional[ApprovalMode] = None
    signal: Optional[AbortSignal] = None
    config_overrides: Optional[Mapping[str, Any]] = None


class CodexExec:
    """Handles spawning and communication with the Codex CLI."""

    def __init__(
        self,
        executable_path: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
    ):
        self.executable_path = executable_path or self._find_codex_path()
        self._env_override = dict(env) if env is not None else None
        self._ensure_executable()

    def _ensure_executable(self) -> None:
        """Ensure the bundled codex binary is executable on POSIX systems."""
        if platform.system().lower() == "windows":
            return

        try:
            current_mode = os.stat(self.executable_path).st_mode
        except FileNotFoundError:
            # The path may refer to a binary on PATH; defer errors to process spawn.
            return
        except OSError as exc:
            raise CodexError(
                f"Failed to stat Codex CLI binary at {self.executable_path!r}"
            ) from exc

        if current_mode & stat.S_IXUSR:
            return

        try:
            os.chmod(
                self.executable_path,
                current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
            )
        except OSError as exc:
            raise CodexError(
                "Codex CLI binary is not executable and permissions could not be updated. "
                f"Try running: chmod +x {self.executable_path!r}"
            ) from exc

    def _find_codex_path(self) -> str:
        """Find the path to the codex binary based on platform."""
        system = platform.system().lower()
        machine = platform.machine().lower()

        target_triple = None
        if system == "linux":
            if machine in ["x86_64", "amd64"]:
                target_triple = "x86_64-unknown-linux-musl"
            elif machine in ["aarch64", "arm64"]:
                target_triple = "aarch64-unknown-linux-musl"
        elif system == "darwin":
            if machine in ["x86_64", "amd64"]:
                target_triple = "x86_64-apple-darwin"
            elif machine in ["aarch64", "arm64"]:
                target_triple = "aarch64-apple-darwin"
        elif system == "windows":
            if machine in ["x86_64", "amd64"]:
                target_triple = "x86_64-pc-windows-msvc"
            elif machine in ["aarch64", "arm64"]:
                target_triple = "aarch64-pc-windows-msvc"

        if not target_triple:
            raise CodexError(f"Unsupported platform: {system} ({machine})")

        module_dir = Path(__file__).parent
        vendor_root = module_dir / "vendor"
        arch_root = vendor_root / target_triple
        binary_name = "codex.exe" if system == "windows" else "codex"
        binary_path = arch_root / "codex" / binary_name

        if binary_path.exists():
            return str(binary_path)

        path_binary = shutil.which(binary_name)
        if path_binary:
            return path_binary

        raise CodexError(
            "Codex CLI binary not found. Expected it at "
            f"{binary_path} or on PATH. If you're working from source, run "
            "`python scripts/setup_binary.py` to download the vendor binaries. "
            "Otherwise, install the Codex CLI separately or pass "
            "`CodexOptions.codex_path_override`."
        )

    async def run(self, args: CodexExecArgs) -> AsyncGenerator[str, None]:
        """
        Run the Codex CLI with the given arguments and yield decoded output lines.

        Args:
            args: Structured arguments for the CLI invocation.

        Yields:
            Each line of JSON output as a UTF-8 decoded string.

        Raises:
            CodexCLIError: If the CLI exits with a non-zero status.
            CodexError: If the CLI cannot be spawned or misconfigured.
        """
        command_args = ["exec", "--experimental-json"]

        if args.model:
            command_args.extend(["--model", args.model])

        if args.sandbox_mode:
            command_args.extend(["--sandbox", args.sandbox_mode])

        if args.config_overrides:
            for override in encode_config_overrides(args.config_overrides):
                command_args.extend(["--config", override])

        if args.working_directory:
            command_args.extend(["--cd", args.working_directory])

        if args.additional_directories:
            for directory in args.additional_directories:
                command_args.extend(["--add-dir", directory])

        if args.skip_git_repo_check:
            command_args.append("--skip-git-repo-check")

        if args.output_schema_file:
            command_args.extend(["--output-schema", args.output_schema_file])

        if args.model_reasoning_effort:
            command_args.extend(
                ["--config", f'model_reasoning_effort="{args.model_reasoning_effort}"']
            )

        if args.feature_overrides:
            for key in sorted(args.feature_overrides):
                enabled = args.feature_overrides[key]
                if not isinstance(enabled, bool):
                    raise CodexError(
                        f"feature_overrides[{key!r}] must be a bool, got {type(enabled).__name__}"
                    )
                config_key = key if key.startswith("features.") else f"features.{key}"
                value = "true" if enabled else "false"
                command_args.extend(["--config", f"{config_key}={value}"])

        if args.network_access_enabled is not None:
            enabled_str = "true" if args.network_access_enabled else "false"
            command_args.extend(
                ["--config", f"sandbox_workspace_write.network_access={enabled_str}"]
            )

        if args.web_search_enabled is not None:
            enabled_str = "true" if args.web_search_enabled else "false"
            command_args.extend(
                ["--config", f"features.web_search_request={enabled_str}"]
            )

        if args.web_search_cached_enabled is not None:
            enabled_str = "true" if args.web_search_cached_enabled else "false"
            command_args.extend(
                ["--config", f"features.web_search_cached={enabled_str}"]
            )

        if args.shell_snapshot_enabled is not None:
            enabled_str = "true" if args.shell_snapshot_enabled else "false"
            command_args.extend(["--config", f"features.shell_snapshot={enabled_str}"])

        if args.background_terminals_enabled is not None:
            enabled_str = "true" if args.background_terminals_enabled else "false"
            command_args.extend(["--config", f"features.unified_exec={enabled_str}"])

        if args.apply_patch_freeform_enabled is not None:
            enabled_str = "true" if args.apply_patch_freeform_enabled else "false"
            command_args.extend(
                ["--config", f"features.apply_patch_freeform={enabled_str}"]
            )

        if args.exec_policy_enabled is not None:
            enabled_str = "true" if args.exec_policy_enabled else "false"
            command_args.extend(["--config", f"features.exec_policy={enabled_str}"])

        if args.remote_models_enabled is not None:
            enabled_str = "true" if args.remote_models_enabled else "false"
            command_args.extend(["--config", f"features.remote_models={enabled_str}"])

        if args.request_compression_enabled is not None:
            enabled_str = "true" if args.request_compression_enabled else "false"
            command_args.extend(
                ["--config", f"features.enable_request_compression={enabled_str}"]
            )

        if args.approval_policy:
            command_args.extend(
                ["--config", f'approval_policy="{args.approval_policy}"']
            )

        if args.images:
            for image in args.images:
                command_args.extend(["--image", image])

        if args.thread_id:
            command_args.extend(["resume", args.thread_id])

        if self._env_override is not None:
            env = dict(self._env_override)
        else:
            env = os.environ.copy()
        if INTERNAL_ORIGINATOR_ENV not in env:
            env[INTERNAL_ORIGINATOR_ENV] = PYTHON_SDK_ORIGINATOR

        if args.base_url:
            env["OPENAI_BASE_URL"] = args.base_url

        if args.api_key:
            env["CODEX_API_KEY"] = args.api_key

        if args.signal is not None and args.signal.aborted:
            raise CodexAbortError(
                args.signal.reason
                if args.signal.reason is not None
                else "Operation aborted"
            )

        with span(
            "codex_sdk.exec",
            thread_id=args.thread_id,
            model=args.model,
            sandbox_mode=args.sandbox_mode,
            working_directory=args.working_directory,
            approval_policy=args.approval_policy,
        ):
            process = await asyncio.create_subprocess_exec(
                self.executable_path,
                *command_args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            stderr_chunks: list[str] = []
            stderr_task = None
            abort_task: Optional[asyncio.Task[None]] = None
            aborted = False

            async def watch_abort() -> None:
                nonlocal aborted
                if args.signal is None:
                    return
                await args.signal.wait()
                aborted = True
                if process.returncode is None:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()

            try:
                if args.signal is not None:
                    abort_task = asyncio.create_task(watch_abort())

                if process.stderr:
                    stderr_task = asyncio.create_task(
                        _drain_stream(process.stderr, stderr_chunks)
                    )

                if process.stdin:
                    process.stdin.write(args.input.encode("utf-8"))
                    await process.stdin.drain()
                    process.stdin.close()
                else:
                    raise CodexError("Codex CLI process did not expose stdin.")

                if not process.stdout:
                    raise CodexError("Codex CLI process did not expose stdout.")

                async for line in _iter_lines(process.stdout):
                    yield line

                return_code = await process.wait()
                if stderr_task:
                    await stderr_task
                if aborted or (args.signal is not None and args.signal.aborted):
                    raise CodexAbortError(
                        args.signal.reason
                        if args.signal and args.signal.reason is not None
                        else "Operation aborted"
                    )
                if return_code != 0:
                    raise CodexCLIError(return_code, "".join(stderr_chunks))
            finally:
                if abort_task and not abort_task.done():
                    abort_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await abort_task

                if stderr_task and not stderr_task.done():
                    stderr_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await stderr_task

                if process.returncode is None:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()


async def create_output_schema_file(
    schema: Optional[Mapping[str, Any]],
) -> Tuple[Optional[str], CleanupCallable]:
    """
    Create a temporary file for the output schema and return cleanup function.

    Args:
        schema: Mapping describing the JSON schema for structured output.

    Returns:
        Tuple of schema path (if created) and a cleanup callable that may be awaited.

    Raises:
        CodexError: If the schema is not a JSON object or file creation fails.
    """
    if schema is None:
        return None, _noop

    if not isinstance(schema, Mapping):
        raise CodexError("output_schema must be a plain JSON object")

    temp_dir = tempfile.mkdtemp(prefix="codex-output-schema-")
    temp_path = os.path.join(temp_dir, "schema.json")

    async def cleanup_async() -> None:
        await asyncio.to_thread(_cleanup_dir, temp_dir)

    try:
        await asyncio.to_thread(_write_schema_file, temp_path, schema)
        return temp_path, cleanup_async
    except Exception:
        await cleanup_async()
        raise


async def _drain_stream(stream: asyncio.StreamReader, sink: list[str]) -> None:
    """Continuously read from a stream and append decoded chunks to sink."""
    while True:
        chunk = await stream.readline()
        if not chunk:
            break
        sink.append(chunk.decode("utf-8"))


async def _iter_lines(stream: asyncio.StreamReader) -> AsyncGenerator[str, None]:
    """Yield decoded lines from a stream until it is exhausted."""
    while True:
        line = await stream.readline()
        if not line:
            break
        yield line.decode("utf-8").rstrip("\n\r")


def _write_schema_file(path: str, schema: Mapping[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(dict(schema), handle)


def _cleanup_dir(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


async def _noop() -> None:
    return None
