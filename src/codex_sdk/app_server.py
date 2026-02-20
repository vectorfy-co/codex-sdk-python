"""Async client for the Codex app-server (JSON-RPC over stdio)."""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    TypedDict,
    Union,
    cast,
)

from .config_overrides import ConfigOverrides, encode_config_overrides
from .exceptions import CodexAppServerError, CodexError, CodexParseError
from .exec import INTERNAL_ORIGINATOR_ENV, PYTHON_SDK_ORIGINATOR


@dataclass
class AppServerClientInfo:
    """Metadata identifying the client for app-server initialize."""

    name: str
    title: str
    version: str

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation for initialize()."""
        return {
            "name": self.name,
            "title": self.title,
            "version": self.version,
        }


@dataclass
class AppServerOptions:
    """Options for configuring the app-server client."""

    codex_path_override: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    env: Optional[Mapping[str, str]] = None
    config_overrides: Optional[ConfigOverrides] = None
    client_info: Optional[AppServerClientInfo] = None
    # Enables experimental app-server methods/fields gated behind initialize.capabilities.
    experimental_api_enabled: bool = False
    auto_initialize: bool = True
    request_timeout: Optional[float] = None


@dataclass
class AppServerNotification:
    """Represents a JSON-RPC notification from the app-server."""

    method: str
    params: Optional[Dict[str, Any]]


@dataclass
class AppServerRequest:
    """Represents a JSON-RPC request sent from the app-server to the client."""

    id: Any
    method: str
    params: Optional[Dict[str, Any]]


@dataclass
class ApprovalDecisions:
    """Default decisions for app-server approval requests."""

    command_execution: Optional[Union[str, Mapping[str, Any]]] = None
    file_change: Optional[Union[str, Mapping[str, Any]]] = None
    execpolicy_amendment: Optional[Mapping[str, Any]] = None


class AppServerTurnSession:
    """Wrapper around a running turn that streams notifications and handles approvals."""

    def __init__(
        self,
        client: "AppServerClient",
        *,
        thread_id: str,
        turn_id: str,
        approvals: Optional[ApprovalDecisions] = None,
        initial_turn: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create a session object for an already-started app-server turn.

        Args:
            client: AppServerClient used to communicate with the app-server.
            thread_id: Thread id associated with the turn.
            turn_id: Turn id returned by `turn/start`.
            approvals: Optional auto-approval defaults for requestApproval prompts.
            initial_turn: Optional turn object returned at session creation.
        """
        self._client = client
        self.thread_id = thread_id
        self.turn_id = turn_id
        self.initial_turn = initial_turn
        self.final_turn: Optional[Dict[str, Any]] = None
        self._approvals = approvals
        self._notifications: asyncio.Queue[Optional[AppServerNotification]] = (
            asyncio.Queue()
        )
        self._requests: asyncio.Queue[Optional[AppServerRequest]] = asyncio.Queue()
        self._task: Optional[asyncio.Task[None]] = None
        self._done = asyncio.Event()
        self._closed = False

    async def __aenter__(self) -> "AppServerTurnSession":
        """Start the background pump and return the session."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[Any],
    ) -> None:
        """Close the session when used as an async context manager."""
        await self.close()

    async def start(self) -> None:
        """Start the background pump if it is not running already."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._pump())

    async def close(self) -> None:
        """Cancel background work and unblock any pending iterators."""
        if self._closed:
            return
        self._closed = True
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        await self._notifications.put(None)
        await self._requests.put(None)
        self._done.set()

    async def wait(self) -> Optional[Dict[str, Any]]:
        """Wait for the turn to reach a terminal notification and return it."""
        await self.start()
        await self._done.wait()
        return self.final_turn

    async def notifications(self) -> AsyncGenerator[AppServerNotification, None]:
        """Yield notifications observed during the turn."""
        await self.start()
        while True:
            item = await self._notifications.get()
            if item is None:
                break
            yield item

    async def next_notification(self) -> Optional[AppServerNotification]:
        """Return the next notification or None when the stream is closed."""
        await self.start()
        return await self._notifications.get()

    async def requests(self) -> AsyncGenerator[AppServerRequest, None]:
        """Yield requests observed during the turn."""
        await self.start()
        while True:
            item = await self._requests.get()
            if item is None:
                break
            yield item

    async def next_request(self) -> Optional[AppServerRequest]:
        """Return the next request or None when the stream is closed."""
        await self.start()
        return await self._requests.get()

    async def _pump(self) -> None:
        try:
            while True:
                notification_task = asyncio.create_task(
                    self._client.next_notification()
                )
                request_task = asyncio.create_task(self._client.next_request())
                done, pending = await asyncio.wait(
                    {notification_task, request_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in pending:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

                should_exit = False
                for task in done:
                    item = task.result()
                    if item is None:
                        should_exit = True
                        continue
                    if isinstance(item, AppServerNotification):
                        await self._notifications.put(item)
                        if self._is_turn_completed(item):
                            self.final_turn = _extract_turn(item)
                            should_exit = True
                    else:
                        handled = await self._handle_request(item)
                        if not handled:
                            await self._requests.put(item)

                if should_exit:
                    await self._drain_pending_requests()
                    return
        finally:
            self._done.set()
            await self._notifications.put(None)
            await self._requests.put(None)

    async def _drain_pending_requests(self) -> None:
        while True:
            try:
                request = await asyncio.wait_for(
                    self._client.next_request(), timeout=0.01
                )
            except asyncio.TimeoutError:
                break
            if request is None:
                break
            handled = await self._handle_request(request)
            if not handled:
                await self._requests.put(request)

    async def _handle_request(self, request: AppServerRequest) -> bool:
        if self._approvals is None:
            return False

        if not self._matches_turn(request.params):
            return False

        if request.method == "item/commandExecution/requestApproval":
            decision = self._approvals.command_execution
            if decision is None:
                return False
            payload = {
                "decision": _normalize_decision(
                    decision, self._approvals.execpolicy_amendment
                )
            }
            await self._client.respond(request.id, payload)
            return True

        if request.method == "item/fileChange/requestApproval":
            decision = self._approvals.file_change
            if decision is None:
                return False
            payload = {"decision": _normalize_decision(decision, None)}
            await self._client.respond(request.id, payload)
            return True

        return False

    def _matches_turn(self, params: Optional[Dict[str, Any]]) -> bool:
        if params is None:
            return True
        thread_id = params.get("threadId") or params.get("thread_id")
        turn_id = params.get("turnId") or params.get("turn_id")
        if thread_id is not None and thread_id != self.thread_id:
            return False
        if turn_id is not None and turn_id != self.turn_id:
            return False
        return True

    def _is_turn_completed(self, notification: AppServerNotification) -> bool:
        if notification.method != "turn/completed":
            return False
        turn = _extract_turn(notification)
        if not turn:
            return False
        turn_id = turn.get("id") if isinstance(turn, dict) else None
        return isinstance(turn_id, str) and turn_id == self.turn_id


class AppServerClient:
    """Async client for the Codex app-server."""

    def __init__(self, options: Optional[AppServerOptions] = None):
        """Create an AppServerClient.

        Args:
            options: Optional configuration controlling the codex binary path, env,
                client identity, timeouts, and whether to auto-initialize.
        """
        if options is None:
            options = AppServerOptions()
        self._options = options
        self._process: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._stderr_task: Optional[asyncio.Task[None]] = None
        self._pending: Dict[int, asyncio.Future[Any]] = {}
        self._notifications: asyncio.Queue[Optional[AppServerNotification]] = (
            asyncio.Queue()
        )
        self._requests: asyncio.Queue[Optional[AppServerRequest]] = asyncio.Queue()
        self._next_id = 1
        self._closed = False
        self._reader_error: Optional[BaseException] = None
        self._stderr_chunks: List[str] = []

    async def __aenter__(self) -> "AppServerClient":
        """Start the app-server process and return the client."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[Any],
    ) -> None:
        """Close the underlying app-server process on exit."""
        await self.close()

    async def start(self) -> None:
        """Start the underlying `codex app-server` subprocess (idempotent)."""
        if self._process is not None:
            return

        executable = self._resolve_executable()
        command_args = ["app-server"]
        if self._options.config_overrides:
            for override in encode_config_overrides(self._options.config_overrides):
                command_args.extend(["--config", override])

        env = self._build_env()

        process = await asyncio.create_subprocess_exec(
            executable,
            *command_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        if process.stdin is None or process.stdout is None:
            raise CodexError("Codex app-server did not expose stdin/stdout")

        self._process = process
        self._reader_task = asyncio.create_task(self._reader_loop())
        if process.stderr is not None:
            self._stderr_task = asyncio.create_task(
                _drain_stream(process.stderr, self._stderr_chunks)
            )

        if self._options.auto_initialize:
            await self.initialize(self._options.client_info)

    async def close(self) -> None:
        """Terminate the app-server subprocess and unblock pending requests."""
        if self._process is None:
            return

        self._closed = True
        self._fail_all_pending(CodexError("App-server closed"))

        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task

        if self._stderr_task and not self._stderr_task.done():
            self._stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stderr_task

        if self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()

        await self._notifications.put(None)
        await self._requests.put(None)
        self._process = None

    async def initialize(
        self, client_info: Optional[AppServerClientInfo] = None
    ) -> Dict[str, Any]:
        """
        Initialize the app-server client and register client information with the app-server.

        Ensures the underlying subprocess is started, sends an `initialize` request containing the provided (or default) client information and the experimental API capability when enabled, then emits an `initialized` notification.

        Args:
            client_info: Client identity to register; if omitted, a default client identity is
                used.

        Returns:
            The response payload returned by the app-server for the `initialize` request.
        """
        if self._process is None:
            await self.start()

        if client_info is None:
            client_info = self._default_client_info()

        params: Dict[str, Any] = {"clientInfo": client_info.as_dict()}
        if self._options.experimental_api_enabled:
            params["capabilities"] = {"experimentalApi": True}

        result = await self._request_dict("initialize", params)
        await self.notify("initialized")
        return result

    async def request(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Send a JSON-RPC request and await its result.

        Args:
            method: JSON-RPC method name.
            params: Optional method parameters (must be JSON-serializable).

        Returns:
            The raw result value from the app-server.

        Raises:
            CodexError: If the client is not started or if the reader loop failed.
            asyncio.TimeoutError: If request_timeout is configured and expires.
        """
        self._ensure_ready()
        if self._reader_error is not None:
            raise CodexError("App-server reader failed") from self._reader_error

        req_id = self._next_id
        self._next_id += 1
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[req_id] = future
        await self._send({"id": req_id, "method": method, "params": params})

        timeout = self._options.request_timeout
        if timeout is None:
            result = await future
        else:
            result = await asyncio.wait_for(future, timeout=timeout)
        return result

    async def _request_dict(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return cast(Dict[str, Any], await self.request(method, params))

    async def notify(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        self._ensure_ready()
        await self._send({"method": method, "params": params})

    async def notifications(self) -> AsyncGenerator[AppServerNotification, None]:
        """Yield notifications emitted by the app-server until the client is closed."""
        while True:
            item = await self._notifications.get()
            if item is None:
                break
            yield item

    async def next_notification(self) -> Optional[AppServerNotification]:
        """Return the next notification or None when the client is closed."""
        item = await self._notifications.get()
        return item

    async def requests(self) -> AsyncGenerator[AppServerRequest, None]:
        """Yield incoming requests emitted by the app-server until the client is closed."""
        while True:
            item = await self._requests.get()
            if item is None:
                break
            yield item

    async def next_request(self) -> Optional[AppServerRequest]:
        """Return the next incoming request or None when the client is closed."""
        return await self._requests.get()

    async def respond(
        self,
        request_id: Any,
        result: Optional[Any] = None,
        *,
        error: Optional[CodexAppServerError] = None,
    ) -> None:
        """Respond to an app-server initiated request.

        Args:
            request_id: The request id from the app-server's JSON-RPC request.
            result: Result value to return to the app-server.
            error: Optional error payload to return instead of a result.
        """
        if error is not None:
            payload = {
                "id": request_id,
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "data": error.data,
                },
            }
        else:
            payload = {"id": request_id, "result": result}
        await self._send(payload)

    async def thread_start(self, **params: Any) -> Dict[str, Any]:
        """Start a new thread on the app-server."""
        return await self._request_dict("thread/start", _coerce_keys(params))

    async def thread_resume(self, thread_id: str, **params: Any) -> Dict[str, Any]:
        """Resume an existing thread on the app-server."""
        payload = {"threadId": thread_id}
        payload.update(_coerce_keys(params))
        return await self._request_dict("thread/resume", payload)

    async def thread_fork(self, thread_id: str, **params: Any) -> Dict[str, Any]:
        """Fork an existing thread on the app-server."""
        payload = {"threadId": thread_id}
        payload.update(_coerce_keys(params))
        return await self._request_dict("thread/fork", payload)

    async def thread_loaded_list(
        self, *, cursor: Optional[str] = None, limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """List threads currently loaded by the app-server."""
        params: Dict[str, Any] = {}
        if cursor is not None:
            params["cursor"] = cursor
        if limit is not None:
            params["limit"] = limit
        return await self._request_dict("thread/loaded/list", params or None)

    async def thread_list(
        self,
        *,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
        sort_key: Optional[str] = None,
        model_providers: Optional[Sequence[str]] = None,
        source_kinds: Optional[Sequence[str]] = None,
        archived: Optional[bool] = None,
        cwd: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve a page of threads from the app-server with optional filtering and sorting.

        Args:
            cursor: Pagination cursor to continue listing from.
            limit: Maximum number of threads to return.
            sort_key: Key to sort results by (server-defined).
            model_providers: Filter threads by one or more model provider identifiers.
            source_kinds: Filter threads by one or more source kinds.
            archived: If set, restrict results to archived (`True`) or unarchived (`False`)
                threads.
            cwd: Optional working directory scope for server-side filtering.

        Returns:
            The raw response dictionary returned by the app-server for the `thread/list`
            request.
        """
        params: Dict[str, Any] = {}
        if cursor is not None:
            params["cursor"] = cursor
        if limit is not None:
            params["limit"] = limit
        if sort_key is not None:
            params["sort_key"] = sort_key
        if model_providers is not None:
            params["model_providers"] = list(model_providers)
        if source_kinds is not None:
            params["source_kinds"] = list(source_kinds)
        if archived is not None:
            params["archived"] = archived
        if cwd is not None:
            params["cwd"] = str(cwd)
        return await self._request_dict("thread/list", _coerce_keys(params) or None)

    async def thread_read(
        self, thread_id: str, *, include_turns: bool = False
    ) -> Dict[str, Any]:
        """Read a thread by id from the app-server."""
        payload = {"thread_id": thread_id, "include_turns": include_turns}
        return await self._request_dict("thread/read", _coerce_keys(payload))

    async def thread_archive(self, thread_id: str) -> Dict[str, Any]:
        """
        Archive the thread identified by `thread_id`.

        Args:
            thread_id: Identifier of the thread to archive.

        Returns:
            The app-server's response payload for the archive operation.
        """
        return await self._request_dict("thread/archive", {"threadId": thread_id})

    async def thread_name_set(self, thread_id: str, *, name: str) -> Dict[str, Any]:
        """
        Set the display name for a thread.

        Args:
            thread_id: Identifier of the thread to rename.
            name: New name to assign to the thread.

        Returns:
            Response payload returned by the app-server.
        """
        return await self._request_dict(
            "thread/name/set", {"threadId": thread_id, "name": name}
        )

    async def thread_unarchive(self, thread_id: str) -> Dict[str, Any]:
        """
        Unarchives the thread identified by `thread_id`.

        Args:
            thread_id: Identifier of the thread to unarchive.

        Returns:
            Response dictionary from the app-server for the unarchive operation.
        """
        return await self._request_dict("thread/unarchive", {"threadId": thread_id})

    async def thread_compact_start(
        self, thread_id: str, *, instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Starts a compaction operation for the specified thread on the app-server.

        Args:
            thread_id: Identifier of the thread to compact.
            instructions: Optional server hint for compaction behavior.

        Returns:
            dict: The app-server's result payload for the compaction start request.
        """
        payload: Dict[str, Any] = {"thread_id": thread_id}
        if instructions is not None:
            payload["instructions"] = instructions
        return await self._request_dict("thread/compact/start", _coerce_keys(payload))

    async def thread_background_terminals_clean(
        self, thread_id: str, *, terminal_ids: Sequence[str]
    ) -> Dict[str, Any]:
        """
        Clean up background terminal sessions for a thread.

        Args:
            thread_id: Identifier of the thread.
            terminal_ids: Terminal ids to clean.

        Returns:
            App-server response payload.
        """
        payload = {"thread_id": thread_id, "terminal_ids": list(terminal_ids)}
        return await self._request_dict(
            "thread/backgroundTerminals/clean", _coerce_keys(payload)
        )

    async def thread_rollback(
        self, thread_id: str, *, num_turns: int
    ) -> Dict[str, Any]:
        """
        Rolls back a thread by removing the specified number of turns.

        Args:
            thread_id: Identifier of the thread to roll back.
            num_turns: Number of most recent turns to remove from the thread.

        Returns:
            dict: Result dictionary returned by the app-server for the rollback operation.
        """
        return await self._request_dict(
            "thread/rollback", {"threadId": thread_id, "numTurns": num_turns}
        )

    async def config_requirements_read(self) -> Dict[str, Any]:
        """Read config requirements metadata from the app-server."""
        return await self._request_dict("configRequirements/read")

    async def config_read(
        self,
        *,
        include_layers: bool = False,
        cwd: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Read configuration from the app-server.

        Args:
            include_layers: If True, include configuration layer details in the response.
            cwd: Optional working directory used for config resolution.
        """
        params = {
            "include_layers": include_layers,
            "cwd": str(cwd) if cwd is not None else None,
        }
        return await self._request_dict("config/read", _coerce_keys(params))

    async def config_value_write(
        self,
        *,
        key_path: str,
        value: Any,
        merge_strategy: str,
        file_path: Optional[str] = None,
        expected_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Write a single configuration value via the app-server."""
        params = {
            "key_path": key_path,
            "value": value,
            "merge_strategy": merge_strategy,
            "file_path": file_path,
            "expected_version": expected_version,
        }
        return await self._request_dict("config/value/write", _coerce_keys(params))

    async def config_batch_write(
        self,
        *,
        edits: Sequence[Mapping[str, Any]],
        file_path: Optional[str] = None,
        expected_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Apply multiple configuration edits in a single app-server request."""
        params = {
            "edits": list(edits),
            "file_path": file_path,
            "expected_version": expected_version,
        }
        return await self._request_dict("config/batchWrite", _coerce_keys(params))

    async def skills_list(
        self,
        *,
        cwds: Optional[Sequence[Union[str, Path]]] = None,
        force_reload: bool = False,
    ) -> Dict[str, Any]:
        """
        List skills available to the app-server, optionally scoped to specific working directories.

        Args:
            cwds: Sequence of directory paths to scope the skills listing; each path will be
                converted to a string. If omitted, the server uses its default scope.
            force_reload: If true, instructs the server to reload skill data before returning
                results.

        Returns:
            The parsed response payload from the `skills/list` app-server method.
        """
        payload: Dict[str, Any] = {"force_reload": force_reload}
        if cwds:
            payload["cwds"] = [str(path) for path in cwds]
        return await self._request_dict("skills/list", _coerce_keys(payload))

    async def skills_remote_read(
        self,
        *,
        cwds: Optional[Sequence[Union[str, Path]]] = None,
        enabled: Optional[bool] = None,
        hazelnut_scope: Optional[str] = None,
        product_surface: Optional[str] = None,
        params: Optional["SkillsRemoteReadRequest"] = None,
    ) -> Dict[str, Any]:
        """
        Read remote skills metadata from the app server.

        Args:
            cwds: Optional workspace roots to scope the remote skill listing.
            enabled: Optional filter for enabled/disabled remote skills.
            hazelnut_scope: Optional Hazelnut scope identifier.
            product_surface: Optional product surface identifier.
            params: Optional raw request payload for protocol-forward fields.

        Returns:
            result (Dict[str, Any]): The app-server response payload for the `skills/remote/read` request.
        """
        payload: Dict[str, Any] = {}
        if params is not None:
            payload.update(dict(params))
        if cwds is not None:
            payload["cwds"] = [str(path) for path in cwds]
        if enabled is not None:
            payload["enabled"] = enabled
        if hazelnut_scope is not None:
            payload["hazelnut_scope"] = hazelnut_scope
        if product_surface is not None:
            payload["product_surface"] = product_surface
        return await self._request_dict("skills/remote/read", _coerce_keys(payload))

    async def skills_remote_write(
        self,
        *,
        hazelnut_id: Optional[str] = None,
        is_preload: Optional[bool] = None,
        params: Optional["SkillsRemoteWriteRequest"] = None,
    ) -> Dict[str, Any]:
        """
        Start a remote skill write operation.

        Args:
            hazelnut_id: Optional Hazelnut identifier.
            is_preload: Optional preload flag.
            params: Optional raw request payload.

        Returns:
            Result returned by the app-server for the "skills/remote/write" request.
        """
        payload: Dict[str, Any] = {}
        if params is not None:
            payload.update(dict(params))
        if hazelnut_id is not None:
            payload["hazelnut_id"] = hazelnut_id
        if is_preload is not None:
            payload["is_preload"] = is_preload
        return await self._request_dict("skills/remote/write", _coerce_keys(payload))

    async def skills_config_write(
        self,
        *,
        path: Optional[str] = None,
        enabled: Optional[bool] = None,
        params: Optional["SkillsConfigWriteRequest"] = None,
    ) -> Dict[str, Any]:
        """
        Set skill configuration state.

        Args:
            path: Optional configuration path identifying the skill.
            enabled: Optional enabled state for the skill.
            params: Optional typed request payload for evolving protocol fields.

        Returns:
            The app-server response as a dictionary.
        """
        # TODO(app-server-schema): tighten request shape after protocol stabilizes.
        payload: Dict[str, Any] = {}
        if params is not None:
            payload.update(_coerce_keys(dict(params)))
        if path is not None:
            payload["path"] = path
        if enabled is not None:
            payload["enabled"] = enabled
        return await self._request_dict("skills/config/write", payload)

    async def turn_start(
        self,
        thread_id: str,
        input: AppServerInput,
        **params: Any,
    ) -> Dict[str, Any]:
        """
        Start a new turn in the specified thread using the provided user input.

        Args:
            thread_id: Identifier of the thread to start the turn in.
            input: User input for the turn; may be a string or a sequence of input items and
                will be normalized to the app-server format.
            **params: Additional optional request parameters; keys with None values are omitted
                and snake_case keys are converted to camelCase.

        Returns:
            The app-server's response payload for the started turn.
        """
        payload = {"threadId": thread_id, "input": normalize_app_server_input(input)}
        payload.update(_coerce_keys(params))
        return await self._request_dict("turn/start", payload)

    async def review_start(
        self,
        thread_id: str,
        *,
        target: Mapping[str, Any],
        delivery: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start a review run for the given thread and target payload."""
        payload: Dict[str, Any] = {"thread_id": thread_id, "target": dict(target)}
        if delivery is not None:
            payload["delivery"] = delivery
        return await self._request_dict("review/start", _coerce_keys(payload))

    async def turn_session(
        self,
        thread_id: str,
        input: AppServerInput,
        *,
        approvals: Optional[ApprovalDecisions] = None,
        **params: Any,
    ) -> AppServerTurnSession:
        """Start a turn and return a session wrapper that streams notifications."""
        result = await self.turn_start(thread_id, input, **params)
        turn = result.get("turn") if isinstance(result, dict) else None
        turn_id = None
        if isinstance(turn, dict):
            turn_id = turn.get("id")
        if not isinstance(turn_id, str) or not turn_id:
            raise CodexError("turn/start response missing turn id")
        session = AppServerTurnSession(
            self,
            thread_id=thread_id,
            turn_id=turn_id,
            approvals=approvals,
            initial_turn=turn,
        )
        await session.start()
        return session

    async def turn_interrupt(self, thread_id: str, turn_id: str) -> Dict[str, Any]:
        """Interrupt an in-progress turn."""
        return await self._request_dict(
            "turn/interrupt", {"threadId": thread_id, "turnId": turn_id}
        )

    async def turn_steer(
        self, thread_id: str, turn_id: str, *, prompt: str
    ) -> Dict[str, Any]:
        """
        Send steering guidance to an in-progress turn.

        Args:
            thread_id: Identifier of the thread.
            turn_id: Identifier of the turn.
            prompt: Steering prompt text.

        Returns:
            App-server response payload.
        """
        payload = {"thread_id": thread_id, "turn_id": turn_id, "prompt": prompt}
        return await self._request_dict("turn/steer", _coerce_keys(payload))

    async def model_list(
        self,
        *,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
        include_hidden: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """List models available to the app-server."""
        params: Dict[str, Any] = {}
        if cursor is not None:
            params["cursor"] = cursor
        if limit is not None:
            params["limit"] = limit
        if include_hidden is not None:
            params["include_hidden"] = include_hidden
        return await self._request_dict("model/list", _coerce_keys(params) or None)

    async def app_list(
        self, *, cursor: Optional[str] = None, limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """List apps/connectors available to the app-server."""
        params: Dict[str, Any] = {}
        if cursor is not None:
            params["cursor"] = cursor
        if limit is not None:
            params["limit"] = limit
        return await self._request_dict("app/list", params or None)

    async def collaboration_mode_list(self) -> Dict[str, Any]:
        """List supported collaboration modes from the app-server."""
        return await self._request_dict("collaborationMode/list", {})

    async def experimental_feature_list(
        self, *, cursor: Optional[str] = None, limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        List experimental features available from the app-server.

        Args:
            cursor: Optional pagination cursor.
            limit: Optional page size.

        Returns:
            App-server response payload.
        """
        params: Dict[str, Any] = {}
        if cursor is not None:
            params["cursor"] = cursor
        if limit is not None:
            params["limit"] = limit
        return await self._request_dict("experimentalFeature/list", params or None)

    async def command_exec(
        self,
        *,
        command: Sequence[str],
        timeout_ms: Optional[int] = None,
        cwd: Optional[Union[str, Path]] = None,
        sandbox_policy: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a command via the app-server command execution endpoint."""
        params: Dict[str, Any] = {"command": list(command)}
        if timeout_ms is not None:
            params["timeout_ms"] = timeout_ms
        if cwd is not None:
            params["cwd"] = str(cwd)
        if sandbox_policy is not None:
            params["sandbox_policy"] = dict(sandbox_policy)
        return await self._request_dict("command/exec", _coerce_keys(params))

    async def mcp_server_oauth_login(
        self, *, name: str, scopes: Optional[Sequence[str]] = None
    ) -> Dict[str, Any]:
        """Start an MCP server OAuth login flow."""
        params: Dict[str, Any] = {"name": name}
        if scopes is not None:
            params["scopes"] = list(scopes)
        return await self._request_dict("mcpServer/oauth/login", _coerce_keys(params))

    async def mcp_server_refresh(self) -> Dict[str, Any]:
        """Refresh MCP server configuration/status."""
        return await self._request_dict("config/mcpServer/reload")

    async def mcp_server_status_list(
        self, *, cursor: Optional[str] = None, limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """List MCP server status entries with optional pagination."""
        params: Dict[str, Any] = {}
        if cursor is not None:
            params["cursor"] = cursor
        if limit is not None:
            params["limit"] = limit
        return await self._request_dict("mcpServerStatus/list", params or None)

    async def account_login_start(self, *, params: Mapping[str, Any]) -> Dict[str, Any]:
        """Start an account login flow via the app-server."""
        return await self._request_dict("account/login/start", dict(params))

    async def account_login_cancel(self, *, login_id: str) -> Dict[str, Any]:
        """Cancel an in-progress login flow."""
        return await self._request_dict("account/login/cancel", {"loginId": login_id})

    async def account_logout(self) -> Dict[str, Any]:
        """Log out of the current account session."""
        return await self._request_dict("account/logout")

    async def account_rate_limits_read(self) -> Dict[str, Any]:
        """Read current account rate limits from the app-server."""
        return await self._request_dict("account/rateLimits/read")

    async def account_read(self, *, refresh_token: bool = False) -> Dict[str, Any]:
        """Read account information, optionally refreshing the token."""
        return await self._request_dict(
            "account/read", {"refreshToken": refresh_token} if refresh_token else None
        )

    async def account_chatgpt_auth_tokens_refresh(
        self, *, params: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """
        Refresh ChatGPT auth tokens via the app-server.

        Args:
            params: Refresh payload (snake_case or camelCase keys are accepted).

        Returns:
            App-server response payload.
        """
        return await self._request_dict(
            "account/chatgptAuthTokens/refresh", _coerce_keys(dict(params))
        )

    async def item_tool_call(self, *, params: "ItemToolCallRequest") -> Dict[str, Any]:
        """
        Send an item tool-call payload.

        Args:
            params: Typed request payload for `item/tool/call`.

        Returns:
            App-server response payload.
        """
        # TODO(app-server-schema): tighten request shape after protocol stabilizes.
        return await self._request_dict("item/tool/call", _coerce_keys(dict(params)))

    async def item_tool_request_user_input(
        self, *, params: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """
        Send an item request-user-input payload.

        Args:
            params: Request payload for `item/tool/requestUserInput`.

        Returns:
            App-server response payload.
        """
        return await self._request_dict(
            "item/tool/requestUserInput", _coerce_keys(dict(params))
        )

    async def item_command_execution_request_approval(
        self, *, params: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """
        Send an item command-execution approval payload.

        Args:
            params: Request payload for `item/commandExecution/requestApproval`.

        Returns:
            App-server response payload.
        """
        return await self._request_dict(
            "item/commandExecution/requestApproval", _coerce_keys(dict(params))
        )

    async def item_file_change_request_approval(
        self, *, params: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """
        Send an item file-change approval payload.

        Args:
            params: Request payload for `item/fileChange/requestApproval`.

        Returns:
            App-server response payload.
        """
        return await self._request_dict(
            "item/fileChange/requestApproval", _coerce_keys(dict(params))
        )

    async def mock_experimental_method(
        self, *, params: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Call a mock experimental app-server endpoint.

        Args:
            params: Optional request payload.

        Returns:
            App-server response payload.
        """
        if not self._options.experimental_api_enabled:
            raise CodexError(
                "`mock/experimentalMethod` requires "
                "AppServerOptions(experimental_api_enabled=True)."
            )
        payload = _coerce_keys(dict(params)) if params is not None else {}
        return await self._request_dict("mock/experimentalMethod", payload)

    async def feedback_upload(
        self,
        *,
        classification: str,
        reason: Optional[str] = None,
        thread_id: Optional[str] = None,
        include_logs: bool = False,
    ) -> Dict[str, Any]:
        """Upload user feedback to the app-server."""
        params = {
            "classification": classification,
            "reason": reason,
            "thread_id": thread_id,
            "include_logs": include_logs,
        }
        return await self._request_dict("feedback/upload", _coerce_keys(params))

    def _ensure_ready(self) -> None:
        if self._process is None:
            raise CodexError("App-server process is not running")

    async def _send(self, payload: Dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise CodexError("App-server stdin is not available")
        message = json.dumps({k: v for k, v in payload.items() if v is not None})
        self._process.stdin.write(message.encode("utf-8") + b"\n")
        await self._process.stdin.drain()

    async def _reader_loop(self) -> None:
        if self._process is None or self._process.stdout is None:
            return
        try:
            async for line in _iter_lines(self._process.stdout):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CodexParseError(
                        f"Failed to parse app-server message: {line}"
                    ) from exc

                if isinstance(data, dict) and "id" in data and "method" in data:
                    await self._requests.put(
                        AppServerRequest(
                            id=data.get("id"),
                            method=str(data.get("method")),
                            params=data.get("params"),
                        )
                    )
                elif isinstance(data, dict) and "id" in data:
                    await self._handle_response(data)
                elif isinstance(data, dict) and "method" in data:
                    await self._notifications.put(
                        AppServerNotification(
                            method=str(data.get("method")),
                            params=data.get("params"),
                        )
                    )
                else:
                    raise CodexParseError(f"Unknown app-server message: {data}")
        except Exception as exc:  # pragma: no cover - defensive
            self._reader_error = exc
            self._fail_all_pending(exc)
        finally:
            self._closed = True
            await self._notifications.put(None)
            await self._requests.put(None)

    async def _handle_response(self, data: Dict[str, Any]) -> None:
        req_id = data.get("id")
        if not isinstance(req_id, int):
            # String ids are valid, but this client only issues ints.
            return
        future = self._pending.pop(req_id, None)
        if future is None:
            return
        if "error" in data:
            error = data.get("error") or {}
            code = error.get("code", -1)
            message = error.get("message", "Unknown error")
            raise_exc = CodexAppServerError(
                code=code, message=message, data=error.get("data")
            )
            future.set_exception(raise_exc)
            return
        future.set_result(data.get("result"))

    def _fail_all_pending(self, exc: BaseException) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(exc)
        self._pending.clear()

    def _resolve_executable(self) -> str:
        from .exec import CodexExec

        exec = CodexExec(self._options.codex_path_override, env=self._options.env)
        return exec.executable_path

    def _build_env(self) -> Dict[str, str]:
        if self._options.env is not None:
            env = dict(self._options.env)
        else:
            import os

            env = os.environ.copy()
        if INTERNAL_ORIGINATOR_ENV not in env:
            env[INTERNAL_ORIGINATOR_ENV] = PYTHON_SDK_ORIGINATOR
        if self._options.base_url:
            env["OPENAI_BASE_URL"] = self._options.base_url
        if self._options.api_key:
            env["CODEX_API_KEY"] = self._options.api_key
        return env

    def _default_client_info(self) -> AppServerClientInfo:
        from . import __version__

        return AppServerClientInfo(
            name="codex_sdk_python",
            title="Codex SDK Python",
            version=__version__,
        )


class AppServerByteRange(TypedDict):
    """Byte range for a text element (inclusive/exclusive semantics are server-defined)."""

    start: int
    end: int


class AppServerTextElement(TypedDict, total=False):
    """Text element metadata for app-server inputs."""

    byte_range: AppServerByteRange
    byteRange: AppServerByteRange
    placeholder: Optional[str]


class AppServerTextInput(TypedDict, total=False):
    """Text input item for the app-server protocol."""

    type: str
    text: str
    text_elements: List[AppServerTextElement]
    textElements: List[AppServerTextElement]


class AppServerImageInput(TypedDict):
    """Remote image input item for the app-server protocol."""

    type: str
    url: str


class AppServerLocalImageInput(TypedDict):
    """Local image input item for the app-server protocol."""

    type: str
    path: str


class AppServerSkillInput(TypedDict):
    """Skill input item for the app-server protocol."""

    type: str
    name: str
    path: str


class SkillsConfigWriteRequest(TypedDict, total=False):
    """Typed payload for `skills/config/write` requests."""

    path: str
    enabled: bool
    mode: str


class SkillsRemoteReadRequest(TypedDict, total=False):
    """Typed payload for `skills/remote/read` requests."""

    cwds: List[str]
    enabled: bool
    hazelnut_scope: str
    hazelnutScope: str
    product_surface: str
    productSurface: str


class SkillsRemoteWriteRequest(TypedDict, total=False):
    """Typed payload for `skills/remote/write` requests."""

    hazelnut_id: str
    hazelnutId: str
    is_preload: bool
    isPreload: bool


class ItemToolCallRequest(TypedDict, total=False):
    """Typed payload for `item/tool/call` requests."""

    name: str
    tool_name: str
    toolName: str
    tool_call_id: str
    toolCallId: str
    arguments: Mapping[str, Any]
    args: Mapping[str, Any]


AppServerUserInput = Union[
    AppServerTextInput,
    AppServerImageInput,
    AppServerLocalImageInput,
    AppServerSkillInput,
    Mapping[str, Any],
]
AppServerInput = Union[Sequence[AppServerUserInput], str]


def normalize_app_server_input(input: AppServerInput) -> List[Dict[str, Any]]:
    """Normalize supported SDK inputs into the canonical app-server wire shape."""
    if isinstance(input, str):
        return [{"type": "text", "text": input}]

    items: List[Dict[str, Any]] = []
    for raw in input:
        if not isinstance(raw, Mapping):
            raise CodexError("App-server input items must be mappings")
        item = dict(raw)
        item_type = item.get("type")
        if item_type == "local_image":
            item["type"] = "localImage"
            item_type = "localImage"
        if item_type == "text":
            _normalize_text_elements(item)
        if item_type == "localImage" and isinstance(item.get("path"), Path):
            item["path"] = str(item["path"])
        if item_type == "skill" and isinstance(item.get("path"), Path):
            item["path"] = str(item["path"])
        items.append(item)

    return items


def _normalize_text_elements(item: Dict[str, Any]) -> None:
    """Normalize snake_case metadata keys to the app-server's camelCase shape."""
    elements = None
    if isinstance(item.get("textElements"), list):
        elements = item.get("textElements")
    elif isinstance(item.get("text_elements"), list):
        elements = item.pop("text_elements")

    if elements is None:
        return

    normalized: List[Any] = []
    for element in elements:
        if isinstance(element, Mapping):
            entry = dict(element)
            if "byte_range" in entry and "byteRange" not in entry:
                entry["byteRange"] = entry.pop("byte_range")
            normalized.append(entry)
        else:
            normalized.append(element)

    item["textElements"] = normalized


def _coerce_keys(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Coerce snake_case keys to camelCase and drop None values."""
    coerced: Dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            continue
        if "_" in key:
            key = _snake_to_camel(key)
        coerced[key] = value
    return coerced


def _snake_to_camel(value: str) -> str:
    """Convert snake_case strings to lowerCamelCase."""
    parts = value.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


def _normalize_decision(
    decision: Union[str, Mapping[str, Any]],
    execpolicy_amendment: Optional[Mapping[str, Any]],
) -> Union[str, Dict[str, Any]]:
    """Normalize approval decisions into the JSON-RPC protocol encoding."""
    if isinstance(decision, Mapping):
        return dict(decision)
    if not isinstance(decision, str):
        raise CodexError("Approval decision must be a string or mapping")

    normalized = decision.strip()
    if normalized in {
        "accept_with_execpolicy_amendment",
        "acceptWithExecpolicyAmendment",
    }:
        if execpolicy_amendment is None:
            raise CodexError(
                "execpolicy_amendment is required for accept_with_execpolicy_amendment"
            )
        amendment_payload = _coerce_keys(execpolicy_amendment)
        return {
            "acceptWithExecpolicyAmendment": {"execpolicyAmendment": amendment_payload}
        }

    if "_" in normalized:
        normalized = _snake_to_camel(normalized)
    return normalized


def _extract_turn(notification: AppServerNotification) -> Optional[Dict[str, Any]]:
    """Extract a turn object from a notification payload (best effort)."""
    params = notification.params
    if not isinstance(params, dict):
        return None
    turn = params.get("turn")
    if isinstance(turn, dict):
        return turn
    if "id" in params:
        return params
    return None


async def _drain_stream(stream: asyncio.StreamReader, sink: list[str]) -> None:
    """Continuously read a stream and append decoded lines to sink."""
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
