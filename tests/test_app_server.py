import asyncio
import contextlib
import json
from typing import Any, List, Optional

import pytest

from codex_sdk.app_server import (
    AppServerClient,
    AppServerClientInfo,
    AppServerNotification,
    AppServerOptions,
    AppServerRequest,
    _coerce_keys,
    _drain_stream,
    _extract_turn,
    _iter_lines,
    _normalize_decision,
    normalize_app_server_input,
)
from codex_sdk.exceptions import CodexAppServerError, CodexError
from codex_sdk.exec import INTERNAL_ORIGINATOR_ENV, PYTHON_SDK_ORIGINATOR


class QueueStream:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()

    async def readline(self) -> bytes:
        return await self._queue.get()

    def feed(self, line: str) -> None:
        self._queue.put_nowait(line.encode("utf-8"))


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
        self, stdout: QueueStream, stderr: Optional[QueueStream] = None
    ) -> None:
        self.stdin = FakeStdin()
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = None

    async def wait(self) -> int:
        self.returncode = 0
        return 0

    def terminate(self) -> None:
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = 0


@pytest.mark.asyncio
async def test_app_server_initialize_and_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = QueueStream()
    process = FakeProcess(stdout)

    async def fake_spawn(*_cmd: Any, **_kwargs: Any) -> FakeProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    client = AppServerClient(
        AppServerOptions(
            auto_initialize=True,
            client_info=AppServerClientInfo(
                name="codex_sdk_python", title="Codex SDK Python", version="0.0.0"
            ),
        )
    )

    start_task = asyncio.create_task(client.start())
    await asyncio.sleep(0)

    init_request = json.loads(process.stdin.writes[0].decode("utf-8"))
    assert init_request["method"] == "initialize"
    assert "capabilities" not in init_request.get("params", {})

    stdout.feed('{"id":1,"result":{"userAgent":"codex"}}')
    await start_task

    initialized = json.loads(process.stdin.writes[1].decode("utf-8"))
    assert initialized["method"] == "initialized"

    request_task = asyncio.create_task(
        client.request("thread/loaded/list", {"limit": 1})
    )
    await asyncio.sleep(0)

    stdout.feed('{"id":2,"result":{"data":["thr_1"]}}')
    result = await request_task
    assert result["data"] == ["thr_1"]

    stdout.feed('{"method":"thread/started","params":{"thread":{"id":"thr_2"}}}')
    notification = await client.next_notification()
    assert notification is not None
    assert notification.method == "thread/started"

    await client.close()


@pytest.mark.asyncio
async def test_app_server_initialize_with_experimental_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = QueueStream()
    process = FakeProcess(stdout)

    async def fake_spawn(*_cmd: Any, **_kwargs: Any) -> FakeProcess:
        """
        Helper used in tests to simulate spawning a subprocess by returning a preconfigured FakeProcess.

        Returns:
            FakeProcess: The fake process instance to be used as the spawned subprocess.
        """
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    client = AppServerClient(
        AppServerOptions(auto_initialize=False, experimental_api_enabled=True)
    )
    await client.start()

    init_task = asyncio.create_task(client.initialize())
    await asyncio.sleep(0)
    init_request = json.loads(process.stdin.writes[-1].decode("utf-8"))
    assert init_request["method"] == "initialize"
    assert init_request["params"]["capabilities"] == {"experimentalApi": True}

    stdout.feed('{"id":1,"result":{"userAgent":"codex"}}')
    await init_task
    await client.close()


@pytest.mark.asyncio
async def test_app_server_error_response(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = QueueStream()
    process = FakeProcess(stdout)

    async def fake_spawn(*_cmd: Any, **_kwargs: Any) -> FakeProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    client = AppServerClient(
        AppServerOptions(
            auto_initialize=True,
            client_info=AppServerClientInfo(
                name="codex_sdk_python", title="Codex SDK Python", version="0.0.0"
            ),
        )
    )

    start_task = asyncio.create_task(client.start())
    await asyncio.sleep(0)

    stdout.feed('{"id":1,"result":{"userAgent":"codex"}}')
    await start_task

    request_task = asyncio.create_task(client.request("thread/loaded/list"))
    await asyncio.sleep(0)

    stdout.feed('{"id":2,"error":{"code":400,"message":"boom"}}')

    with pytest.raises(CodexAppServerError):
        await request_task

    await client.close()


@pytest.mark.asyncio
async def test_app_server_notify_and_respond(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = QueueStream()
    process = FakeProcess(stdout)

    async def fake_spawn(*_cmd: Any, **_kwargs: Any) -> FakeProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    client = AppServerClient(AppServerOptions(auto_initialize=False))
    await client.start()

    await client.notify("ping", {"ok": True})
    payload = json.loads(process.stdin.writes[-1].decode("utf-8"))
    assert payload["method"] == "ping"

    await client.respond(5, result={"ok": True})
    payload = json.loads(process.stdin.writes[-1].decode("utf-8"))
    assert payload["id"] == 5
    assert payload["result"] == {"ok": True}

    await client.respond(
        6, error=CodexAppServerError(code=400, message="nope", data={"x": 1})
    )
    payload = json.loads(process.stdin.writes[-1].decode("utf-8"))
    assert payload["error"]["code"] == 400
    assert payload["error"]["message"] == "nope"

    stdout.feed('{"id":7,"method":"item/commandExecution/requestApproval","params":{}}')
    stdout.feed('{"method":"thread/started","params":{"thread":{"id":"thr"}}}')

    request = await client.next_request()
    assert request is not None
    assert request.method == "item/commandExecution/requestApproval"

    notification = await client.next_notification()
    assert notification is not None
    assert notification.method == "thread/started"

    await client.close()


@pytest.mark.asyncio
async def test_app_server_methods_and_input_normalization(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    stdout = QueueStream()
    process = FakeProcess(stdout)

    async def fake_spawn(*_cmd: Any, **_kwargs: Any) -> FakeProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)
    client = AppServerClient(AppServerOptions(auto_initialize=False))
    await client.start()

    async def expect_request(task: asyncio.Task, expected_method: str, result: Any):
        await asyncio.sleep(0)
        payload = json.loads(process.stdin.writes[-1].decode("utf-8"))
        assert payload["method"] == expected_method
        stdout.feed(json.dumps({"id": payload["id"], "result": result}))
        return await task, payload

    task = asyncio.create_task(
        client.thread_start(model="gpt-5", approval_policy="on-request")
    )
    _, payload = await expect_request(task, "thread/start", {"thread": {"id": "t1"}})
    assert payload["params"]["approvalPolicy"] == "on-request"

    task = asyncio.create_task(client.thread_resume("t1"))
    await expect_request(task, "thread/resume", {"thread": {"id": "t1"}})

    task = asyncio.create_task(client.thread_fork("t1"))
    await expect_request(task, "thread/fork", {"thread": {"id": "t2"}})

    task = asyncio.create_task(client.thread_loaded_list(cursor="c", limit=1))
    _, payload = await expect_request(task, "thread/loaded/list", {"data": ["t1"]})
    assert payload["params"] == {"cursor": "c", "limit": 1}

    task = asyncio.create_task(client.config_requirements_read())
    _, payload = await expect_request(
        task, "configRequirements/read", {"requirements": None}
    )
    assert "params" not in payload

    input_items = [
        {"type": "text", "text": "hi"},
        {"type": "local_image", "path": tmp_path / "img.png"},
        {"type": "skill", "name": "skill", "path": tmp_path / "SKILL.md"},
    ]
    task = asyncio.create_task(client.turn_start("t1", input_items))
    _, payload = await expect_request(task, "turn/start", {"turn": {"id": "turn_1"}})
    sent_input = payload["params"]["input"]
    assert sent_input[1]["type"] == "localImage"
    assert isinstance(sent_input[1]["path"], str)
    assert isinstance(sent_input[2]["path"], str)

    task = asyncio.create_task(client.turn_interrupt("t1", "turn_1"))
    _, payload = await expect_request(task, "turn/interrupt", {})
    assert payload["params"] == {"threadId": "t1", "turnId": "turn_1"}

    task = asyncio.create_task(
        client.thread_list(
            cursor="c2",
            limit=2,
            sort_key="updated_at",
            model_providers=["openai"],
            source_kinds=["cli"],
            archived=True,
        )
    )
    _, payload = await expect_request(
        task, "thread/list", {"data": [], "nextCursor": None}
    )
    assert payload["params"]["sortKey"] == "updated_at"
    assert payload["params"]["modelProviders"] == ["openai"]
    assert payload["params"]["sourceKinds"] == ["cli"]
    assert payload["params"]["archived"] is True

    task = asyncio.create_task(client.thread_read("t1", include_turns=True))
    _, payload = await expect_request(task, "thread/read", {"thread": {"id": "t1"}})
    assert payload["params"] == {"threadId": "t1", "includeTurns": True}

    task = asyncio.create_task(client.thread_archive("t1"))
    _, payload = await expect_request(task, "thread/archive", {})
    assert payload["params"] == {"threadId": "t1"}

    task = asyncio.create_task(client.thread_name_set("t1", name="hello"))
    _, payload = await expect_request(task, "thread/name/set", {})
    assert payload["params"] == {"threadId": "t1", "name": "hello"}

    task = asyncio.create_task(client.thread_unarchive("t1"))
    _, payload = await expect_request(
        task, "thread/unarchive", {"thread": {"id": "t1"}}
    )
    assert payload["params"] == {"threadId": "t1"}

    task = asyncio.create_task(client.thread_compact_start("t1"))
    _, payload = await expect_request(task, "thread/compact/start", {})
    assert payload["params"] == {"threadId": "t1"}

    task = asyncio.create_task(client.thread_rollback("t1", num_turns=2))
    _, payload = await expect_request(task, "thread/rollback", {"thread": {"id": "t1"}})
    assert payload["params"] == {"threadId": "t1", "numTurns": 2}

    task = asyncio.create_task(client.config_read(include_layers=True, cwd=tmp_path))
    _, payload = await expect_request(
        task, "config/read", {"config": {}, "origins": {}}
    )
    assert payload["params"] == {"includeLayers": True, "cwd": str(tmp_path)}

    task = asyncio.create_task(
        client.config_value_write(
            key_path="analytics.enabled",
            value=True,
            merge_strategy="set",
            file_path="/tmp/config.toml",
            expected_version="v1",
        )
    )
    _, payload = await expect_request(task, "config/value/write", {"ok": True})
    assert payload["params"]["keyPath"] == "analytics.enabled"

    task = asyncio.create_task(
        client.config_batch_write(
            edits=[
                {"keyPath": "analytics.enabled", "value": True, "mergeStrategy": "set"}
            ]
        )
    )
    _, payload = await expect_request(task, "config/batchWrite", {"ok": True})
    assert isinstance(payload["params"]["edits"], list)

    task = asyncio.create_task(client.skills_list(cwds=[tmp_path], force_reload=True))
    _, payload = await expect_request(task, "skills/list", {"data": []})
    assert payload["params"]["forceReload"] is True

    task = asyncio.create_task(client.skills_remote_read())
    _, payload = await expect_request(task, "skills/remote/read", {"data": []})
    assert payload["params"] == {}

    task = asyncio.create_task(
        client.skills_remote_write(hazelnut_id="hazelnut_1", is_preload=True)
    )
    _, payload = await expect_request(task, "skills/remote/write", {"ok": True})
    assert payload["params"] == {"hazelnutId": "hazelnut_1", "isPreload": True}

    task = asyncio.create_task(client.skills_config_write(path="foo", enabled=True))
    _, payload = await expect_request(task, "skills/config/write", {"ok": True})
    assert payload["params"] == {"path": "foo", "enabled": True}

    task = asyncio.create_task(
        client.review_start(
            "t1",
            target={"type": "uncommittedChanges"},
            delivery="inline",
        )
    )
    _, payload = await expect_request(task, "review/start", {"turn": {"id": "turn_r"}})
    assert payload["params"]["threadId"] == "t1"

    task = asyncio.create_task(client.model_list(cursor="m", limit=1))
    _, payload = await expect_request(task, "model/list", {"data": []})
    assert payload["params"] == {"cursor": "m", "limit": 1}

    task = asyncio.create_task(client.app_list(cursor="a", limit=2))
    _, payload = await expect_request(task, "app/list", {"data": []})
    assert payload["params"] == {"cursor": "a", "limit": 2}

    task = asyncio.create_task(client.collaboration_mode_list())
    _, payload = await expect_request(task, "collaborationMode/list", {"data": []})
    assert payload["params"] == {}

    task = asyncio.create_task(
        client.command_exec(
            command=["echo", "hi"],
            timeout_ms=10,
            cwd=tmp_path,
            sandbox_policy={"allow": True},
        )
    )
    _, payload = await expect_request(task, "command/exec", {"exitCode": 0})
    assert payload["params"]["command"] == ["echo", "hi"]
    assert payload["params"]["sandboxPolicy"] == {"allow": True}

    task = asyncio.create_task(
        client.mcp_server_oauth_login(name="server", scopes=["a"])
    )
    _, payload = await expect_request(
        task, "mcpServer/oauth/login", {"authorizationUrl": "x"}
    )
    assert payload["params"]["name"] == "server"

    task = asyncio.create_task(client.mcp_server_refresh())
    _, payload = await expect_request(task, "config/mcpServer/reload", {})
    assert "params" not in payload

    task = asyncio.create_task(client.mcp_server_status_list(cursor="c", limit=1))
    _, payload = await expect_request(task, "mcpServerStatus/list", {"data": []})
    assert payload["params"] == {"cursor": "c", "limit": 1}

    task = asyncio.create_task(
        client.account_login_start(params={"type": "apiKey", "apiKey": "key"})
    )
    _, payload = await expect_request(task, "account/login/start", {"type": "apiKey"})
    assert payload["params"]["type"] == "apiKey"

    task = asyncio.create_task(client.account_login_cancel(login_id="login"))
    _, payload = await expect_request(
        task, "account/login/cancel", {"status": "canceled"}
    )
    assert payload["params"] == {"loginId": "login"}

    task = asyncio.create_task(client.account_logout())
    _, payload = await expect_request(task, "account/logout", {})
    assert "params" not in payload

    task = asyncio.create_task(client.account_rate_limits_read())
    _, payload = await expect_request(
        task, "account/rateLimits/read", {"rateLimits": {}}
    )
    assert "params" not in payload

    task = asyncio.create_task(client.account_read(refresh_token=True))
    _, payload = await expect_request(task, "account/read", {"account": {}})
    assert payload["params"]["refreshToken"] is True

    task = asyncio.create_task(
        client.feedback_upload(
            classification="bug",
            reason="test",
            thread_id="t1",
            include_logs=True,
        )
    )
    _, payload = await expect_request(task, "feedback/upload", {"threadId": "t1"})
    assert payload["params"]["classification"] == "bug"

    await client.close()


def test_app_server_helpers() -> None:
    coerced = _coerce_keys({"approval_policy": "on-request", "skip": None})
    assert coerced == {"approvalPolicy": "on-request"}

    assert normalize_app_server_input("hi") == [{"type": "text", "text": "hi"}]
    assert normalize_app_server_input([{"type": "text", "text": "ok"}]) == [
        {"type": "text", "text": "ok"}
    ]
    assert normalize_app_server_input(
        [
            {
                "type": "text",
                "text": "ok",
                "text_elements": [
                    {
                        "byte_range": {"start": 0, "end": 2},
                        "placeholder": "link",
                    }
                ],
            }
        ]
    ) == [
        {
            "type": "text",
            "text": "ok",
            "textElements": [
                {"byteRange": {"start": 0, "end": 2}, "placeholder": "link"}
            ],
        }
    ]
    assert normalize_app_server_input(
        [
            {
                "type": "text",
                "text": "ok",
                "textElements": [{"byteRange": {"start": 1, "end": 3}}],
            }
        ]
    ) == [
        {
            "type": "text",
            "text": "ok",
            "textElements": [{"byteRange": {"start": 1, "end": 3}}],
        }
    ]

    with pytest.raises(CodexError):
        normalize_app_server_input(["bad"])  # type: ignore[list-item]

    client = AppServerClient(
        AppServerOptions(
            base_url="https://example.com",
            api_key="key",
            env={INTERNAL_ORIGINATOR_ENV: "custom", "X": "1"},
        )
    )
    env = client._build_env()
    assert env["OPENAI_BASE_URL"] == "https://example.com"
    assert env["CODEX_API_KEY"] == "key"
    assert env["X"] == "1"
    assert env[INTERNAL_ORIGINATOR_ENV] == "custom"

    info = client._default_client_info()
    assert info.name == "codex_sdk_python"


def test_app_server_client_defaults_options() -> None:
    client = AppServerClient()
    assert isinstance(client._options, AppServerOptions)


def test_app_server_build_env_sets_originator_when_missing() -> None:
    client = AppServerClient(AppServerOptions(env={"X": "1"}))
    env = client._build_env()
    assert env["X"] == "1"
    assert env[INTERNAL_ORIGINATOR_ENV] == PYTHON_SDK_ORIGINATOR


def test_normalize_app_server_input_preserves_non_mapping_text_elements() -> None:
    assert normalize_app_server_input(
        [
            {
                "type": "text",
                "text": "ok",
                "text_elements": ["raw-element"],
            }
        ]
    ) == [
        {
            "type": "text",
            "text": "ok",
            "textElements": ["raw-element"],
        }
    ]


def test_extract_turn_returns_none_when_missing_turn_and_id() -> None:
    assert (
        _extract_turn(AppServerNotification(method="turn/completed", params={})) is None
    )


@pytest.mark.asyncio
async def test_app_server_reader_loop_skips_blank_lines_and_records_parse_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = QueueStream()
    process = FakeProcess(stdout)

    async def fake_spawn(*_cmd: Any, **_kwargs: Any) -> FakeProcess:
        """
        Helper used in tests to simulate spawning a subprocess by returning a preconfigured FakeProcess.

        Returns:
            FakeProcess: The fake process instance to be used as the spawned subprocess.
        """
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    client = AppServerClient(AppServerOptions(auto_initialize=False))
    await client.start()

    stdout.feed("\n")
    stdout.feed("not json")
    await asyncio.sleep(0)

    with pytest.raises(CodexError):
        await client.request("noop")

    await client.close()


@pytest.mark.asyncio
async def test_app_server_reader_loop_records_unknown_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = QueueStream()
    process = FakeProcess(stdout)

    async def fake_spawn(*_cmd: Any, **_kwargs: Any) -> FakeProcess:
        """
        Helper used in tests to simulate spawning a subprocess by returning a preconfigured FakeProcess.

        Returns:
            FakeProcess: The fake process instance to be used as the spawned subprocess.
        """
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    client = AppServerClient(AppServerOptions(auto_initialize=False))
    await client.start()

    stdout.feed("[]")
    await asyncio.sleep(0)

    with pytest.raises(CodexError):
        await client.request("noop")

    await client.close()


@pytest.mark.asyncio
async def test_app_server_close_fails_pending_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = QueueStream()
    process = FakeProcess(stdout)

    async def fake_spawn(*_cmd: Any, **_kwargs: Any) -> FakeProcess:
        """
        Helper used in tests to simulate spawning a subprocess by returning a preconfigured FakeProcess.

        Returns:
            FakeProcess: The fake process instance to be used as the spawned subprocess.
        """
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    client = AppServerClient(AppServerOptions(auto_initialize=False))
    await client.start()

    request_task = asyncio.create_task(client.request("thread/loaded/list"))
    await asyncio.sleep(0)

    await client.close()

    with pytest.raises(CodexError):
        await request_task


@pytest.mark.asyncio
async def test_app_server_close_kills_process_on_wait_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TimeoutProcess(FakeProcess):
        def __init__(self, stdout: QueueStream) -> None:
            """
            Initialize the instance with the provided QueueStream as its stdout and set the killed flag to False.

            Parameters:
                stdout (QueueStream): Stream used to emulate the process's stdout.
            """
            super().__init__(stdout)
            self.killed = False

        def kill(self) -> None:
            """
            Terminate the process and record that a kill was requested.

            Sets the instance's `killed` flag to True and then delegates to the superclass `kill` method to perform process termination.
            """
            self.killed = True
            super().kill()

    stdout = QueueStream()
    process = TimeoutProcess(stdout)

    async def fake_spawn(*_cmd: Any, **_kwargs: Any) -> FakeProcess:
        """
        Helper used in tests to simulate spawning a subprocess by returning a preconfigured FakeProcess.

        Returns:
            FakeProcess: The fake process instance to be used as the spawned subprocess.
        """
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    import codex_sdk.app_server as app_server_module

    async def fake_wait_for(awaitable: Any, *_args: Any, **_kwargs: Any) -> None:
        """
        Simulate asyncio.wait_for that always times out: schedule the given awaitable as a task, cancel it, and raise asyncio.TimeoutError.

        Parameters:
            awaitable: A coroutine or awaitable to be scheduled; it will be cancelled.
            *_args, **_kwargs: Ignored; present for API compatibility.
        """
        task = asyncio.create_task(awaitable)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        raise asyncio.TimeoutError()

    monkeypatch.setattr(app_server_module.asyncio, "wait_for", fake_wait_for)

    client = AppServerClient(AppServerOptions(auto_initialize=False))
    await client.start()
    await client.close()
    assert process.killed is True


@pytest.mark.asyncio
async def test_app_server_ensure_ready_and_handle_response() -> None:
    client = AppServerClient(AppServerOptions(auto_initialize=False))
    with pytest.raises(CodexError):
        await client.notify("nope")

    # Response with non-int id is ignored
    await client._handle_response({"id": "str"})
    # Response with no pending future is ignored
    await client._handle_response({"id": 9, "result": {"ok": True}})


@pytest.mark.asyncio
async def test_app_server_context_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = QueueStream()
    process = FakeProcess(stdout)

    async def fake_spawn(*_cmd: Any, **_kwargs: Any) -> FakeProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    async with AppServerClient(AppServerOptions(auto_initialize=False)) as client:
        assert client._process is process
        task = asyncio.create_task(client.request("noop"))
        await asyncio.sleep(0)
        stdout.feed('{"id":1,"result":{"ok":true}}')
        await task


@pytest.mark.asyncio
async def test_app_server_start_with_config_and_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = QueueStream()
    stderr = QueueStream()
    process = FakeProcess(stdout, stderr=stderr)
    captured = {}

    async def fake_spawn(*cmd: Any, **kwargs: Any) -> FakeProcess:
        captured["cmd"] = cmd
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    client = AppServerClient(
        AppServerOptions(
            auto_initialize=False, config_overrides={"analytics.enabled": True}
        )
    )
    await client.start()
    cmd_list = list(captured["cmd"])
    assert "--config" in cmd_list
    await client.close()


@pytest.mark.asyncio
async def test_app_server_start_requires_stdio(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProcessNoStdin(FakeProcess):
        def __init__(self, stdout: QueueStream) -> None:
            super().__init__(stdout)
            self.stdin = None

    stdout = QueueStream()
    process = FakeProcessNoStdin(stdout)

    async def fake_spawn(*_cmd: Any, **_kwargs: Any) -> FakeProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    client = AppServerClient(AppServerOptions(auto_initialize=False))
    with pytest.raises(CodexError):
        await client.start()


@pytest.mark.asyncio
async def test_app_server_close_no_process() -> None:
    client = AppServerClient(AppServerOptions(auto_initialize=False))
    await client.close()


@pytest.mark.asyncio
async def test_app_server_initialize_default_client_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = QueueStream()
    process = FakeProcess(stdout)

    async def fake_spawn(*_cmd: Any, **_kwargs: Any) -> FakeProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)
    client = AppServerClient(AppServerOptions(auto_initialize=False))
    await client.start()

    init_task = asyncio.create_task(client.initialize())
    await asyncio.sleep(0)
    init_request = json.loads(process.stdin.writes[-1].decode("utf-8"))
    stdout.feed(
        json.dumps({"id": init_request["id"], "result": {"userAgent": "codex"}})
    )
    result = await init_task
    assert result["userAgent"] == "codex"
    await client.close()


@pytest.mark.asyncio
async def test_app_server_start_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = QueueStream()
    process = FakeProcess(stdout)

    async def fake_spawn(*_cmd: Any, **_kwargs: Any) -> FakeProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)
    client = AppServerClient(AppServerOptions(auto_initialize=False))
    await client.start()
    await client.start()
    await client.close()


@pytest.mark.asyncio
async def test_app_server_request_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = QueueStream()
    process = FakeProcess(stdout)

    async def fake_spawn(*_cmd: Any, **_kwargs: Any) -> FakeProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)
    client = AppServerClient(
        AppServerOptions(auto_initialize=False, request_timeout=0.01)
    )
    await client.start()

    with pytest.raises(asyncio.TimeoutError):
        await client.request("slow")
    await client.close()


@pytest.mark.asyncio
async def test_app_server_reader_error(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = QueueStream()
    process = FakeProcess(stdout)

    async def fake_spawn(*_cmd: Any, **_kwargs: Any) -> FakeProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)
    client = AppServerClient(AppServerOptions(auto_initialize=False))
    await client.start()
    client._reader_error = RuntimeError("boom")
    with pytest.raises(CodexError):
        await client.request("noop")
    await client.close()


@pytest.mark.asyncio
async def test_app_server_generators(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = QueueStream()
    process = FakeProcess(stdout)

    async def fake_spawn(*_cmd: Any, **_kwargs: Any) -> FakeProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)
    client = AppServerClient(AppServerOptions(auto_initialize=False))
    await client.start()

    await client._notifications.put(AppServerNotification(method="ping", params=None))
    await client._notifications.put(None)
    await client._requests.put(AppServerRequest(id=1, method="req", params=None))
    await client._requests.put(None)

    notes = [note.method async for note in client.notifications()]
    reqs = [req.method async for req in client.requests()]
    assert notes == ["ping"]
    assert reqs == ["req"]
    await client.close()


@pytest.mark.asyncio
async def test_app_server_send_and_reader_loop_guards() -> None:
    client = AppServerClient(AppServerOptions(auto_initialize=False))
    client._process = FakeProcess(QueueStream())
    client._process.stdin = None  # type: ignore[assignment]
    with pytest.raises(CodexError):
        await client._send({"method": "noop"})

    client._process.stdout = None  # type: ignore[assignment]
    await client._reader_loop()


def test_app_server_decision_helpers() -> None:
    assert _normalize_decision("accept_for_session", None) == "acceptForSession"
    assert _normalize_decision({"accept": True}, None) == {"accept": True}
    with pytest.raises(CodexError):
        _normalize_decision(123, None)  # type: ignore[arg-type]
    with pytest.raises(CodexError):
        _normalize_decision("accept_with_execpolicy_amendment", None)


@pytest.mark.asyncio
async def test_app_server_extract_turn_and_stream_helpers():
    assert (
        _extract_turn(AppServerNotification(method="turn/completed", params=None))
        is None
    )
    assert _extract_turn(
        AppServerNotification(method="turn/completed", params={"id": "t"})
    ) == {"id": "t"}

    reader = asyncio.StreamReader()
    reader.feed_data(b"one\n")
    reader.feed_eof()
    lines = [line async for line in _iter_lines(reader)]
    assert lines == ["one"]

    reader2 = asyncio.StreamReader()
    reader2.feed_data(b"two\n")
    reader2.feed_eof()
    sink: List[str] = []
    await _drain_stream(reader2, sink)
    assert sink == ["two\n"]
