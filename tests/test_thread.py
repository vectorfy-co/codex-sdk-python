from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

pydantic = pytest.importorskip("pydantic")
BaseModel = pydantic.BaseModel

from codex_sdk import CodexOptions, ThreadOptions
from codex_sdk.events import Usage
from codex_sdk.exceptions import CodexError, CodexParseError, TurnFailedError
from codex_sdk.items import (
    AgentMessageItem,
    CommandExecutionItem,
    ErrorItem,
    FileChangeItem,
    FileUpdateChange,
    McpToolCallItem,
    McpToolCallItemError,
    McpToolCallItemResult,
    ReasoningItem,
    TodoItem,
    TodoListItem,
    WebSearchItem,
)
from codex_sdk.hooks import ThreadHooks
from codex_sdk.thread import ParsedTurn, Thread, Turn, TurnOptions, normalize_input


class FakeExecSequence:
    def __init__(self, runs: List[List[str]]):
        self._runs = runs
        self.calls: List[Any] = []

    async def run(self, args: Any):
        self.calls.append(args)
        run_index = min(len(self.calls) - 1, len(self._runs) - 1)
        for line in self._runs[run_index]:
            yield line


@pytest.mark.asyncio
async def test_thread_run_twice_passes_thread_id_on_second_call():
    runs = [
        [
            '{"type":"thread.started","thread_id":"thread-1"}',
            '{"type":"item.completed","item":{"id":"m1","type":"agent_message","text":"one"}}',
            '{"type":"turn.completed","usage":{"input_tokens":1,"cached_input_tokens":0,"output_tokens":1}}',
        ],
        [
            '{"type":"turn.started"}',
            '{"type":"item.completed","item":{"id":"m2","type":"agent_message","text":"two"}}',
            '{"type":"turn.completed","usage":{"input_tokens":1,"cached_input_tokens":0,"output_tokens":1}}',
        ],
    ]
    exec = FakeExecSequence(runs)
    thread = Thread(exec, CodexOptions(), ThreadOptions())

    await thread.run("first")
    await thread.run("second")

    assert exec.calls[0].thread_id is None
    assert exec.calls[1].thread_id == "thread-1"


@pytest.mark.asyncio
async def test_thread_run_sets_final_response_to_last_agent_message():
    exec = FakeExecSequence(
        [
            [
                '{"type":"thread.started","thread_id":"thread-1"}',
                '{"type":"item.completed","item":{"id":"m1","type":"agent_message","text":"one"}}',
                '{"type":"item.completed","item":{"id":"m2","type":"agent_message","text":"two"}}',
                '{"type":"turn.completed","usage":{"input_tokens":1,"cached_input_tokens":0,"output_tokens":1}}',
            ]
        ]
    )
    thread = Thread(exec, CodexOptions(), ThreadOptions())
    turn = await thread.run("hello")
    assert turn.final_response == "two"


@pytest.mark.asyncio
async def test_thread_run_raises_turn_failed_error_after_items():
    exec = FakeExecSequence(
        [
            [
                '{"type":"thread.started","thread_id":"thread-1"}',
                '{"type":"item.completed","item":{"id":"m1","type":"agent_message","text":"partial"}}',
                '{"type":"turn.failed","error":{"message":"boom"}}',
                '{"type":"item.completed","item":{"id":"m2","type":"agent_message","text":"ignored"}}',
            ]
        ]
    )
    thread = Thread(exec, CodexOptions(), ThreadOptions())
    with pytest.raises(TurnFailedError) as exc:
        await thread.run("hello")
    assert "boom" in str(exc.value)


def test_thread_run_sync_works_outside_event_loop():
    exec = FakeExecSequence(
        [
            [
                '{"type":"thread.started","thread_id":"thread-1"}',
                '{"type":"item.completed","item":{"id":"m1","type":"agent_message","text":"ok"}}',
                '{"type":"turn.completed","usage":{"input_tokens":1,"cached_input_tokens":0,"output_tokens":1}}',
            ]
        ]
    )
    thread = Thread(exec, CodexOptions(), ThreadOptions())
    result = thread.run_sync("hi")
    assert result.final_response == "ok"


@pytest.mark.asyncio
async def test_thread_run_sync_raises_inside_event_loop():
    thread = Thread(FakeExecSequence([[]]), CodexOptions(), ThreadOptions())
    with pytest.raises(CodexError):
        thread.run_sync("hi")


@pytest.mark.asyncio
async def test_thread_run_json_parses_json_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    exec = FakeExecSequence(
        [
            [
                '{"type":"thread.started","thread_id":"thread-1"}',
                '{"type":"item.completed","item":{"id":"m1","type":"agent_message","text":"{\\"answer\\":\\"ok\\"}"}}',
                '{"type":"turn.completed","usage":{"input_tokens":1,"cached_input_tokens":0,"output_tokens":1}}',
            ]
        ]
    )
    thread = Thread(exec, CodexOptions(), ThreadOptions())
    parsed = await thread.run_json("hi", output_schema={"type": "object"})
    assert parsed.output == {"answer": "ok"}


@pytest.mark.asyncio
async def test_thread_run_json_raises_on_invalid_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    exec = FakeExecSequence(
        [
            [
                '{"type":"thread.started","thread_id":"thread-1"}',
                '{"type":"item.completed","item":{"id":"m1","type":"agent_message","text":"not json"}}',
                '{"type":"turn.completed","usage":{"input_tokens":1,"cached_input_tokens":0,"output_tokens":1}}',
            ]
        ]
    )
    thread = Thread(exec, CodexOptions(), ThreadOptions())
    with pytest.raises(CodexParseError):
        await thread.run_json("hi", output_schema={"type": "object"})


@pytest.mark.asyncio
async def test_thread_run_json_sync_raises_inside_event_loop():
    thread = Thread(FakeExecSequence([[]]), CodexOptions(), ThreadOptions())
    with pytest.raises(CodexError):
        thread.run_json_sync("hi", output_schema={"type": "object"})


@pytest.mark.asyncio
async def test_thread_run_pydantic_validates_output():
    class Result(BaseModel):
        answer: str

    exec = FakeExecSequence(
        [
            [
                '{"type":"thread.started","thread_id":"thread-1"}',
                '{"type":"item.completed","item":{"id":"m1","type":"agent_message","text":"{\\"answer\\":\\"ok\\"}"}}',
                '{"type":"turn.completed","usage":{"input_tokens":1,"cached_input_tokens":0,"output_tokens":1}}',
            ]
        ]
    )
    thread = Thread(exec, CodexOptions(), ThreadOptions())
    parsed = await thread.run_pydantic("hi", output_model=Result)
    assert isinstance(parsed.output, Result)
    assert parsed.output.answer == "ok"


@pytest.mark.asyncio
async def test_thread_run_pydantic_requires_base_model():
    thread = Thread(FakeExecSequence([[]]), CodexOptions(), ThreadOptions())
    with pytest.raises(CodexError):
        await thread.run_pydantic("hi", output_model=dict)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_thread_run_pydantic_adds_additional_properties(
    monkeypatch: pytest.MonkeyPatch,
):
    class Result(BaseModel):
        answer: str

        @classmethod
        def model_json_schema(cls, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            schema = super().model_json_schema(*args, **kwargs)
            schema.pop("additionalProperties", None)
            return schema

    exec = FakeExecSequence([[]])
    thread = Thread(exec, CodexOptions(), ThreadOptions())
    captured: Dict[str, Any] = {}

    async def fake_run_json(
        input: Any,
        *,
        output_schema: Dict[str, Any],
        turn_options: Optional[TurnOptions] = None,
    ) -> ParsedTurn[Any]:
        captured["schema"] = output_schema
        turn = Turn(items=[], final_response="", usage=None)
        return ParsedTurn(turn=turn, output={"answer": "ok"})

    monkeypatch.setattr(thread, "run_json", fake_run_json)

    parsed = await thread.run_pydantic("hi", output_model=Result)
    assert isinstance(parsed.output, Result)
    assert captured["schema"]["additionalProperties"] is False


@pytest.mark.asyncio
async def test_thread_run_pydantic_sync_raises_inside_event_loop():
    class Result(BaseModel):
        answer: str

    thread = Thread(FakeExecSequence([[]]), CodexOptions(), ThreadOptions())
    with pytest.raises(CodexError):
        thread.run_pydantic_sync("hi", output_model=Result)


def test_parse_item_branches():
    thread = Thread(FakeExecSequence([[]]), CodexOptions(), ThreadOptions())

    assert thread._parse_item({"id": "m", "type": "agent_message", "text": "hi"}) == (
        AgentMessageItem(id="m", type="agent_message", text="hi")
    )

    assert thread._parse_item({"id": "r", "type": "reasoning", "text": "..."}) == (
        ReasoningItem(id="r", type="reasoning", text="...")
    )

    cmd = thread._parse_item(
        {
            "id": "c",
            "type": "command_execution",
            "command": "echo hi",
            "aggregated_output": "hi",
            "exit_code": 0,
            "status": "completed",
        }
    )
    assert cmd == CommandExecutionItem(
        id="c",
        type="command_execution",
        command="echo hi",
        aggregated_output="hi",
        exit_code=0,
        status="completed",
    )

    file_item = thread._parse_item(
        {
            "id": "p",
            "type": "file_change",
            "changes": [{"path": "a.txt", "kind": "update"}],
            "status": "completed",
        }
    )
    assert file_item == FileChangeItem(
        id="p",
        type="file_change",
        changes=[FileUpdateChange(path="a.txt", kind="update")],
        status="completed",
    )

    mcp_item = thread._parse_item(
        {
            "id": "mcp",
            "type": "mcp_tool_call",
            "server": "srv",
            "tool": "tool",
            "status": "failed",
            "arguments": {"a": 1},
            "result": {"content": ["x"], "structured_content": {"k": "v"}},
            "error": {"message": "nope"},
        }
    )
    assert mcp_item == McpToolCallItem(
        id="mcp",
        type="mcp_tool_call",
        server="srv",
        tool="tool",
        status="failed",
        arguments={"a": 1},
        result=McpToolCallItemResult(content=["x"], structured_content={"k": "v"}),
        error=McpToolCallItemError(message="nope"),
    )

    assert thread._parse_item({"id": "w", "type": "web_search", "query": "q"}) == (
        WebSearchItem(id="w", type="web_search", query="q")
    )

    todo = thread._parse_item(
        {
            "id": "t",
            "type": "todo_list",
            "items": [{"text": "one", "completed": False}],
        }
    )
    assert todo == TodoListItem(
        id="t", type="todo_list", items=[TodoItem(text="one", completed=False)]
    )

    assert thread._parse_item({"id": "e", "type": "error", "message": "bad"}) == (
        ErrorItem(id="e", type="error", message="bad")
    )

    with pytest.raises(CodexParseError):
        thread._parse_item({"id": "u", "type": "unknown"})


def test_parse_event_branches():
    thread = Thread(FakeExecSequence([[]]), CodexOptions(), ThreadOptions())

    started = thread._parse_event({"type": "thread.started", "thread_id": "t"})
    assert started.type == "thread.started"
    assert getattr(started, "thread_id") == "t"

    item_started = thread._parse_event(
        {
            "type": "item.started",
            "item": {"id": "m", "type": "agent_message", "text": "hi"},
        }
    )
    assert item_started.type == "item.started"
    assert getattr(item_started, "item").type == "agent_message"

    item_updated = thread._parse_event(
        {"type": "item.updated", "item": {"id": "m", "type": "reasoning", "text": "x"}}
    )
    assert item_updated.type == "item.updated"
    assert getattr(item_updated, "item").type == "reasoning"

    stream_error = thread._parse_event({"type": "error", "message": "boom"})
    assert stream_error.type == "error"
    assert getattr(stream_error, "message") == "boom"


@pytest.mark.asyncio
async def test_run_streamed_returns_wrapper():
    exec = FakeExecSequence(
        [
            [
                '{"type":"thread.started","thread_id":"thread-1"}',
                '{"type":"turn.started"}',
                '{"type":"turn.completed","usage":{"input_tokens":1,"cached_input_tokens":0,"output_tokens":1}}',
            ]
        ]
    )
    thread = Thread(exec, CodexOptions(), ThreadOptions())
    streamed = await thread.run_streamed("hi")
    events = []
    async for event in streamed.events:
        events.append(event.type)
    assert events == ["thread.started", "turn.started", "turn.completed"]


@pytest.mark.asyncio
async def test_run_streamed_raises_parse_error_and_cleans_schema(tmp_path: Path):
    exec = FakeExecSequence([["not json"]])
    thread = Thread(exec, CodexOptions(), ThreadOptions())

    schema_dir = tmp_path / "schema-dir"
    schema_dir.mkdir()
    schema_path = schema_dir / "schema.json"
    schema_path.write_text("{}", encoding="utf-8")
    cleanup_called = False

    async def fake_create_schema_file(_: Any):
        async def cleanup() -> None:
            nonlocal cleanup_called
            cleanup_called = True
            schema_path.unlink()
            schema_dir.rmdir()

        return str(schema_path), cleanup

    import codex_sdk.thread as thread_module

    thread_module_create_schema_file = thread_module.create_output_schema_file
    thread_module.create_output_schema_file = fake_create_schema_file  # type: ignore[assignment]
    try:
        with pytest.raises(CodexParseError):
            async for _ in thread.run_streamed_events(
                "hi", TurnOptions(output_schema={"type": "object"})
            ):
                pass
    finally:
        thread_module.create_output_schema_file = thread_module_create_schema_file  # type: ignore[assignment]

    assert cleanup_called is True


def test_normalize_input_string_only():
    from codex_sdk.thread import normalize_input

    prompt, images = normalize_input("hi")
    assert prompt == "hi"
    assert images == []


def test_normalize_input_concatenates_text_and_collects_images():
    from codex_sdk.thread import normalize_input

    prompt, images = normalize_input(
        [
            {"type": "text", "text": "a"},
            {"type": "local_image", "path": "/tmp/a.png"},
            {"type": "text", "text": "b"},
        ]
    )
    assert prompt == "a\n\nb"
    assert images == ["/tmp/a.png"]


@pytest.mark.asyncio
async def test_turn_filters_work():
    turn = Turn(
        items=[
            AgentMessageItem(id="a", type="agent_message", text="hi"),
            ReasoningItem(id="r", type="reasoning", text="..."),
            CommandExecutionItem(
                id="c",
                type="command_execution",
                command="echo",
                aggregated_output="",
                exit_code=0,
                status="completed",
            ),
            FileChangeItem(
                id="f",
                type="file_change",
                changes=[FileUpdateChange(path="a.txt", kind="add")],
                status="completed",
            ),
            McpToolCallItem(
                id="m", type="mcp_tool_call", server="s", tool="t", status="completed"
            ),
            WebSearchItem(id="w", type="web_search", query="q"),
            TodoListItem(
                id="t", type="todo_list", items=[TodoItem(text="x", completed=False)]
            ),
            ErrorItem(id="e", type="error", message="bad"),
        ],
        final_response="",
        usage=Usage(input_tokens=1, cached_input_tokens=0, output_tokens=1),
    )

    assert [m.id for m in turn.agent_messages()] == ["a"]
    assert [r.id for r in turn.reasoning()] == ["r"]
    assert [c.id for c in turn.commands()] == ["c"]
    assert [f.id for f in turn.file_changes()] == ["f"]
    assert [m.id for m in turn.mcp_tool_calls()] == ["m"]
    assert [w.id for w in turn.web_searches()] == ["w"]
    assert [t.id for t in turn.todo_lists()] == ["t"]
    assert [e.id for e in turn.errors()] == ["e"]


def test_run_json_sync_outside_event_loop():
    exec = FakeExecSequence(
        [
            [
                '{"type":"thread.started","thread_id":"thread-1"}',
                '{"type":"item.completed","item":{"id":"m1","type":"agent_message","text":"{\\"ok\\":true}"}}',
                '{"type":"turn.completed","usage":{"input_tokens":1,"cached_input_tokens":0,"output_tokens":1}}',
            ]
        ]
    )
    thread = Thread(exec, CodexOptions(), ThreadOptions())
    parsed = thread.run_json_sync("hi", output_schema={"type": "object"})
    assert parsed.output == {"ok": True}


def test_run_sync_with_options():
    exec = FakeExecSequence(
        [
            [
                '{"type":"thread.started","thread_id":"thread-1"}',
                '{"type":"item.completed","item":{"id":"m1","type":"agent_message","text":"ok"}}',
                '{"type":"turn.completed","usage":{"input_tokens":1,"cached_input_tokens":0,"output_tokens":1}}',
            ]
        ]
    )
    thread = Thread(exec, CodexOptions(), ThreadOptions())
    result = thread.run_sync("hi", TurnOptions())
    assert result.final_response == "ok"


def test_run_pydantic_sync_outside_event_loop():
    class Result(BaseModel):
        answer: str

    exec = FakeExecSequence(
        [
            [
                '{"type":"thread.started","thread_id":"thread-1"}',
                '{"type":"item.completed","item":{"id":"m1","type":"agent_message","text":"{\\"answer\\":\\"ok\\"}"}}',
                '{"type":"turn.completed","usage":{"input_tokens":1,"cached_input_tokens":0,"output_tokens":1}}',
            ]
        ]
    )
    thread = Thread(exec, CodexOptions(), ThreadOptions())
    parsed = thread.run_pydantic_sync("hi", output_model=Result)
    assert parsed.output.answer == "ok"


@pytest.mark.asyncio
async def test_run_with_hooks_invokes_callbacks():
    exec = FakeExecSequence(
        [
            [
                '{"type":"thread.started","thread_id":"thread-1"}',
                '{"type":"item.completed","item":{"id":"m1","type":"agent_message","text":"hello"}}',
                '{"type":"turn.completed","usage":{"input_tokens":1,"cached_input_tokens":0,"output_tokens":1}}',
            ]
        ]
    )
    thread = Thread(exec, CodexOptions(), ThreadOptions())
    seen = []
    hooks = ThreadHooks(on_event=lambda event: seen.append(event.type))
    turn = await thread.run_with_hooks("hi", hooks=hooks)
    assert turn.final_response == "hello"
    assert "turn.completed" in seen


@pytest.mark.asyncio
async def test_run_handles_non_agent_items():
    exec = FakeExecSequence(
        [
            [
                '{"type":"thread.started","thread_id":"thread-1"}',
                '{"type":"item.completed","item":{"id":"c1","type":"command_execution","command":"echo hi","aggregated_output":"","status":"completed"}}',
                '{"type":"item.completed","item":{"id":"m1","type":"agent_message","text":"final"}}',
                '{"type":"turn.completed","usage":{"input_tokens":1,"cached_input_tokens":0,"output_tokens":1}}',
            ]
        ]
    )
    thread = Thread(exec, CodexOptions(), ThreadOptions())
    turn = await thread.run("hi")
    assert turn.final_response == "final"


@pytest.mark.asyncio
async def test_run_with_hooks_raises_on_failure():
    exec = FakeExecSequence(
        [
            [
                '{"type":"thread.started","thread_id":"thread-1"}',
                '{"type":"turn.failed","error":{"message":"boom"}}',
            ]
        ]
    )
    thread = Thread(exec, CodexOptions(), ThreadOptions())
    hooks = ThreadHooks(on_event=lambda _event: None)
    with pytest.raises(TurnFailedError):
        await thread.run_with_hooks("hi", hooks=hooks)


@pytest.mark.asyncio
async def test_run_pydantic_preserves_additional_properties():
    class Result(BaseModel):
        answer: str

        @classmethod
        def model_json_schema(cls, *args, **kwargs):
            schema = super().model_json_schema(*args, **kwargs)
            schema["additionalProperties"] = True
            return schema

    exec = FakeExecSequence(
        [
            [
                '{"type":"thread.started","thread_id":"thread-1"}',
                '{"type":"item.completed","item":{"id":"m1","type":"agent_message","text":"{\\"answer\\":\\"ok\\"}"}}',
                '{"type":"turn.completed","usage":{"input_tokens":1,"cached_input_tokens":0,"output_tokens":1}}',
            ]
        ]
    )
    thread = Thread(exec, CodexOptions(), ThreadOptions())
    parsed = await thread.run_pydantic("hi", output_model=Result)
    assert parsed.output.answer == "ok"


@pytest.mark.asyncio
async def test_parse_item_mcp_tool_call_without_result():
    thread = Thread(FakeExecSequence([[]]), CodexOptions(), ThreadOptions())
    parsed = thread._parse_item(
        {
            "id": "mcp-1",
            "type": "mcp_tool_call",
            "server": "s",
            "tool": "t",
            "status": "completed",
        }
    )
    assert parsed.result is None
    assert parsed.error is None


@pytest.mark.asyncio
async def test_run_pydantic_requires_base_model(monkeypatch: pytest.MonkeyPatch):
    exec = FakeExecSequence([[]])
    thread = Thread(exec, CodexOptions(), ThreadOptions())

    class Dummy:
        pass

    class FakeModule:
        BaseModel = None

    import importlib

    monkeypatch.setattr(importlib, "import_module", lambda _name: FakeModule())
    with pytest.raises(CodexError):
        await thread.run_pydantic("hi", output_model=Dummy)


@pytest.mark.asyncio
async def test_run_streamed_with_options():
    exec = FakeExecSequence(
        [
            [
                '{"type":"turn.started"}',
                '{"type":"turn.completed","usage":{"input_tokens":1,"cached_input_tokens":0,"output_tokens":1}}',
            ]
        ]
    )
    thread = Thread(exec, CodexOptions(), ThreadOptions())
    streamed = await thread.run_streamed("hi", TurnOptions())
    events = []
    async for event in streamed.events:
        events.append(event.type)
    assert events == ["turn.started", "turn.completed"]


@pytest.mark.asyncio
async def test_cleanup_non_awaitable(monkeypatch: pytest.MonkeyPatch):
    exec = FakeExecSequence(
        [
            [
                '{"type":"turn.completed","usage":{"input_tokens":1,"cached_input_tokens":0,"output_tokens":1}}',
            ]
        ]
    )
    thread = Thread(exec, CodexOptions(), ThreadOptions())

    async def fake_create_schema_file(_: Any):
        def cleanup() -> None:
            return None

        return None, cleanup

    import codex_sdk.thread as thread_module

    monkeypatch.setattr(thread_module, "create_output_schema_file", fake_create_schema_file)
    async for _ in thread.run_streamed_events("hi"):
        pass


def test_normalize_input_collects_images():
    prompt, images = normalize_input(
        [
            {"type": "text", "text": "hello"},
            {"type": "local_image", "path": "/tmp/a.png"},
        ]
    )
    assert prompt == "hello"
    assert images == ["/tmp/a.png"]
