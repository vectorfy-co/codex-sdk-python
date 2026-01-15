import asyncio
import importlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from codex_sdk.events import Usage
from codex_sdk.integrations.pydantic_ai import (
    CodexHandoff,
    _item_summary,
    _jsonable,
    codex_handoff_tool,
)
from codex_sdk.items import (
    AgentMessageItem,
    CommandExecutionItem,
    ErrorItem,
    FileChangeItem,
    FileUpdateChange,
    McpToolCallItem,
    ReasoningItem,
    TodoItem,
    TodoListItem,
    WebSearchItem,
)
from codex_sdk.thread import Turn

pytest.importorskip("pydantic_ai")
tools = importlib.import_module("pydantic_ai.tools")
Tool = tools.Tool


@dataclass
class Demo:
    value: str


class FakeThread:
    def __init__(self, thread_id: str, items: List[Any]):
        self.id = thread_id
        self._items = items
        self.prompts: List[str] = []

    async def run(self, prompt: str) -> Turn:
        self.prompts.append(prompt)
        return Turn(
            items=list(self._items),
            final_response="final",
            usage=Usage(input_tokens=1, cached_input_tokens=2, output_tokens=3),
        )


class FakeCodex:
    def __init__(self, items: List[Any]):
        self._items = items
        self.started: List[Optional[Any]] = []
        self._count = 0

    def start_thread(self, options: Optional[Any] = None) -> FakeThread:
        self.started.append(options)
        self._count += 1
        return FakeThread(f"thread-{self._count}", items=self._items)


def test_jsonable_converts_dataclasses_and_collections():
    value = {"x": Demo("ok"), "y": [Demo("a"), Demo("b")]}
    assert _jsonable(value) == {
        "x": {"value": "ok"},
        "y": [{"value": "a"}, {"value": "b"}],
    }


def test_item_summary_known_types():
    assert _item_summary(AgentMessageItem(id="m", type="agent_message", text="hi")) == {
        "type": "agent_message",
        "text": "hi",
    }
    assert _item_summary(ReasoningItem(id="r", type="reasoning", text="...")) == {
        "type": "reasoning",
        "text": "...",
    }
    assert _item_summary(
        CommandExecutionItem(
            id="c",
            type="command_execution",
            command="echo",
            aggregated_output="",
            exit_code=0,
            status="completed",
        )
    ) == {
        "type": "command_execution",
        "command": "echo",
        "status": "completed",
        "exit_code": 0,
    }
    assert _item_summary(
        FileChangeItem(
            id="f",
            type="file_change",
            changes=[FileUpdateChange(path="a.txt", kind="update")],
            status="completed",
        )
    ) == {
        "type": "file_change",
        "status": "completed",
        "changes": [{"path": "a.txt", "kind": "update"}],
    }
    assert _item_summary(
        McpToolCallItem(
            id="mcp", type="mcp_tool_call", server="s", tool="t", status="completed"
        )
    ) == {
        "type": "mcp_tool_call",
        "server": "s",
        "tool": "t",
        "status": "completed",
    }
    assert _item_summary(WebSearchItem(id="w", type="web_search", query="q")) == {
        "type": "web_search",
        "query": "q",
    }
    assert _item_summary(
        TodoListItem(
            id="t", type="todo_list", items=[TodoItem(text="x", completed=False)]
        )
    ) == {"type": "todo_list", "items": [{"text": "x", "completed": False}]}
    assert _item_summary(ErrorItem(id="e", type="error", message="bad")) == {
        "type": "error",
        "message": "bad",
    }


@pytest.mark.asyncio
async def test_codex_handoff_persists_thread_by_default():
    items = [AgentMessageItem(id="m", type="agent_message", text="hi")]
    codex = FakeCodex(items)
    handoff = CodexHandoff(codex=codex)

    first = await handoff.run("one")
    second = await handoff.run("two")

    assert first["thread_id"] == "thread-1"
    assert second["thread_id"] == "thread-1"
    assert codex._count == 1


@pytest.mark.asyncio
async def test_codex_handoff_does_not_persist_thread_when_disabled():
    items = [AgentMessageItem(id="m", type="agent_message", text="hi")]
    codex = FakeCodex(items)
    handoff = CodexHandoff(codex=codex, persist_thread=False)

    first = await handoff.run("one")
    second = await handoff.run("two")

    assert first["thread_id"] == "thread-1"
    assert second["thread_id"] == "thread-2"
    assert codex._count == 2


@pytest.mark.asyncio
async def test_codex_handoff_includes_items_and_usage():
    items = [
        AgentMessageItem(id="m1", type="agent_message", text="one"),
        AgentMessageItem(id="m2", type="agent_message", text="two"),
    ]
    codex = FakeCodex(items)
    handoff = CodexHandoff(
        codex=codex,
        include_items=True,
        items_limit=1,
        include_usage=True,
    )

    result = await handoff.run("hi")
    assert result["final_response"] == "final"
    assert result["usage"] == {
        "input_tokens": 1,
        "cached_input_tokens": 2,
        "output_tokens": 3,
    }
    assert result["items"] == [{"type": "agent_message", "text": "two"}]


@pytest.mark.asyncio
async def test_codex_handoff_timeout_uses_asyncio_wait_for(
    monkeypatch: pytest.MonkeyPatch,
):
    items = [AgentMessageItem(id="m", type="agent_message", text="hi")]
    codex = FakeCodex(items)
    handoff = CodexHandoff(codex=codex, timeout_seconds=0.01)

    captured: Dict[str, Any] = {}
    original_wait_for = asyncio.wait_for

    async def fake_wait_for(awaitable, *, timeout):
        captured["timeout"] = timeout
        return await original_wait_for(awaitable, timeout=timeout)

    import codex_sdk.integrations.pydantic_ai as integration_module

    monkeypatch.setattr(integration_module.asyncio, "wait_for", fake_wait_for)
    await handoff.run("hi")
    assert captured["timeout"] == 0.01


def test_codex_handoff_tool_factory_builds_tool():
    items = [AgentMessageItem(id="m", type="agent_message", text="hi")]
    codex = FakeCodex(items)
    tool = codex_handoff_tool(codex=codex, name="codex_x", description="desc")

    assert isinstance(tool, Tool)
    assert tool.name == "codex_x"
    assert tool.description == "desc"
