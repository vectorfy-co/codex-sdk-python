"""PydanticAI integration helpers."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Optional

from ..codex import Codex
from ..options import CodexOptions, ThreadOptions
from ..telemetry import span
from ..thread import Thread


def _require_pydantic_ai() -> Any:
    try:
        import pydantic_ai
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pydantic-ai is required for codex_sdk.integrations.pydantic_ai; "
            'install with: uv add "codex-sdk-python[pydantic-ai]"'
        ) from exc
    return pydantic_ai


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _item_summary(item: Any) -> Dict[str, Any]:
    item_type = getattr(item, "type", None)
    if item_type == "agent_message":
        return {"type": "agent_message", "text": getattr(item, "text", "")}
    if item_type == "reasoning":
        return {"type": "reasoning", "text": getattr(item, "text", "")}
    if item_type == "command_execution":
        return {
            "type": "command_execution",
            "command": getattr(item, "command", ""),
            "status": getattr(item, "status", ""),
            "exit_code": getattr(item, "exit_code", None),
        }
    if item_type == "file_change":
        return {
            "type": "file_change",
            "status": getattr(item, "status", ""),
            "changes": _jsonable(getattr(item, "changes", [])),
        }
    if item_type == "mcp_tool_call":
        return {
            "type": "mcp_tool_call",
            "server": getattr(item, "server", ""),
            "tool": getattr(item, "tool", ""),
            "status": getattr(item, "status", ""),
        }
    if item_type == "web_search":
        return {"type": "web_search", "query": getattr(item, "query", "")}
    if item_type == "todo_list":
        return {"type": "todo_list", "items": _jsonable(getattr(item, "items", []))}
    if item_type == "error":
        return {"type": "error", "message": getattr(item, "message", "")}
    return {"type": str(item_type), "raw": _jsonable(item)}


@dataclass
class CodexHandoff:
    """
    Helper that exposes Codex as a PydanticAI tool.

    The tool maintains its own Codex thread (conversation) by default, so repeated tool calls
    continue the same Codex session unless `persist_thread=False`.
    """

    codex: Codex
    thread_options: ThreadOptions = field(default_factory=ThreadOptions)
    persist_thread: bool = True

    include_items: bool = False
    items_limit: int = 50
    include_usage: bool = True

    timeout_seconds: Optional[float] = None

    _thread: Optional[Thread] = None

    def _get_thread(self) -> Thread:
        if self._thread is None or not self.persist_thread:
            self._thread = self.codex.start_thread(self.thread_options)
        return self._thread

    async def run(self, prompt: str) -> Dict[str, Any]:
        thread = self._get_thread()
        with span(
            "codex_sdk.pydantic_ai.handoff",
            thread_id=thread.id,
            model=self.thread_options.model,
            sandbox_mode=self.thread_options.sandbox_mode,
        ):
            if self.timeout_seconds is None:
                turn = await thread.run(prompt)
            else:
                turn = await asyncio.wait_for(
                    thread.run(prompt), timeout=self.timeout_seconds
                )

        response: Dict[str, Any] = {
            "thread_id": thread.id,
            "final_response": turn.final_response,
        }
        if self.include_usage:
            response["usage"] = _jsonable(turn.usage)
        if self.include_items:
            response["items"] = [
                _item_summary(item) for item in turn.items[-self.items_limit :]
            ]
        return response

    def tool(
        self,
        *,
        name: str = "codex_handoff",
        description: Optional[str] = None,
    ) -> Any:
        """
        Return a `pydantic_ai.Tool` instance ready to be passed to `Agent(..., tools=[...])`.
        """
        pydantic_ai = _require_pydantic_ai()
        Tool = getattr(pydantic_ai, "Tool")
        return Tool(self.run, name=name, description=description, takes_ctx=False)


def codex_handoff_tool(
    *,
    codex: Optional[Codex] = None,
    codex_options: Optional[CodexOptions] = None,
    thread_options: Optional[ThreadOptions] = None,
    name: str = "codex_handoff",
    description: Optional[str] = None,
    persist_thread: bool = True,
    include_items: bool = False,
    items_limit: int = 50,
    include_usage: bool = True,
    timeout_seconds: Optional[float] = None,
) -> Any:
    """
    Create a PydanticAI Tool that delegates work to Codex.

    Example:
        tool = codex_handoff_tool(thread_options=ThreadOptions(sandbox_mode='workspace-write'))
        agent = Agent('openai:gpt-5', tools=[tool])
    """
    if codex is None:
        codex = Codex(codex_options)
    if thread_options is None:
        thread_options = ThreadOptions()

    handoff = CodexHandoff(
        codex=codex,
        thread_options=thread_options,
        persist_thread=persist_thread,
        include_items=include_items,
        items_limit=items_limit,
        include_usage=include_usage,
        timeout_seconds=timeout_seconds,
    )
    return handoff.tool(name=name, description=description)
