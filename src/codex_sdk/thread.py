"""Thread class for managing conversations with the Codex agent."""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    Generic,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    TypedDict,
    TypeVar,
    Union,
)

from .config_overrides import merge_config_overrides
from .events import ThreadError, ThreadEvent, Usage
from .exceptions import CodexError, CodexParseError, TurnFailedError
from .exec import CodexExec, CodexExecArgs, create_output_schema_file
from .hooks import ThreadHooks, dispatch_event
from .items import (
    AgentMessageItem,
    CommandExecutionItem,
    ErrorItem,
    FileChangeItem,
    McpToolCallItem,
    ReasoningItem,
    ThreadItem,
    TodoListItem,
    WebSearchItem,
)
from .options import CodexOptions, ThreadOptions, TurnOptions
from .telemetry import span

T = TypeVar("T")


@dataclass
class Turn:
    """Completed turn."""

    items: List[ThreadItem]
    final_response: str
    usage: Optional[Usage]

    def agent_messages(self) -> List[AgentMessageItem]:
        return [item for item in self.items if item.type == "agent_message"]

    def reasoning(self) -> List[ReasoningItem]:
        return [item for item in self.items if item.type == "reasoning"]

    def commands(self) -> List[CommandExecutionItem]:
        return [item for item in self.items if item.type == "command_execution"]

    def file_changes(self) -> List[FileChangeItem]:
        return [item for item in self.items if item.type == "file_change"]

    def mcp_tool_calls(self) -> List[McpToolCallItem]:
        return [item for item in self.items if item.type == "mcp_tool_call"]

    def web_searches(self) -> List[WebSearchItem]:
        return [item for item in self.items if item.type == "web_search"]

    def todo_lists(self) -> List[TodoListItem]:
        return [item for item in self.items if item.type == "todo_list"]

    def errors(self) -> List[ErrorItem]:
        return [item for item in self.items if item.type == "error"]


# Alias for Turn to describe the result of run()
RunResult = Turn


@dataclass
class StreamedTurn:
    """The result of the run_streamed method."""

    events: AsyncGenerator[ThreadEvent, None]


# Alias for StreamedTurn to describe the result of run_streamed()
RunStreamedResult = StreamedTurn


@dataclass
class ParsedTurn(Generic[T]):
    """A completed turn plus parsed output."""

    turn: Turn
    output: T


class TextInput(TypedDict):
    type: Literal["text"]
    text: str


class LocalImageInput(TypedDict):
    type: Literal["local_image"]
    path: str


UserInput = Union[TextInput, LocalImageInput]

# Input alias to mirror the TypeScript SDK
Input = Union[str, Sequence[UserInput]]


class Thread:
    """Represents a thread of conversation with the agent.

    One thread can have multiple consecutive turns.
    """

    def __init__(
        self,
        exec: CodexExec,
        options: CodexOptions,
        thread_options: ThreadOptions,
        thread_id: Optional[str] = None,
    ):
        self._exec = exec
        self._options = options
        self._id = thread_id
        self._thread_options = thread_options

    @property
    def id(self) -> Optional[str]:
        """Return the ID of the thread. Populated after the first turn starts."""
        return self._id

    def run_sync(
        self, input: Input, turn_options: Optional[TurnOptions] = None
    ) -> Turn:
        """
        Synchronous wrapper around `run()`.

        Raises:
            CodexError: If called from within a running event loop.
        """
        if turn_options is None:
            turn_options = TurnOptions()

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(input, turn_options))

        raise CodexError(
            "run_sync() cannot be used from a running event loop; use await run()."
        )

    def run_json_sync(
        self,
        input: Input,
        *,
        output_schema: Mapping[str, Any],
        turn_options: Optional[TurnOptions] = None,
    ) -> ParsedTurn[Any]:
        """Synchronous wrapper around `run_json()`."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.run_json(
                    input, output_schema=output_schema, turn_options=turn_options
                )
            )
        raise CodexError(
            "run_json_sync() cannot be used from a running event loop; use await run_json()."
        )

    def run_pydantic_sync(
        self,
        input: Input,
        *,
        output_model: Any,
        turn_options: Optional[TurnOptions] = None,
    ) -> ParsedTurn[Any]:
        """Synchronous wrapper around `run_pydantic()`."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.run_pydantic(
                    input, output_model=output_model, turn_options=turn_options
                )
            )
        raise CodexError(
            "run_pydantic_sync() cannot be used from a running event loop; use await run_pydantic()."
        )

    async def run_streamed(
        self, input: Input, turn_options: Optional[TurnOptions] = None
    ) -> RunStreamedResult:
        """
        Provide the input to the agent and stream events as they are produced.

        Args:
            input: Input prompt to send to the agent.
            turn_options: Optional turn configuration.

        Returns:
            StreamedTurn containing an async generator of thread events.

        Raises:
            CodexParseError: If a streamed event cannot be parsed.
            CodexError: Propagated errors from the Codex CLI invocation.
        """
        if turn_options is None:
            turn_options = TurnOptions()

        return StreamedTurn(events=self._run_streamed_internal(input, turn_options))

    async def run_streamed_events(
        self, input: Input, turn_options: Optional[TurnOptions] = None
    ) -> AsyncGenerator[ThreadEvent, None]:
        """
        Provide the input to the agent and yield events directly.

        This helper enables a concise `async for event in thread.run_streamed_events(...)`
        pattern without unpacking the StreamedTurn wrapper.

        Args:
            input: Input prompt to send to the agent.
            turn_options: Optional turn configuration.

        Yields:
            Parsed ThreadEvent objects as they arrive.

        Raises:
            CodexParseError: If a streamed event cannot be parsed.
            CodexError: Propagated errors from the Codex CLI invocation.
        """
        if turn_options is None:
            turn_options = TurnOptions()

        async for event in self._run_streamed_internal(input, turn_options):
            yield event

    async def _run_streamed_internal(
        self, input: Input, turn_options: TurnOptions
    ) -> AsyncGenerator[ThreadEvent, None]:
        """Internal method for streaming events."""
        prompt, images = normalize_input(input)
        schema_path, cleanup = await create_output_schema_file(
            turn_options.output_schema
        )

        try:
            with span(
                "codex_sdk.thread.turn",
                thread_id=self._id,
                model=self._thread_options.model,
                sandbox_mode=self._thread_options.sandbox_mode,
                working_directory=self._thread_options.working_directory,
            ):
                args = CodexExecArgs(
                    input=prompt,
                    base_url=self._options.base_url,
                    api_key=self._options.api_key,
                    thread_id=self._id,
                    images=images,
                    model=self._thread_options.model,
                    sandbox_mode=self._thread_options.sandbox_mode,
                    working_directory=self._thread_options.working_directory,
                    additional_directories=self._thread_options.additional_directories,
                    skip_git_repo_check=self._thread_options.skip_git_repo_check,
                    output_schema_file=schema_path,
                    model_reasoning_effort=self._thread_options.model_reasoning_effort,
                    network_access_enabled=self._thread_options.network_access_enabled,
                    web_search_enabled=self._thread_options.web_search_enabled,
                    web_search_cached_enabled=self._thread_options.web_search_cached_enabled,
                    skills_enabled=self._thread_options.skills_enabled,
                    shell_snapshot_enabled=self._thread_options.shell_snapshot_enabled,
                    background_terminals_enabled=self._thread_options.background_terminals_enabled,
                    apply_patch_freeform_enabled=self._thread_options.apply_patch_freeform_enabled,
                    exec_policy_enabled=self._thread_options.exec_policy_enabled,
                    remote_models_enabled=self._thread_options.remote_models_enabled,
                    request_compression_enabled=self._thread_options.request_compression_enabled,
                    feature_overrides=self._thread_options.feature_overrides,
                    approval_policy=self._thread_options.approval_policy,
                    config_overrides=merge_config_overrides(
                        self._options.config_overrides,
                        self._thread_options.config_overrides,
                    ),
                    signal=turn_options.signal,
                )

                async for line in self._exec.run(args):
                    try:
                        parsed = json.loads(line)
                        event = self._parse_event(parsed)
                        if event.type == "thread.started":
                            self._id = event.thread_id
                        yield event
                    except json.JSONDecodeError as e:
                        raise CodexParseError(f"Failed to parse item: {line}") from e
        finally:
            cleanup_result = cleanup()
            if inspect.isawaitable(cleanup_result):
                await cleanup_result

    def _parse_event(self, data: dict) -> ThreadEvent:
        """Parse a JSON event into the appropriate ThreadEvent type."""
        from .events import (
            ItemCompletedEvent,
            ItemStartedEvent,
            ItemUpdatedEvent,
            ThreadError,
            ThreadErrorEvent,
            ThreadStartedEvent,
            TurnCompletedEvent,
            TurnFailedEvent,
            TurnStartedEvent,
            Usage,
        )

        event_type = data.get("type")

        if event_type == "thread.started":
            return ThreadStartedEvent(
                type="thread.started", thread_id=data["thread_id"]
            )
        elif event_type == "turn.started":
            return TurnStartedEvent(type="turn.started")
        elif event_type == "turn.completed":
            usage_data = data["usage"]
            usage = Usage(
                input_tokens=usage_data["input_tokens"],
                cached_input_tokens=usage_data["cached_input_tokens"],
                output_tokens=usage_data["output_tokens"],
            )
            return TurnCompletedEvent(type="turn.completed", usage=usage)
        elif event_type == "turn.failed":
            error_data = data["error"]
            error = ThreadError(message=error_data["message"])
            return TurnFailedEvent(type="turn.failed", error=error)
        elif event_type == "item.started":
            return ItemStartedEvent(
                type="item.started", item=self._parse_item(data["item"])
            )
        elif event_type == "item.updated":
            return ItemUpdatedEvent(
                type="item.updated", item=self._parse_item(data["item"])
            )
        elif event_type == "item.completed":
            return ItemCompletedEvent(
                type="item.completed", item=self._parse_item(data["item"])
            )
        elif event_type == "error":
            return ThreadErrorEvent(type="error", message=data["message"])
        else:
            raise CodexParseError(f"Unknown event type: {event_type}")

    def _parse_item(self, data: dict) -> ThreadItem:
        """Parse a JSON item into the appropriate ThreadItem type."""
        from .items import (
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

        item_type = data.get("type")

        if item_type == "agent_message":
            return AgentMessageItem(
                id=data["id"], type="agent_message", text=data["text"]
            )
        elif item_type == "reasoning":
            return ReasoningItem(id=data["id"], type="reasoning", text=data["text"])
        elif item_type == "command_execution":
            return CommandExecutionItem(
                id=data["id"],
                type="command_execution",
                command=data["command"],
                aggregated_output=data["aggregated_output"],
                exit_code=data.get("exit_code"),
                status=data["status"],
            )
        elif item_type == "file_change":
            changes = [
                FileUpdateChange(path=change["path"], kind=change["kind"])
                for change in data["changes"]
            ]
            return FileChangeItem(
                id=data["id"],
                type="file_change",
                changes=changes,
                status=data["status"],
            )
        elif item_type == "mcp_tool_call":
            result_data = data.get("result")
            result = None
            if isinstance(result_data, dict):
                content = result_data.get("content", [])
                structured_content = result_data.get("structured_content")
                result = McpToolCallItemResult(
                    content=list(content) if isinstance(content, list) else [],
                    structured_content=structured_content,
                )

            error_data = data.get("error")
            error = None
            if isinstance(error_data, dict) and "message" in error_data:
                error = McpToolCallItemError(message=str(error_data["message"]))

            return McpToolCallItem(
                id=data["id"],
                type="mcp_tool_call",
                server=data["server"],
                tool=data["tool"],
                status=data["status"],
                arguments=data.get("arguments"),
                result=result,
                error=error,
            )
        elif item_type == "web_search":
            return WebSearchItem(id=data["id"], type="web_search", query=data["query"])
        elif item_type == "todo_list":
            items = [
                TodoItem(text=item["text"], completed=item["completed"])
                for item in data["items"]
            ]
            return TodoListItem(id=data["id"], type="todo_list", items=items)
        elif item_type == "error":
            return ErrorItem(id=data["id"], type="error", message=data["message"])
        else:
            raise CodexParseError(f"Unknown item type: {item_type}")

    async def run(
        self, input: Input, turn_options: Optional[TurnOptions] = None
    ) -> Turn:
        """
        Provide the input to the agent and return the completed turn.

        Args:
            input: Input prompt to send to the agent.
            turn_options: Optional turn configuration.

        Returns:
            The completed turn containing items, the final agent message, and usage data.

        Raises:
            TurnFailedError: If the turn ends with a failure event.
            CodexParseError: If stream output cannot be parsed.
            CodexError: Propagated errors from the Codex CLI invocation.
        """
        if turn_options is None:
            turn_options = TurnOptions()

        items: List[ThreadItem] = []
        final_response: str = ""
        usage: Optional[Usage] = None
        turn_failure: Optional[ThreadError] = None

        async for event in self._run_streamed_internal(input, turn_options):
            if event.type == "item.completed":
                if event.item.type == "agent_message":
                    final_response = event.item.text
                items.append(event.item)
            elif event.type == "turn.completed":
                usage = event.usage
            elif event.type == "turn.failed":
                turn_failure = event.error
                break

        if turn_failure:
            raise TurnFailedError(turn_failure.message, error=turn_failure)

        return Turn(items=items, final_response=final_response, usage=usage)

    async def run_with_hooks(
        self,
        input: Input,
        *,
        hooks: ThreadHooks,
        turn_options: Optional[TurnOptions] = None,
    ) -> Turn:
        """
        Run a turn while dispatching streamed events to hooks.

        Args:
            input: Input prompt to send to the agent.
            hooks: Hook callbacks invoked for streamed events.
            turn_options: Optional turn configuration.

        Returns:
            The completed turn containing items, the final agent message, and usage data.
        """
        if turn_options is None:
            turn_options = TurnOptions()

        items: List[ThreadItem] = []
        final_response: str = ""
        usage: Optional[Usage] = None
        turn_failure: Optional[ThreadError] = None

        async for event in self._run_streamed_internal(input, turn_options):
            await dispatch_event(hooks, event)
            if event.type == "item.completed":
                if event.item.type == "agent_message":
                    final_response = event.item.text
                items.append(event.item)
            elif event.type == "turn.completed":
                usage = event.usage
            elif event.type == "turn.failed":
                turn_failure = event.error
                break

        if turn_failure:
            raise TurnFailedError(turn_failure.message, error=turn_failure)

        return Turn(items=items, final_response=final_response, usage=usage)

    async def run_json(
        self,
        input: Input,
        *,
        output_schema: Mapping[str, Any],
        turn_options: Optional[TurnOptions] = None,
    ) -> ParsedTurn[Any]:
        """
        Run a turn with a JSON schema and parse the final response as JSON.
        """
        signal = turn_options.signal if turn_options is not None else None
        turn = await self.run(
            input, TurnOptions(output_schema=output_schema, signal=signal)
        )
        try:
            parsed = json.loads(turn.final_response)
        except json.JSONDecodeError as exc:
            raise CodexParseError(
                f"Failed to parse JSON output: {turn.final_response}"
            ) from exc
        return ParsedTurn(turn=turn, output=parsed)

    async def run_pydantic(
        self,
        input: Input,
        *,
        output_model: Any,
        turn_options: Optional[TurnOptions] = None,
    ) -> ParsedTurn[Any]:
        """
        Run a turn with an output schema derived from a Pydantic model and validate the result.
        """
        try:
            import importlib

            pydantic = importlib.import_module("pydantic")
            BaseModel = getattr(pydantic, "BaseModel", None)
            if BaseModel is None:
                raise ImportError("pydantic.BaseModel not found")
        except ImportError as exc:  # pragma: no cover
            raise CodexError(
                'Pydantic is required for run_pydantic(); install with: uv add "codex-sdk-python[pydantic]"'
            ) from exc

        if not isinstance(output_model, type) or not issubclass(
            output_model, BaseModel
        ):
            raise CodexError("output_model must be a Pydantic BaseModel subclass")

        model_cls: Any = output_model
        schema = model_cls.model_json_schema()
        if isinstance(schema, dict) and "additionalProperties" not in schema:
            schema["additionalProperties"] = False

        parsed_turn = await self.run_json(
            input, output_schema=schema, turn_options=turn_options
        )
        validated = model_cls.model_validate(parsed_turn.output)
        return ParsedTurn(turn=parsed_turn.turn, output=validated)


def normalize_input(input: Input) -> tuple[str, List[str]]:
    if isinstance(input, str):
        return input, []

    prompt_parts: List[str] = []
    images: List[str] = []
    for item in input:
        if item["type"] == "text":
            prompt_parts.append(item["text"])
        elif item["type"] == "local_image":
            images.append(item["path"])

    return "\n\n".join(prompt_parts), images
