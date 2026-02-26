"""PydanticAI model-provider integration backed by Codex app-server.

This module provides a `pydantic_ai.models.Model` implementation that delegates
completion + tool-call planning to Codex via a persistent app-server session.

The goal is to let PydanticAI own the tool loop (tool execution, retries, output
validation), while Codex behaves like a backend model that emits either:

- tool calls (to be executed by PydanticAI), or
- a final text response (when text output is allowed).
"""

from __future__ import annotations

import asyncio
import json
import logging
from base64 import b64encode
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional, Sequence

from ..app_server import AppServerClient, AppServerNotification, AppServerOptions
from ..exceptions import CodexError, TurnFailedError
from ..options import CodexOptions, ThreadOptions
from ..telemetry import span

try:
    from pydantic_ai.messages import (
        ModelMessage,
        ModelRequest,
        ModelResponse,
        ModelResponseStreamEvent,
        TextPart,
        ToolCallPart,
    )
    from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
    from pydantic_ai.profiles import ModelProfile, ModelProfileSpec
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.tools import ToolDefinition
    from pydantic_ai.usage import RequestUsage
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pydantic-ai is required for codex_sdk.integrations.pydantic_ai_model; "
        'install with: uv add "codex-sdk-python[pydantic-ai]"'
    ) from exc


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ToolCallEnvelope:
    """Parsed tool-call envelope returned by Codex turns."""

    tool_call_id: str
    tool_name: str
    arguments_json: str


@dataclass(frozen=True)
class _TextDeltaInstruction:
    """Instruction for streamed text-delta emission."""

    vendor_part_id: Any
    content: str


@dataclass(frozen=True)
class _ToolCallInstruction:
    """Instruction for streamed tool-call emission."""

    vendor_part_id: Any
    tool_name: str
    args: Any
    tool_call_id: Optional[str]


@dataclass
class _TurnAccumulationState:
    """State accumulated while consuming app-server notifications for a turn."""

    latest_agent_text: str = ""
    item_text_by_id: Dict[str, str] = field(default_factory=dict)
    last_updated_item_id: Optional[str] = None
    vendor_part_ids: Dict[str, int] = field(default_factory=dict)
    next_vendor_part_id: int = 0
    usage: Optional[RequestUsage] = None
    turn_failed_message: Optional[str] = None


def _jsonable(value: Any) -> Any:
    """Convert values into JSON-serializable structures for prompt/debug output."""
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "model_dump") and callable(getattr(value, "model_dump")):
        return value.model_dump(mode="json")
    if isinstance(value, bytes):
        return {"type": "bytes", "base64": b64encode(value).decode("ascii")}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _json_dumps(value: Any) -> str:
    """Dump a value to a deterministic JSON string for prompt embedding."""
    try:
        return json.dumps(
            _jsonable(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
    except TypeError:
        return str(value)


def _build_envelope_schema(tool_names: Sequence[str]) -> Dict[str, Any]:
    """Build the JSON schema used to constrain Codex output to tool calls + final text."""
    name_schema: Dict[str, Any] = {"type": "string"}
    if tool_names:
        name_schema = {"type": "string", "enum": list(tool_names)}

    return {
        "type": "object",
        "properties": {
            "tool_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": name_schema,
                        "arguments": {"type": "string"},
                    },
                    "required": ["id", "name", "arguments"],
                    "additionalProperties": False,
                },
            },
            "final": {"type": "string"},
        },
        "required": ["tool_calls", "final"],
        "additionalProperties": False,
    }


def _render_tool_definitions(
    *,
    function_tools: Sequence[ToolDefinition],
    output_tools: Sequence[ToolDefinition],
) -> str:
    """Render tool definitions into prompt text."""
    lines: List[str] = []
    lines.extend(_render_tool_section("Function tools:", function_tools))
    if output_tools:
        if lines:
            lines.append("")
        lines.extend(
            _render_tool_section(
                "Output tools (use ONE of these to finish when text is not allowed):",
                output_tools,
            )
        )
    return "\n".join(lines).strip()


def _render_tool_section(title: str, tools: Sequence[ToolDefinition]) -> List[str]:
    """Render one tool section to prompt lines."""
    if not tools:
        return []

    lines: List[str] = [title]
    for tool in tools:
        lines.append(f"- {tool.name}")
        if tool.description:
            lines.append(f"  description: {tool.description}")
        lines.append(f"  kind: {tool.kind}")
        lines.append(
            f"  parameters_json_schema: {_json_dumps(tool.parameters_json_schema)}"
        )
        if tool.outer_typed_dict_key:
            lines.append(f"  outer_typed_dict_key: {tool.outer_typed_dict_key}")
        if tool.strict is not None:
            lines.append(f"  strict: {str(tool.strict).lower()}")
        if getattr(tool, "sequential", False):
            lines.append("  sequential: true")
        metadata = getattr(tool, "metadata", None)
        if metadata is not None:
            lines.append(f"  metadata: {_json_dumps(metadata)}")
        timeout = getattr(tool, "timeout", None)
        if timeout is not None:
            lines.append(f"  timeout: {timeout}")
    return lines


def _tool_calls_from_envelope(output: Any) -> List[_ToolCallEnvelope]:
    """Extract tool call envelopes from an envelope object."""
    if not isinstance(output, dict):
        return []

    raw_calls = output.get("tool_calls")
    if not isinstance(raw_calls, list):
        return []

    calls: List[_ToolCallEnvelope] = []
    for call in raw_calls:
        if not isinstance(call, dict):
            continue
        tool_call_id = call.get("id")
        tool_name = call.get("name")
        arguments = call.get("arguments")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            continue
        if not isinstance(tool_name, str) or not tool_name:
            continue
        if not isinstance(arguments, str):
            continue
        calls.append(
            _ToolCallEnvelope(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments_json=arguments,
            )
        )
    return calls


def _final_from_envelope(output: Any) -> str:
    """Extract the final text from an envelope object."""
    if not isinstance(output, dict):
        return ""
    final = output.get("final")
    return final if isinstance(final, str) else ""


def _render_message_history(messages: Sequence[ModelMessage]) -> str:
    """Render a compact text representation of PydanticAI message history."""
    lines: List[str] = []

    for message in messages:
        if isinstance(message, ModelRequest):
            if message.instructions:
                lines.append("[instructions]")
                lines.append(message.instructions)
            for part in message.parts:
                kind = getattr(part, "part_kind", None)
                if kind == "system-prompt":
                    lines.append("[system]")
                    lines.append(getattr(part, "content", ""))
                elif kind == "user-prompt":
                    lines.append("[user]")
                    content = getattr(part, "content", "")
                    if isinstance(content, str):
                        lines.append(content)
                    else:
                        lines.append(_json_dumps(content))
                elif kind == "tool-return":
                    tool_name = getattr(part, "tool_name", "")
                    tool_call_id = getattr(part, "tool_call_id", "")
                    response = getattr(part, "model_response_str", None)
                    if callable(response):
                        tool_text = response()
                    else:
                        tool_text = _json_dumps(getattr(part, "content", None))
                    lines.append(f"[tool:{tool_name} id={tool_call_id}]")
                    lines.append(tool_text)
                elif kind == "retry-prompt":
                    response = getattr(part, "model_response", None)
                    lines.append("[retry]")
                    if callable(response):
                        lines.append(response())
                    else:
                        lines.append(_json_dumps(getattr(part, "content", "")))
                else:
                    lines.append("[request-part]")
                    lines.append(_json_dumps(part))
        else:
            lines.append("[assistant]")
            for part in message.parts:
                part_kind = getattr(part, "part_kind", None)
                if part_kind == "text":
                    lines.append(getattr(part, "content", ""))
                elif part_kind == "tool-call":
                    tool_name = getattr(part, "tool_name", "")
                    tool_call_id = getattr(part, "tool_call_id", "")
                    args = getattr(part, "args", None)
                    args_json = args if isinstance(args, str) else _json_dumps(args)
                    lines.append(
                        f"[tool-call:{tool_name} id={tool_call_id}] {args_json}"
                    )
                elif part_kind == "thinking":
                    pass
                else:
                    lines.append(f"[assistant-part:{part_kind}]")

    return "\n\n".join([line for line in lines if line]).strip()


def _now_utc() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _extract_json_object(text: str) -> Optional[Any]:
    """Best-effort parse of a JSON object from model text output."""
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            return None
        fragment = stripped[start : end + 1]
        try:
            return json.loads(fragment)
        except json.JSONDecodeError:
            return None


def _is_envelope_candidate(value: Any) -> bool:
    """Return whether a decoded JSON object looks like the envelope shape."""
    return isinstance(value, dict) and ("tool_calls" in value or "final" in value)


def _to_int(value: Any) -> int:
    """Best-effort int conversion for usage fields."""
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _usage_from_mapping(raw: Any) -> Optional[RequestUsage]:
    """Parse a RequestUsage from a raw mapping payload."""
    if not isinstance(raw, Mapping):
        return None

    input_tokens = _to_int(raw.get("inputTokens", raw.get("input_tokens")))
    cached_input_tokens = _to_int(
        raw.get("cachedInputTokens", raw.get("cached_input_tokens"))
    )
    output_tokens = _to_int(raw.get("outputTokens", raw.get("output_tokens")))
    details = (
        {"cached_input_tokens": cached_input_tokens} if cached_input_tokens else {}
    )
    return RequestUsage(
        input_tokens=input_tokens,
        cache_read_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        details=details,
    )


def _notification_items(notification: AppServerNotification) -> List[Dict[str, Any]]:
    """Extract item payloads from a notification payload (best effort)."""
    params = notification.params
    if not isinstance(params, Mapping):
        return []

    items: List[Dict[str, Any]] = []
    item = params.get("item")
    if isinstance(item, Mapping):
        items.append(dict(item))

    raw_items = params.get("items")
    if isinstance(raw_items, list):
        for raw in raw_items:
            if isinstance(raw, Mapping):
                items.append(dict(raw))

    turn = params.get("turn")
    if isinstance(turn, Mapping):
        turn_item = turn.get("item")
        if isinstance(turn_item, Mapping):
            items.append(dict(turn_item))
        turn_items = turn.get("items")
        if isinstance(turn_items, list):
            for raw in turn_items:
                if isinstance(raw, Mapping):
                    items.append(dict(raw))

    return items


def _extract_agent_text_from_item(item: Mapping[str, Any]) -> Optional[str]:
    """Extract agent text from a normalized item mapping."""
    item_type = item.get("type")
    if item_type not in {"agent_message", "agentMessage"}:
        return None
    text = item.get("text")
    if isinstance(text, str):
        return text
    content = item.get("content")
    if isinstance(content, str):
        return content
    return None


def _extract_usage_from_turn(
    turn: Optional[Mapping[str, Any]],
) -> Optional[RequestUsage]:
    """Extract usage metadata from a turn payload (best effort)."""
    if not isinstance(turn, Mapping):
        return None
    usage = _usage_from_mapping(turn.get("usage"))
    if usage is not None:
        return usage
    return _usage_from_mapping(turn.get("tokenUsage"))


def _extract_turn_text(turn: Optional[Mapping[str, Any]]) -> str:
    """Extract final text from a turn payload (best effort)."""
    if not isinstance(turn, Mapping):
        return ""

    text_keys = [
        "finalResponse",
        "final_response",
        "outputText",
        "output_text",
        "text",
        "output",
        "response",
    ]
    for key in text_keys:
        value = turn.get(key)
        if isinstance(value, str):
            return value

    output = turn.get("output")
    if isinstance(output, Mapping):
        for key in ("text", "final", "response"):
            value = output.get(key)
            if isinstance(value, str):
                return value

    items = turn.get("items")
    if isinstance(items, list):
        for raw in reversed(items):
            if not isinstance(raw, Mapping):
                continue
            text = _extract_agent_text_from_item(raw)
            if text:
                return text

    return ""


def _extract_turn_failure_message(notification: AppServerNotification) -> Optional[str]:
    """Extract a failure message from turn failure notifications."""
    if notification.method != "turn/failed":
        return None
    params = notification.params
    if not isinstance(params, Mapping):
        return "Turn failed"
    error = params.get("error")
    if isinstance(error, Mapping):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message
    message = params.get("message")
    if isinstance(message, str) and message:
        return message
    return "Turn failed"


class CodexStreamedResponse(StreamedResponse):
    """Incremental streamed response wrapper for app-server-backed CodexModel."""

    _STREAM_END = object()

    def __init__(
        self,
        *,
        model_request_parameters: ModelRequestParameters,
        model_name: str,
        provider_name: Optional[str],
        parts: Optional[Sequence[Any]] = None,
        thread_id: Optional[str] = None,
        usage: Optional[RequestUsage] = None,
    ) -> None:
        """Create a streamed response wrapper.

        Args:
            model_request_parameters: PydanticAI request parameters for this response.
            model_name: Model identifier to expose to PydanticAI.
            provider_name: Provider/system identifier for metadata.
            parts: Optional precomputed parts for compatibility with non-incremental
                stream construction.
            thread_id: Optional Codex thread identifier for provider details.
            usage: Optional request usage snapshot.
        """
        super().__init__(model_request_parameters=model_request_parameters)
        self._model_name = model_name
        self._provider_name = provider_name
        self._thread_id = thread_id
        self._usage = usage or RequestUsage()
        self._timestamp = _now_utc()
        self._instruction_queue: "asyncio.Queue[Any]" = asyncio.Queue()
        self._stream_error: Optional[BaseException] = None
        self._precomputed_parts: Optional[List[Any]] = list(parts) if parts else None

    def push_text_delta(self, *, vendor_part_id: Any, content: str) -> None:
        """Queue a text-delta instruction."""
        self._instruction_queue.put_nowait(
            _TextDeltaInstruction(vendor_part_id=vendor_part_id, content=content)
        )

    def push_tool_call(
        self,
        *,
        vendor_part_id: Any,
        tool_name: str,
        args: Any,
        tool_call_id: Optional[str],
    ) -> None:
        """Queue a tool-call instruction."""
        self._instruction_queue.put_nowait(
            _ToolCallInstruction(
                vendor_part_id=vendor_part_id,
                tool_name=tool_name,
                args=args,
                tool_call_id=tool_call_id,
            )
        )

    def finish(
        self,
        *,
        usage: Optional[RequestUsage] = None,
        thread_id: Optional[str] = None,
        error: Optional[BaseException] = None,
    ) -> None:
        """Mark the stream as complete and publish terminal metadata."""
        if usage is not None:
            self._usage = usage
        if thread_id is not None:
            self._thread_id = thread_id
        if error is not None:
            self._stream_error = error
        self._instruction_queue.put_nowait(self._STREAM_END)

    @staticmethod
    def _iter_events(raw: Any) -> Iterator[ModelResponseStreamEvent]:
        """Normalize parts-manager outputs to a stream of response events.

        PydanticAI changed `handle_text_delta(...)` from returning a single event to
        returning an iterator in newer releases. This helper supports both APIs.
        """
        if raw is None:
            return
        if hasattr(raw, "event_kind"):
            yield raw
            return
        try:
            iterator = iter(raw)
        except TypeError:
            logger.debug("Ignoring unexpected stream event payload: %r", raw)
            return
        for event in iterator:
            if event is None:
                continue
            if hasattr(event, "event_kind"):
                yield event
            else:
                logger.debug("Ignoring unknown stream event item: %r", event)

    async def _get_event_iterator(
        self,
    ) -> AsyncIterator[ModelResponseStreamEvent]:
        """Yield stream events as queued instructions arrive."""
        if self._precomputed_parts is not None:
            parts = self._precomputed_parts
            self._precomputed_parts = None
            for index, part in enumerate(parts):
                if isinstance(part, TextPart):
                    events = self._parts_manager.handle_text_delta(
                        vendor_part_id=index,
                        content=part.content,
                    )
                elif isinstance(part, ToolCallPart):
                    events = self._parts_manager.handle_tool_call_part(
                        vendor_part_id=index,
                        tool_name=part.tool_name,
                        args=part.args,
                        tool_call_id=part.tool_call_id,
                    )
                else:
                    logger.debug(
                        "Skipping unsupported streamed part",
                        extra={
                            "vendor_part_id": index,
                            "part_type": type(part).__name__,
                            "part_kind": getattr(part, "part_kind", None),
                        },
                    )
                    events = None
                for event in self._iter_events(events):
                    yield event
            if self._stream_error is not None:
                raise self._stream_error
            return

        while True:
            instruction = await self._instruction_queue.get()
            if instruction is self._STREAM_END:
                break
            if isinstance(instruction, _TextDeltaInstruction):
                events = self._parts_manager.handle_text_delta(
                    vendor_part_id=instruction.vendor_part_id,
                    content=instruction.content,
                )
            elif isinstance(instruction, _ToolCallInstruction):
                events = self._parts_manager.handle_tool_call_part(
                    vendor_part_id=instruction.vendor_part_id,
                    tool_name=instruction.tool_name,
                    args=instruction.args,
                    tool_call_id=instruction.tool_call_id,
                )
            else:
                logger.debug("Skipping unknown stream instruction: %r", instruction)
                events = None
            for event in self._iter_events(events):
                yield event

        if self._stream_error is not None:
            raise self._stream_error

    def get(self) -> ModelResponse:
        """Return a ModelResponse view over currently collected stream parts."""
        provider_details: Dict[str, Any] = {}
        if self._thread_id:
            provider_details["thread_id"] = self._thread_id
        return ModelResponse(
            parts=self._parts_manager.get_parts(),
            model_name=self.model_name,
            provider_name=self.provider_name,
            timestamp=self.timestamp,
            usage=self.usage(),
            provider_details=provider_details,
        )

    @property
    def model_name(self) -> str:
        """Return the model identifier used for the response."""
        return self._model_name

    @property
    def provider_name(self) -> Optional[str]:
        """Return the provider/system name for the response, if set."""
        return self._provider_name

    @property
    def provider_url(self) -> Optional[str]:
        """Return the provider URL when available (Codex currently does not expose one)."""
        return None

    @property
    def timestamp(self) -> datetime:
        """Get the UTC timestamp when this response object was created."""
        return self._timestamp


class CodexModel(Model):
    """Use Codex app-server as a PydanticAI model provider."""

    def __init__(
        self,
        *,
        app_server: Optional[AppServerClient] = None,
        app_server_options: Optional[AppServerOptions] = None,
        codex_options: Optional[CodexOptions] = None,
        thread_options: Optional[ThreadOptions] = None,
        profile: Optional[ModelProfileSpec] = None,
        settings: Optional[ModelSettings] = None,
        system: str = "openai",
        performance_profile: str = "balanced",
        thread_reuse_mode: str = "run",
    ) -> None:
        """Create an app-server-backed Codex model provider.

        The model defaults to app-server transport and run-scoped thread reuse.
        """
        if performance_profile not in {"balanced", "max"}:
            raise CodexError(
                "performance_profile must be 'balanced' or 'max'; "
                f"received {performance_profile!r}"
            )
        if thread_reuse_mode not in {"run", "always"}:
            raise CodexError(
                "thread_reuse_mode must be 'run' or 'always'; "
                f"received {thread_reuse_mode!r}"
            )

        self._performance_profile = performance_profile
        self._thread_reuse_mode = thread_reuse_mode
        self._thread_options = self._prepare_thread_options(thread_options)

        if profile is None:
            profile = ModelProfile(supports_tools=True)
        super().__init__(settings=settings, profile=profile)

        self._system = system
        self._request_lock = asyncio.Lock()
        self._closed = False

        self._app_client = app_server
        self._owns_app_client = app_server is None
        self._app_server_options: Optional[AppServerOptions] = (
            self._resolve_app_server_options(
                app_server_options=app_server_options,
                codex_options=codex_options,
            )
        )

        self._thread_id: Optional[str] = None
        self._messages_seen = 0

    @property
    def model_name(self) -> str:
        """Return the model identifier for this provider."""
        return self._thread_options.model or "codex"

    @property
    def system(self) -> str:
        """Return the provider system identifier (vendor name)."""
        return self._system

    def prepare_request(
        self,
        model_settings: Optional[ModelSettings],
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[Optional[ModelSettings], ModelRequestParameters]:
        """Hook to customize request settings/parameters before execution."""
        return model_settings, model_request_parameters

    async def close(self) -> None:
        """Close owned app-server resources and reset cached thread state."""
        async with self._request_lock:
            self._closed = True
            try:
                if self._owns_app_client and self._app_client is not None:
                    await self._app_client.close()
            finally:
                self._app_client = None
                self._thread_id = None
                self._messages_seen = 0

    async def _ensure_client(self) -> AppServerClient:
        """Ensure an app-server client exists and is started."""
        if self._closed:
            raise CodexError("CodexModel is closed")
        if self._app_server_options is None:
            raise CodexError("App-server options are not configured")

        if self._app_client is None:
            self._app_client = AppServerClient(self._app_server_options)
            self._owns_app_client = True

        await self._app_client.start()
        return self._app_client

    def _prepare_thread_options(
        self, thread_options: Optional[ThreadOptions]
    ) -> ThreadOptions:
        """Apply model-provider defaults to thread options."""
        if thread_options is None:
            thread_options = ThreadOptions()

        if thread_options.skip_git_repo_check is None:
            thread_options.skip_git_repo_check = True
        if thread_options.sandbox_mode is None:
            thread_options.sandbox_mode = "read-only"
        if thread_options.approval_policy is None:
            thread_options.approval_policy = "never"
        if (
            thread_options.web_search_mode is None
            and thread_options.web_search_enabled is None
            and thread_options.web_search_cached_enabled is None
        ):
            thread_options.web_search_mode = "disabled"
        if thread_options.network_access_enabled is None:
            thread_options.network_access_enabled = False

        if (
            self._performance_profile == "max"
            and thread_options.model_reasoning_effort is None
        ):
            thread_options.model_reasoning_effort = "minimal"

        return thread_options

    def _resolve_app_server_options(
        self,
        *,
        app_server_options: Optional[AppServerOptions],
        codex_options: Optional[CodexOptions],
    ) -> AppServerOptions:
        """Resolve app-server options."""
        if app_server_options is not None:
            return app_server_options

        if codex_options is None:
            return AppServerOptions()

        return AppServerOptions(
            codex_path_override=codex_options.codex_path_override,
            base_url=codex_options.base_url,
            api_key=codex_options.api_key,
            env=codex_options.env,
            config_overrides=codex_options.config_overrides,
        )

    def _thread_start_params(self) -> Dict[str, Any]:
        """Build app-server `thread/start` params from current thread options."""
        params: Dict[str, Any] = {}
        options = self._thread_options

        if options.model is not None:
            params["model"] = options.model
        if options.working_directory is not None:
            params["cwd"] = str(options.working_directory)
        if options.sandbox_mode is not None:
            params["sandbox_mode"] = options.sandbox_mode
        if options.skip_git_repo_check is not None:
            params["skip_git_repo_check"] = options.skip_git_repo_check
        if options.model_reasoning_effort is not None:
            params["model_reasoning_effort"] = options.model_reasoning_effort
        if options.model_instructions_file is not None:
            params["model_instructions_file"] = str(options.model_instructions_file)
        if options.model_personality is not None:
            params["model_personality"] = options.model_personality
        if options.max_threads is not None:
            params["max_threads"] = options.max_threads
        if options.network_access_enabled is not None:
            params["network_access_enabled"] = options.network_access_enabled
        if options.web_search_mode is not None:
            params["web_search_mode"] = options.web_search_mode
        if options.web_search_enabled is not None:
            params["web_search_enabled"] = options.web_search_enabled
        if options.web_search_cached_enabled is not None:
            params["web_search_cached_enabled"] = options.web_search_cached_enabled
        if options.shell_snapshot_enabled is not None:
            params["shell_snapshot_enabled"] = options.shell_snapshot_enabled
        if options.background_terminals_enabled is not None:
            params["background_terminals_enabled"] = (
                options.background_terminals_enabled
            )
        if options.apply_patch_freeform_enabled is not None:
            params["apply_patch_freeform_enabled"] = (
                options.apply_patch_freeform_enabled
            )
        if options.exec_policy_enabled is not None:
            params["exec_policy_enabled"] = options.exec_policy_enabled
        if options.remote_models_enabled is not None:
            params["remote_models_enabled"] = options.remote_models_enabled
        if options.collaboration_modes_enabled is not None:
            params["collaboration_modes_enabled"] = options.collaboration_modes_enabled
        if options.connectors_enabled is not None:
            params["connectors_enabled"] = options.connectors_enabled
        if options.responses_websockets_enabled is not None:
            params["responses_websockets_enabled"] = (
                options.responses_websockets_enabled
            )
        if options.request_compression_enabled is not None:
            params["request_compression_enabled"] = options.request_compression_enabled
        if options.feature_overrides is not None:
            params["feature_overrides"] = _jsonable(options.feature_overrides)
        if options.approval_policy is not None:
            params["approval_policy"] = options.approval_policy
        if options.additional_directories is not None:
            params["additional_directories"] = list(options.additional_directories)
        if options.config_overrides is not None:
            params["config_overrides"] = _jsonable(options.config_overrides)

        return params

    async def _ensure_thread(self, messages: Sequence[ModelMessage]) -> str:
        """Ensure a reusable Codex thread exists for the current message stream."""
        history_len = len(messages)
        if self._thread_id is not None:
            should_reset = history_len < self._messages_seen
            if self._thread_reuse_mode == "run" and history_len <= self._messages_seen:
                should_reset = True
            if should_reset:
                self._thread_id = None
                self._messages_seen = 0

        if self._thread_id:
            return self._thread_id

        client = await self._ensure_client()
        response = await client.thread_start(**self._thread_start_params())
        thread = response.get("thread") if isinstance(response, dict) else None
        thread_id = thread.get("id") if isinstance(thread, Mapping) else None
        if not isinstance(thread_id, str) or not thread_id:
            raise CodexError("thread/start response missing thread id")
        self._thread_id = thread_id
        self._messages_seen = 0
        return thread_id

    def _slice_incremental_messages(
        self, messages: Sequence[ModelMessage]
    ) -> tuple[List[ModelMessage], int]:
        """Return only messages not yet sent to Codex thread state."""
        start = min(self._messages_seen, len(messages))
        incremental = list(messages[start:])
        if not incremental and messages:
            incremental = [messages[-1]]
        return incremental, len(messages)

    def _build_prompt(
        self,
        *,
        messages: Sequence[ModelMessage],
        model_request_parameters: ModelRequestParameters,
    ) -> str:
        """Build prompt text for a single app-server turn."""
        tool_defs = [
            *model_request_parameters.function_tools,
            *model_request_parameters.output_tools,
        ]
        tool_names = [tool.name for tool in tool_defs]
        envelope_schema = _build_envelope_schema(tool_names)
        tool_manifest = _render_tool_definitions(
            function_tools=model_request_parameters.function_tools,
            output_tools=model_request_parameters.output_tools,
        )
        allow_text_output = model_request_parameters.allow_text_output

        prompt_sections = [
            "You are a model in a tool-calling loop controlled by the host application.",
            "You MUST NOT run shell commands, edit files, or call any built-in tools.",
            "Request tools ONLY by emitting tool calls in the JSON output (matching the output schema).",
            "",
            "JSON output rules:",
            "- Always return an object with keys: tool_calls (array) and final (string).",
            '- Each tool call is: {"id": "...", "name": "...", "arguments": "{...json...}"}',
            "- arguments MUST be a JSON string encoding an object.",
            "- If you are calling any tools, set final to an empty string.",
            f"- The output object must validate this schema: {_json_dumps(envelope_schema)}",
        ]
        if allow_text_output:
            prompt_sections.append(
                "- If no tools are needed, set tool_calls to [] and put your full answer in final."
            )
        else:
            prompt_sections.append(
                "- Text output is NOT allowed; to finish, call exactly one output tool and keep final empty."
            )

        if tool_manifest:
            prompt_sections.extend(["", tool_manifest])

        history = _render_message_history(messages)
        if history:
            prompt_sections.extend(["", "Conversation so far:", history])

        return "\n".join(prompt_sections).strip()

    def _handle_notification(
        self,
        *,
        notification: AppServerNotification,
        state: _TurnAccumulationState,
        streamed: Optional[CodexStreamedResponse],
        allow_stream_text: bool,
    ) -> None:
        """Update accumulated state from one app-server notification."""
        failure = _extract_turn_failure_message(notification)
        if failure:
            state.turn_failed_message = failure

        params = notification.params
        if isinstance(params, Mapping):
            usage = _usage_from_mapping(params.get("usage"))
            if usage is not None:
                state.usage = usage
            turn = params.get("turn")
            if isinstance(turn, Mapping):
                turn_usage = _extract_usage_from_turn(turn)
                if turn_usage is not None:
                    state.usage = turn_usage

        for item in _notification_items(notification):
            text = _extract_agent_text_from_item(item)
            if not text:
                continue

            state.latest_agent_text = text
            if not allow_stream_text or streamed is None:
                continue

            item_id_raw = item.get("id")
            item_id = (
                str(item_id_raw) if item_id_raw is not None else "__agent_message__"
            )
            previous = state.item_text_by_id.get(item_id, "")
            if text.startswith(previous):
                delta = text[len(previous) :]
            else:
                delta = text
            state.item_text_by_id[item_id] = text
            state.last_updated_item_id = item_id
            if not delta:
                continue

            if item_id not in state.vendor_part_ids:
                state.vendor_part_ids[item_id] = state.next_vendor_part_id
                state.next_vendor_part_id += 1
            vendor_part_id = state.vendor_part_ids[item_id]
            streamed.push_text_delta(vendor_part_id=vendor_part_id, content=delta)

    async def _run_turn(
        self,
        *,
        thread_id: str,
        prompt: str,
        model_request_parameters: ModelRequestParameters,
        streamed: Optional[CodexStreamedResponse] = None,
    ) -> tuple[List[Any], RequestUsage]:
        """Execute one app-server turn and parse it into PydanticAI parts."""
        client = await self._ensure_client()
        session = await client.turn_session(thread_id, prompt)
        state = _TurnAccumulationState()

        allow_stream_text = bool(
            streamed is not None
            and model_request_parameters.allow_text_output
            and not model_request_parameters.function_tools
            and not model_request_parameters.output_tools
        )

        async for notification in session.notifications():
            self._handle_notification(
                notification=notification,
                state=state,
                streamed=streamed,
                allow_stream_text=allow_stream_text,
            )

        final_turn = await session.wait()
        if state.turn_failed_message:
            raise TurnFailedError(state.turn_failed_message)

        usage = state.usage or _extract_usage_from_turn(final_turn) or RequestUsage()
        final_text = _extract_turn_text(final_turn)
        if not final_text:
            final_text = state.latest_agent_text

        parsed_json = _extract_json_object(final_text) if final_text else None
        parsed_envelope = parsed_json if _is_envelope_candidate(parsed_json) else None

        parts: List[Any] = []
        tool_calls = _tool_calls_from_envelope(parsed_envelope)
        if tool_calls:
            for index, call in enumerate(tool_calls):
                parts.append(
                    ToolCallPart(
                        tool_name=call.tool_name,
                        args=call.arguments_json,
                        tool_call_id=call.tool_call_id,
                    )
                )
                if streamed is not None:
                    streamed.push_tool_call(
                        vendor_part_id=index,
                        tool_name=call.tool_name,
                        args=call.arguments_json,
                        tool_call_id=call.tool_call_id,
                    )
            return parts, usage

        if parsed_envelope is not None:
            final_value = _final_from_envelope(parsed_envelope)
        else:
            final_value = final_text

        if model_request_parameters.allow_text_output and final_value:
            parts.append(TextPart(final_value))
            if streamed is not None:
                if allow_stream_text:
                    if not state.item_text_by_id:
                        streamed.push_text_delta(vendor_part_id=0, content=final_value)
                    else:
                        last_item_id = (
                            state.last_updated_item_id
                            if state.last_updated_item_id in state.item_text_by_id
                            else next(reversed(state.item_text_by_id))
                        )
                        last_text = state.item_text_by_id[last_item_id]
                        if final_value.startswith(last_text):
                            remainder = final_value[len(last_text) :]
                            if remainder:
                                vendor_part_id = state.vendor_part_ids.get(
                                    last_item_id, 0
                                )
                                streamed.push_text_delta(
                                    vendor_part_id=vendor_part_id,
                                    content=remainder,
                                )
                        elif final_value != last_text:
                            streamed.push_text_delta(
                                vendor_part_id=state.next_vendor_part_id,
                                content=final_value,
                            )
                else:
                    streamed.push_text_delta(vendor_part_id=0, content=final_value)

        return parts, usage

    async def _run_codex_request(
        self,
        messages: list[ModelMessage],
        model_settings: Optional[ModelSettings],
        model_request_parameters: ModelRequestParameters,
        streamed: Optional[CodexStreamedResponse] = None,
    ) -> tuple[List[Any], RequestUsage, str, ModelRequestParameters]:
        """Run one model request via app-server transport."""
        model_settings, model_request_parameters = self.prepare_request(
            model_settings, model_request_parameters
        )
        del model_settings

        thread_id = await self._ensure_thread(messages)
        incremental_messages, seen_after = self._slice_incremental_messages(messages)
        prompt = self._build_prompt(
            messages=incremental_messages,
            model_request_parameters=model_request_parameters,
        )

        with span(
            "codex_sdk.pydantic_ai.model_request",
            model=self._thread_options.model,
            sandbox_mode=self._thread_options.sandbox_mode,
            transport="app_server",
            performance_profile=self._performance_profile,
        ):
            parts, usage = await self._run_turn(
                thread_id=thread_id,
                prompt=prompt,
                model_request_parameters=model_request_parameters,
                streamed=streamed,
            )

        self._messages_seen = seen_after
        return parts, usage, thread_id, model_request_parameters

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: Optional[ModelSettings],
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Run a request and return a completed model response."""
        async with self._request_lock:
            try:
                parts, usage, thread_id, _ = await self._run_codex_request(
                    messages=messages,
                    model_settings=model_settings,
                    model_request_parameters=model_request_parameters,
                )
            except Exception:
                self._thread_id = None
                self._messages_seen = 0
                raise

        return ModelResponse(
            parts=parts,
            usage=usage,
            model_name=self.model_name,
            provider_name=self.system,
            provider_details={"thread_id": thread_id},
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: Optional[ModelSettings],
        model_request_parameters: ModelRequestParameters,
        run_context: Optional[Any] = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Run a request and stream incremental response events."""
        del run_context
        async with self._request_lock:
            streamed = CodexStreamedResponse(
                model_request_parameters=model_request_parameters,
                model_name=self.model_name,
                provider_name=self.system,
            )

            async def _runner() -> None:
                thread_id: Optional[str] = self._thread_id
                try:
                    parts, usage, thread_id, _ = await self._run_codex_request(
                        messages=messages,
                        model_settings=model_settings,
                        model_request_parameters=model_request_parameters,
                        streamed=streamed,
                    )
                    del parts
                    streamed.finish(usage=usage, thread_id=thread_id)
                except Exception as exc:  # pragma: no cover - defensive
                    self._thread_id = None
                    self._messages_seen = 0
                    streamed.finish(thread_id=thread_id, error=exc)

            task = asyncio.create_task(_runner())
            try:
                yield streamed
                await task
            finally:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        # Expected when cancelling the background task during cleanup; safe to ignore.
                        pass
