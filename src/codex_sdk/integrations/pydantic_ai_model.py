"""PydanticAI model-provider integration.

This module provides a `pydantic_ai.models.Model` implementation that delegates
completion + tool-call planning to Codex via `codex exec --output-schema`.

The goal is to let PydanticAI own the tool loop (tool execution, retries, output
validation), while Codex behaves like a "backend model" that emits either:

- tool calls (to be executed by PydanticAI), or
- a final text response (when text output is allowed).
"""

from __future__ import annotations

import json
from base64 import b64encode
from contextlib import asynccontextmanager
from dataclasses import InitVar, asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

from ..codex import Codex
from ..options import CodexOptions, ThreadOptions
from ..telemetry import span
from ..thread import TurnOptions

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
    from pydantic_ai.usage import Usage as PydanticUsage
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pydantic-ai is required for codex_sdk.integrations.pydantic_ai_model; "
        'install with: uv add "codex-sdk-python[pydantic-ai]"'
    ) from exc


@dataclass(frozen=True)
class _ToolCallEnvelope:
    """Parsed tool-call envelope returned by Codex `--output-schema` turns."""

    tool_call_id: str
    tool_name: str
    arguments_json: str


def _jsonable(value: Any) -> Any:
    """Convert values into JSON-serializable structures for prompt/debug output."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "model_dump") and callable(getattr(value, "model_dump")):
        return value.model_dump(mode="json")
    if isinstance(value, bytes):
        return {"type": "bytes", "base64": b64encode(value).decode("ascii")}
    if isinstance(value, dict):
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
    """
    Render a human-readable description of provided function and output tool definitions for inclusion in prompts.

    Produces a newline-separated block that lists each tool with name, optional description, kind, parameters JSON schema, and optional fields such as outer_typed_dict_key, strict, sequential, metadata, and timeout. Function tools are listed under "Function tools:" and output tools under "Output tools (use ONE of these to finish when text is not allowed):".

    Parameters:
        function_tools (Sequence[ToolDefinition]): Tool definitions intended to be called as functions.
        output_tools (Sequence[ToolDefinition]): Tool definitions intended as final output options when direct text is disallowed.

    Returns:
        str: The rendered, trimmed multi-line string describing the tools.
    """
    lines: List[str] = []
    if function_tools:
        lines.append("Function tools:")
        for tool in function_tools:
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

    if output_tools:
        if lines:
            lines.append("")
        lines.append(
            "Output tools (use ONE of these to finish when text is not allowed):"
        )
        for tool in output_tools:
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

    return "\n".join(lines).strip()


def _tool_calls_from_envelope(output: Any) -> List[_ToolCallEnvelope]:
    """Extract tool call envelopes from a Codex JSON turn output."""
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
    """Extract the final text from a Codex JSON turn output."""
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
            # ModelResponse
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
                    # Intentionally omit to reduce prompt noise.
                    pass
                else:
                    lines.append(f"[assistant-part:{part_kind}]")

    return "\n\n".join([line for line in lines if line]).strip()


def _now_utc() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


@dataclass
class CodexStreamedResponse(StreamedResponse):
    """Minimal streamed response wrapper for Codex model provider."""

    _model_name: str
    _parts: Sequence[Any]
    _thread_id: str
    _usage_init: InitVar[PydanticUsage]
    _timestamp: datetime = field(default_factory=_now_utc, init=False)

    def __post_init__(self, _usage_init: PydanticUsage) -> None:
        """
        Save the provided PydanticUsage into the instance's _usage attribute.

        Parameters:
            _usage_init (PydanticUsage): Usage information supplied as the dataclass InitVar.
        """
        self._usage = _usage_init

    async def _get_event_iterator(
        self,
    ) -> AsyncIterator[ModelResponseStreamEvent]:
        """
        Iterates over stored response parts and yields stream events for text deltas and tool-call parts.

        The iterator converts each TextPart into a text-delta event and each ToolCallPart into a tool-call event using the parts manager; other part kinds are ignored.

        Returns:
            ModelResponseStreamEvent: An event for each text or tool-call part, yielded in the original parts order.
        """
        for index, part in enumerate(self._parts):
            if isinstance(part, TextPart):
                event = self._parts_manager.handle_text_delta(
                    vendor_part_id=index,
                    content=part.content,
                )
            elif isinstance(part, ToolCallPart):
                event = self._parts_manager.handle_tool_call_part(
                    vendor_part_id=index,
                    tool_name=part.tool_name,
                    args=part.args,
                    tool_call_id=part.tool_call_id,
                )
            else:
                event = None

            if event is not None:
                yield event

    def get(self) -> ModelResponse:
        """
        Construct a complete model response from events received so far.

        Returns:
            ModelResponse: Contains collected parts, the model name, timestamp, usage, and `vendor_details` with the Codex thread_id.
        """
        return ModelResponse(
            parts=self._parts_manager.get_parts(),
            model_name=self.model_name,
            timestamp=self.timestamp,
            usage=self.usage(),
            vendor_details={"thread_id": self._thread_id},
        )

    @property
    def model_name(self) -> str:
        """
        Return the model identifier used for the response.

        Returns:
            The model identifier string.
        """
        return self._model_name

    @property
    def timestamp(self) -> datetime:
        """
        Get the UTC timestamp when this response was created.

        Returns:
            datetime: The timestamp associated with the response.
        """
        return self._timestamp


class CodexModel(Model):
    """Use Codex CLI as a PydanticAI model provider via structured output."""

    def __init__(
        self,
        *,
        codex: Optional[Codex] = None,
        codex_options: Optional[CodexOptions] = None,
        thread_options: Optional[ThreadOptions] = None,
        profile: Optional[ModelProfileSpec] = None,
        settings: Optional[ModelSettings] = None,
        system: str = "openai",
    ) -> None:
        if codex is None:
            codex = Codex(codex_options or CodexOptions())
        if thread_options is None:
            thread_options = ThreadOptions()

        # As a model-provider wrapper, prefer safe + portable defaults.
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

        if profile is None:
            profile = ModelProfile(supports_tools=True)

        super().__init__(settings=settings, profile=profile)
        self._codex = codex
        self._thread_options = thread_options
        self._system = system

    @property
    def model_name(self) -> str:
        return self._thread_options.model or "codex"

    @property
    def system(self) -> str:
        return self._system

    async def _run_codex_request(
        self,
        messages: list[ModelMessage],
        model_settings: Optional[ModelSettings],
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[List[Any], PydanticUsage, str, ModelRequestParameters]:
        """
        Run a Codex thread for the given conversation and request parameters, and parse the JSON envelope into response parts.

        Parameters:
            messages (list[ModelMessage]): Conversation messages to include in the Codex prompt.
            model_settings (Optional[ModelSettings]): Ignored by this implementation.
            model_request_parameters (ModelRequestParameters): Controls function/output tool definitions, whether text output is allowed, and may be customized before the request.

        Returns:
            tuple[List[Any], PydanticUsage, str, ModelRequestParameters]:
                - parts: A list of response parts produced from the envelope (e.g., ToolCallPart instances for tool calls or TextPart for final text).
                - usage: Usage information for the request as a PydanticUsage instance.
                - thread_id: The Codex thread identifier used for the request.
                - model_request_parameters: The (possibly customized) ModelRequestParameters actually used for the request.
        """
        del model_settings
        model_request_parameters = self.customize_request_parameters(
            model_request_parameters
        )

        tool_defs = [
            *model_request_parameters.function_tools,
            *model_request_parameters.output_tools,
        ]
        tool_names = [tool.name for tool in tool_defs]
        output_schema = _build_envelope_schema(tool_names)

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

        prompt = "\n".join(prompt_sections).strip()

        with span(
            "codex_sdk.pydantic_ai.model_request",
            model=self._thread_options.model,
            sandbox_mode=self._thread_options.sandbox_mode,
        ):
            thread = self._codex.start_thread(self._thread_options)
            parsed_turn = await thread.run_json(
                prompt, output_schema=output_schema, turn_options=TurnOptions()
            )

        usage = PydanticUsage(requests=1)
        if parsed_turn.turn.usage is not None:
            cached = parsed_turn.turn.usage.cached_input_tokens
            details = {"cached_input_tokens": cached} if cached else None
            usage = PydanticUsage(
                requests=1,
                request_tokens=parsed_turn.turn.usage.input_tokens,
                response_tokens=parsed_turn.turn.usage.output_tokens,
                total_tokens=parsed_turn.turn.usage.input_tokens
                + parsed_turn.turn.usage.output_tokens,
                details=details,
            )

        tool_calls = _tool_calls_from_envelope(parsed_turn.output)
        parts: List[Any] = []
        if tool_calls:
            parts.extend(
                ToolCallPart(
                    tool_name=call.tool_name,
                    args=call.arguments_json,
                    tool_call_id=call.tool_call_id,
                )
                for call in tool_calls
            )
        else:
            final = _final_from_envelope(parsed_turn.output)
            if allow_text_output and final:
                parts.append(TextPart(final))

        thread_id = thread.id
        assert thread_id is not None
        return parts, usage, thread_id, model_request_parameters

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: Optional[ModelSettings],
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """
        Send the provided message history to Codex and return a ModelResponse containing the model output and metadata.

        Parameters:
            messages: The conversation as a list of ModelMessage objects to send to the model.
            model_settings: Optional model configuration (may be ignored by the Codex backend).
            model_request_parameters: Request-specific parameters that influence Codex execution.

        Returns:
            ModelResponse containing the generated parts, usage information, the model name, and vendor_details with the Codex `thread_id`.
        """
        parts, usage, thread_id, _ = await self._run_codex_request(
            messages,
            model_settings,
            model_request_parameters,
        )
        return ModelResponse(
            parts=parts,
            usage=usage,
            model_name=self.model_name,
            vendor_details={"thread_id": thread_id},
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: Optional[ModelSettings],
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        """
        Produce an asynchronous stream that yields a Codex-backed streamed model response for the given message sequence.

        Parameters:
            messages: Conversation messages to send to the model.
            model_settings: Model-specific settings (may be None).
            model_request_parameters: Additional request parameters controlling the model call.

        Returns:
            An async iterator that yields a single StreamedResponse (CodexStreamedResponse) containing the model name, response parts, usage information, and the Codex thread identifier.
        """
        parts, usage, thread_id, _ = await self._run_codex_request(
            messages,
            model_settings,
            model_request_parameters,
        )
        streamed = CodexStreamedResponse(
            _model_name=self.model_name,
            _parts=parts,
            _thread_id=thread_id,
            _usage_init=usage,
        )
        yield streamed
