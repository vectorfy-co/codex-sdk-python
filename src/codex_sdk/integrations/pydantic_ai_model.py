"""PydanticAI model-provider integration.

This module provides a `pydantic_ai.models.Model` implementation that delegates
completion + tool-call planning to Codex via `codex exec --output-schema`.

The goal is to let PydanticAI own the tool loop (tool execution, retries, output
validation), while Codex behaves like a "backend model" that emits either:

- tool calls (to be executed by PydanticAI), or
- a final text response (when text output is allowed).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import InitVar, dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, cast

from ..codex import Codex
from ..options import CodexOptions, ThreadOptions
from ..telemetry import span
from ..thread import TurnOptions
from ..tool_envelope import (
    ToolCallEnvelope,
    ToolPlanValidationError,
    build_envelope_schema,
    json_dumps,
    jsonable,
    parse_tool_plan,
)

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
    from pydantic_ai.usage import Usage as _Usage

    try:
        from pydantic_ai.usage import RequestUsage as _RequestUsage
    except ImportError:
        _RequestUsage = _Usage
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pydantic-ai is required for codex_sdk.integrations.pydantic_ai_model; "
        'install with: uv add "codex-sdk-python[pydantic-ai]"'
    ) from exc

_REQUEST_USAGE_CTOR = cast(Any, _RequestUsage)
_MODEL_RESPONSE_FIELDS = set(getattr(ModelResponse, "__dataclass_fields__", {}).keys())


def _jsonable(value: Any) -> Any:
    return jsonable(value)


def _json_dumps(value: Any) -> str:
    return json_dumps(value)


def _build_envelope_schema(tool_names: Sequence[str]) -> Dict[str, Any]:
    return build_envelope_schema(tool_names)


def _render_tool_definitions(
    *,
    function_tools: Sequence[ToolDefinition],
    output_tools: Sequence[ToolDefinition],
) -> str:
    lines: List[str] = []
    if function_tools:
        lines.append("Function tools:")
        for tool in function_tools:
            lines.append(f"- {tool.name}")
            if tool.description:
                lines.append(f"  description: {tool.description}")
            kind = getattr(tool, "kind", None)
            if kind is not None:
                lines.append(f"  kind: {kind}")
            lines.append(
                f"  parameters_json_schema: {_json_dumps(tool.parameters_json_schema)}"
            )
            outer_typed_dict_key = getattr(tool, "outer_typed_dict_key", None)
            if outer_typed_dict_key:
                lines.append(f"  outer_typed_dict_key: {outer_typed_dict_key}")
            strict = getattr(tool, "strict", None)
            if strict is not None:
                lines.append(f"  strict: {str(strict).lower()}")
            if bool(getattr(tool, "sequential", False)):
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
            kind = getattr(tool, "kind", None)
            if kind is not None:
                lines.append(f"  kind: {kind}")
            lines.append(
                f"  parameters_json_schema: {_json_dumps(tool.parameters_json_schema)}"
            )
            outer_typed_dict_key = getattr(tool, "outer_typed_dict_key", None)
            if outer_typed_dict_key:
                lines.append(f"  outer_typed_dict_key: {outer_typed_dict_key}")
            strict = getattr(tool, "strict", None)
            if strict is not None:
                lines.append(f"  strict: {str(strict).lower()}")
            if bool(getattr(tool, "sequential", False)):
                lines.append("  sequential: true")
            metadata = getattr(tool, "metadata", None)
            if metadata is not None:
                lines.append(f"  metadata: {_json_dumps(metadata)}")
            timeout = getattr(tool, "timeout", None)
            if timeout is not None:
                lines.append(f"  timeout: {timeout}")

    return "\n".join(lines).strip()


def _tool_calls_from_envelope(output: Any) -> List[ToolCallEnvelope]:
    try:
        plan = parse_tool_plan(output)
    except ToolPlanValidationError:
        return []

    if plan.kind != "tool_calls":
        return []
    return list(plan.calls)


def _final_from_envelope(output: Any) -> str:
    try:
        plan = parse_tool_plan(output)
    except ToolPlanValidationError:
        return ""
    if plan.kind != "final":
        return ""
    return plan.content


def _render_message_history(messages: Sequence[ModelMessage]) -> str:
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
    return datetime.now(timezone.utc)


def _make_request_usage(turn_usage: Optional[Any]) -> _Usage:
    usage = cast(_Usage, _REQUEST_USAGE_CTOR())
    if turn_usage is None:
        return usage

    input_tokens = int(turn_usage.input_tokens)
    output_tokens = int(turn_usage.output_tokens)
    cached_input_tokens = int(turn_usage.cached_input_tokens)

    if hasattr(usage, "input_tokens"):
        return cast(
            _Usage,
            _REQUEST_USAGE_CTOR(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cached_input_tokens,
            ),
        )

    details = (
        {"cache_read_tokens": cached_input_tokens} if cached_input_tokens else None
    )
    return cast(
        _Usage,
        _REQUEST_USAGE_CTOR(
            requests=1,
            request_tokens=input_tokens,
            response_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            details=details,
        ),
    )


def _attach_response_details(response: ModelResponse, details: Dict[str, Any]) -> None:
    if "provider_details" in _MODEL_RESPONSE_FIELDS:
        setattr(response, "provider_details", details)
    elif "vendor_details" in _MODEL_RESPONSE_FIELDS:
        setattr(response, "vendor_details", details)


def _build_model_response(
    *,
    parts: List[Any],
    usage: _Usage,
    model_name: str,
    details: Dict[str, Any],
) -> ModelResponse:
    kwargs: Dict[str, Any] = {"parts": parts, "usage": usage, "model_name": model_name}
    if "provider_name" in _MODEL_RESPONSE_FIELDS:
        kwargs["provider_name"] = "codex"
    if "provider_details" in _MODEL_RESPONSE_FIELDS:
        kwargs["provider_details"] = details
    elif "vendor_details" in _MODEL_RESPONSE_FIELDS:
        kwargs["vendor_details"] = details
    return ModelResponse(**kwargs)


@dataclass
class CodexStreamedResponse(StreamedResponse):
    """Minimal streamed response wrapper for Codex model provider."""

    _model_name: str
    _provider_name: str
    _provider_url: Optional[str]
    _parts: Sequence[Any]
    _usage_init: InitVar[_Usage]
    _response_details: Dict[str, Any] = field(default_factory=dict)
    _timestamp: datetime = field(default_factory=_now_utc, init=False)

    def __post_init__(self, _usage_init: _Usage) -> None:
        self._usage = _usage_init

    async def _get_event_iterator(
        self,
    ) -> AsyncIterator[ModelResponseStreamEvent]:
        for index, part in enumerate(self._parts):
            if isinstance(part, TextPart):
                event = self._parts_manager.handle_text_delta(
                    vendor_part_id=index,
                    content=part.content,
                )
                if event is not None:
                    yield event
                continue

            if isinstance(part, ToolCallPart):
                yield self._parts_manager.handle_tool_call_part(
                    vendor_part_id=index,
                    tool_name=part.tool_name,
                    args=part.args,
                    tool_call_id=part.tool_call_id,
                )
                continue

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def provider_url(self) -> Optional[str]:
        return self._provider_url

    @property
    def timestamp(self) -> datetime:
        return self._timestamp

    def get(self) -> ModelResponse:
        response = super().get()
        _attach_response_details(response, self._response_details)
        return response


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
    ) -> tuple[List[Any], _Usage, str]:
        prepare_request = getattr(self, "prepare_request", None)
        if callable(prepare_request):
            model_settings, model_request_parameters = prepare_request(
                model_settings,
                model_request_parameters,
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

        usage = _make_request_usage(parsed_turn.turn.usage)

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
        return parts, usage, thread_id

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: Optional[ModelSettings],
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        parts, usage, thread_id = await self._run_codex_request(
            messages,
            model_settings,
            model_request_parameters,
        )
        return _build_model_response(
            parts=parts,
            usage=usage,
            model_name=self.model_name,
            details={"thread_id": thread_id},
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: Optional[ModelSettings],
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        parts, usage, thread_id = await self._run_codex_request(
            messages,
            model_settings,
            model_request_parameters,
        )
        streamed = CodexStreamedResponse(
            _model_name=self.model_name,
            _provider_name="codex",
            _provider_url=None,
            _parts=parts,
            _usage_init=usage,
            _response_details={"thread_id": thread_id},
        )
        yield streamed
