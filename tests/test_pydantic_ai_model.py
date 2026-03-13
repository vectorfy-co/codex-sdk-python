import dataclasses
import importlib
import json
from contextlib import asynccontextmanager

import pytest

# ruff: noqa: E402
# Imports are intentionally ordered around import-or-skip behavior.

pydantic = pytest.importorskip("pydantic")
pytest.importorskip("pydantic_ai", exc_type=ImportError)
BaseModel = pydantic.BaseModel

from codex_sdk.app_server import AppServerNotification, AppServerOptions
from codex_sdk.exceptions import CodexError, TurnFailedError
from codex_sdk.integrations.pydantic_ai_model import (
    CodexModel,
    CodexStreamedResponse,
    _build_envelope_schema,
    _extract_json_object,
    _extract_turn_failure_message,
    _extract_turn_text,
    _extract_usage_from_turn,
    _final_from_envelope,
    _is_envelope_candidate,
    _json_dumps,
    _jsonable,
    _notification_items,
    _render_message_history,
    _render_tool_definitions,
    _to_int,
    _tool_calls_from_envelope,
    _TurnAccumulationState,
    _usage_from_mapping,
)
from codex_sdk.options import CodexOptions, ThreadOptions

messages = importlib.import_module("pydantic_ai.messages")
models = importlib.import_module("pydantic_ai.models")
pydantic_ai = importlib.import_module("pydantic_ai")
tools = importlib.import_module("pydantic_ai.tools")

Agent = pydantic_ai.Agent
AgentRunResultEvent = pydantic_ai.AgentRunResultEvent
BuiltinToolCallPart = messages.BuiltinToolCallPart
ModelRequest = messages.ModelRequest
ModelResponse = messages.ModelResponse
PartStartEvent = messages.PartStartEvent
RetryPromptPart = messages.RetryPromptPart
SystemPromptPart = messages.SystemPromptPart
TextPart = messages.TextPart
ThinkingPart = messages.ThinkingPart
ToolCallPart = messages.ToolCallPart
ToolReturnPart = messages.ToolReturnPart
UserPromptPart = messages.UserPromptPart
ModelRequestParameters = models.ModelRequestParameters
ToolDefinition = tools.ToolDefinition
RequestUsage = importlib.import_module("pydantic_ai.usage").RequestUsage


def _usage_input_tokens(usage):
    return usage.input_tokens


def _usage_cached_input_tokens(usage):
    return usage.cache_read_tokens


def _usage_output_tokens(usage):
    return usage.output_tokens


def _envelope_json(output):
    return json.dumps(output, separators=(",", ":"))


@pytest.mark.asyncio
async def test_codex_model_returns_tool_calls():
    output = {
        "tool_calls": [
            {"id": "call_1", "name": "add", "arguments": '{"a":1,"b":2}'},
        ],
        "final": "",
    }
    app = FakeAppServerClient(
        notifications=[],
        final_turn={
            "id": "turn-tools",
            "usage": {"inputTokens": 1, "cachedInputTokens": 2, "outputTokens": 3},
            "finalResponse": _envelope_json(output),
        },
        thread_id="thread-123",
    )
    model = CodexModel(app_server=app)

    messages = [ModelRequest(parts=[UserPromptPart("hi")])]
    params = ModelRequestParameters(
        function_tools=[
            ToolDefinition(
                name="add",
                description="add two ints",
                parameters_json_schema={
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            )
        ]
    )

    response = await model.request(messages, None, params)
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], ToolCallPart)
    assert response.parts[0].tool_name == "add"
    assert response.parts[0].tool_call_id == "call_1"
    assert response.parts[0].args == '{"a":1,"b":2}'
    assert _usage_input_tokens(response.usage) == 1
    assert _usage_cached_input_tokens(response.usage) == 2
    assert _usage_output_tokens(response.usage) == 3
    assert response.usage.details == {"cached_input_tokens": 2}

    prompt = app.turn_session_calls[0]["input"]
    assert '"enum":["add"]' in prompt


def test_render_tool_definitions_renders_optional_tool_fields() -> None:
    """Cover optional tool attributes rendered via getattr()."""
    func = ToolDefinition(name="func", description="d1")
    func.sequential = True
    func.metadata = {"x": 1}
    func.timeout = 3.0

    func_no_desc = ToolDefinition(name="func2", description=None)

    out = ToolDefinition(name="out", description="d2")
    out.sequential = True
    out.metadata = {"y": 2}
    out.timeout = 9

    out_no_desc = ToolDefinition(name="out2", description=None)

    rendered = _render_tool_definitions(
        function_tools=[func, func_no_desc], output_tools=[out, out_no_desc]
    )
    assert "Function tools:" in rendered
    assert (
        "Output tools (use ONE of these to finish when text is not allowed):"
        in rendered
    )
    # This test intentionally injects `sequential` on dummy tool objects.
    assert "sequential: true" in rendered
    assert "metadata:" in rendered
    assert "timeout:" in rendered
    assert "- func2" in rendered
    assert "- out2" in rendered


def test_codex_model_does_not_override_explicit_thread_options() -> None:
    """Cover CodexModel.__init__ branches when thread options are preconfigured."""
    profiles = importlib.import_module("pydantic_ai.profiles")

    app = FakeAppServerClient(
        notifications=[],
        final_turn={
            "id": "turn-explicit",
            "usage": {"inputTokens": 1, "outputTokens": 1},
            "finalResponse": _envelope_json({"tool_calls": [], "final": "ok"}),
        },
    )
    thread_options = importlib.import_module("codex_sdk.options").ThreadOptions(
        skip_git_repo_check=False,
        sandbox_mode="workspace-write",
        approval_policy="on-request",
        web_search_mode="live",
        network_access_enabled=True,
    )

    profile = profiles.ModelProfile(supports_tools=False)
    model = CodexModel(app_server=app, thread_options=thread_options, profile=profile)

    assert model.model_name == "codex"
    assert thread_options.skip_git_repo_check is False
    assert thread_options.sandbox_mode == "workspace-write"
    assert thread_options.approval_policy == "on-request"
    assert thread_options.web_search_mode == "live"
    assert thread_options.network_access_enabled is True


@pytest.mark.asyncio
async def test_codex_model_usage_defaults_when_thread_usage_missing() -> None:
    """Cover the branch where turn.usage is None and we return minimal usage."""
    app = FakeAppServerClient(
        notifications=[],
        final_turn={
            "id": "turn-no-usage",
            "finalResponse": _envelope_json({"tool_calls": [], "final": "hi"}),
        },
    )
    model = CodexModel(app_server=app)

    reqs = [ModelRequest(parts=[UserPromptPart("hi")])]
    params = ModelRequestParameters(output_mode="text", allow_text_output=True)
    response = await model.request(reqs, None, params)

    assert response.usage == RequestUsage()
    assert response.parts and isinstance(response.parts[0], TextPart)


@pytest.mark.asyncio
async def test_codex_model_request_without_history() -> None:
    """Cover the prompt path where message history is empty."""
    app = FakeAppServerClient(
        notifications=[],
        final_turn={
            "id": "turn-empty-history",
            "usage": {"inputTokens": 1, "outputTokens": 1},
            "finalResponse": _envelope_json({"tool_calls": [], "final": "hello"}),
        },
    )
    model = CodexModel(app_server=app)

    params = ModelRequestParameters(output_mode="text", allow_text_output=True)
    response = await model.request([], None, params)

    assert response.parts and isinstance(response.parts[0], TextPart)


@pytest.mark.asyncio
async def test_codex_model_returns_text_when_allowed():
    output = {"tool_calls": [], "final": "hello"}
    app = FakeAppServerClient(
        notifications=[],
        final_turn={
            "id": "turn-text",
            "usage": {"inputTokens": 1, "outputTokens": 1},
            "finalResponse": _envelope_json(output),
        },
    )
    model = CodexModel(app_server=app)

    messages = [ModelRequest(parts=[UserPromptPart("hi")])]
    params = ModelRequestParameters(output_mode="text", allow_text_output=True)

    response = await model.request(messages, None, params)
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], TextPart)
    assert response.parts[0].content == "hello"


@pytest.mark.asyncio
async def test_codex_model_omits_text_when_not_allowed():
    output = {"tool_calls": [], "final": "hello"}
    app = FakeAppServerClient(
        notifications=[],
        final_turn={
            "id": "turn-no-text",
            "usage": {"inputTokens": 1, "outputTokens": 1},
            "finalResponse": _envelope_json(output),
        },
    )
    model = CodexModel(app_server=app)

    messages = [ModelRequest(parts=[UserPromptPart("hi")])]
    params = ModelRequestParameters(output_mode="tool", allow_text_output=False)

    response = await model.request(messages, None, params)
    assert list(response.parts) == []


def test_model_name_and_system_properties():
    app = FakeAppServerClient(
        notifications=[],
        final_turn={
            "id": "turn-system",
            "usage": {"inputTokens": 0, "outputTokens": 0},
            "finalResponse": _envelope_json({"tool_calls": [], "final": ""}),
        },
    )
    model = CodexModel(app_server=app, system="custom")
    assert model.model_name == "codex"
    assert model.system == "custom"


def test_build_envelope_schema_restricts_tool_names():
    schema = _build_envelope_schema(["a", "b"])
    assert schema["properties"]["tool_calls"]["items"]["properties"]["name"][
        "enum"
    ] == [
        "a",
        "b",
    ]


def test_json_helpers_handle_dataclasses_models_and_bytes():
    @dataclasses.dataclass
    class D:
        x: int

    class M(BaseModel):
        x: int

    assert _jsonable(D(1)) == {"x": 1}
    assert _jsonable(M(x=2)) == {"x": 2}
    assert _jsonable(b"hi") == {"type": "bytes", "base64": "aGk="}


def test_json_dumps_falls_back_to_str_for_unserializable_objects():
    text = _json_dumps(object())
    assert "object" in text


def test_render_tool_definitions_includes_output_tools_and_sequential():
    manifest = _render_tool_definitions(
        function_tools=[
            ToolDefinition(
                name="a",
                description="A",
                parameters_json_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            )
        ],
        output_tools=[
            ToolDefinition(
                name="final",
                description="Final",
                kind="output",
                parameters_json_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            )
        ],
    )
    assert "Function tools:" in manifest
    assert "- a" in manifest
    assert "Output tools" in manifest
    assert "- final" in manifest


def test_render_tool_definitions_includes_metadata_and_timeout():
    manifest = _render_tool_definitions(
        function_tools=[
            ToolDefinition(
                name="metadata_tool",
                description="With metadata",
                parameters_json_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                strict=True,
                outer_typed_dict_key="payload",
                kind="function",
            )
        ],
        output_tools=[],
    )
    assert "strict: true" in manifest
    assert "outer_typed_dict_key: payload" in manifest
    assert "kind: function" in manifest


def test_render_tool_definitions_includes_output_tool_metadata():
    manifest = _render_tool_definitions(
        function_tools=[],
        output_tools=[
            ToolDefinition(
                name="final",
                description="Output",
                parameters_json_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                strict=False,
                outer_typed_dict_key="payload",
                kind="output",
            )
        ],
    )
    assert "strict: false" in manifest
    assert "outer_typed_dict_key: payload" in manifest
    assert "kind: output" in manifest


def test_envelope_extractors_filter_invalid_shapes():
    assert _tool_calls_from_envelope("nope") == []
    assert _tool_calls_from_envelope({"tool_calls": "nope"}) == []
    calls = _tool_calls_from_envelope(
        {
            "tool_calls": [
                {"id": "x", "name": "t", "arguments": "{}"},
                {"id": "", "name": "t", "arguments": "{}"},
                {"id": "y", "name": "", "arguments": "{}"},
                {"id": "z", "name": "t", "arguments": 1},
                "bad",
            ],
            "final": "",
        }
    )
    assert [c.tool_call_id for c in calls] == ["x"]
    assert _final_from_envelope("nope") == ""
    assert _final_from_envelope({"final": 1}) == ""
    assert _final_from_envelope({"final": "ok"}) == "ok"


def test_render_message_history_includes_request_and_response_parts():
    class DummyFilePart:
        part_kind = "file"

        def __init__(self, content: bytes) -> None:
            """
            Initialize the object with raw file content.

            Parameters:
                content (bytes): Raw bytes of the file to store on the instance.
            """
            self.content = content

    history = _render_message_history(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart("system"),
                    UserPromptPart("user"),
                    UserPromptPart([{"type": "text", "text": "hi"}]),
                    ToolReturnPart("t", {"x": 1}, tool_call_id="call_1"),
                    RetryPromptPart("retry"),
                    DummyFilePart(b"abc"),
                ],
                instructions="ins",
            ),
            ModelResponse(
                parts=[
                    TextPart("assistant"),
                    ToolCallPart("t", args={"x": 1}, tool_call_id="call_2"),
                    ThinkingPart("..."),
                    BuiltinToolCallPart("builtin", args=None, tool_call_id="call_3"),
                ]
            ),
        ]
    )
    assert "[instructions]" in history
    assert "[system]" in history
    assert "[user]" in history
    assert "[tool:t id=call_1]" in history
    assert "[retry]" in history
    assert "[assistant]" in history
    assert "[tool-call:t id=call_2]" in history
    assert "[assistant-part:builtin-tool-call]" in history


def test_render_message_history_handles_non_callable_tool_return_and_retry():
    class DummyToolReturn:
        part_kind = "tool-return"

        def __init__(self):
            self.tool_name = "tool"
            self.tool_call_id = "call_1"
            self.content = {"ok": True}

    class DummyRetryPrompt:
        part_kind = "retry-prompt"

        def __init__(self):
            self.content = "try again"

    history = _render_message_history(
        [
            ModelRequest(
                parts=[DummyToolReturn(), DummyRetryPrompt()],
                instructions=None,
            )
        ]
    )
    assert "[tool:tool id=call_1]" in history
    assert '{"ok":true}' in history
    assert "[retry]" in history
    assert "try again" in history


@pytest.mark.asyncio
async def test_codex_model_includes_tool_manifest_and_history_in_prompt():
    output = {"tool_calls": [], "final": "hello"}
    app = FakeAppServerClient(
        notifications=[],
        final_turn={
            "id": "turn-manifest",
            "usage": {"inputTokens": 1, "outputTokens": 1},
            "finalResponse": _envelope_json(output),
        },
    )
    model = CodexModel(app_server=app)

    messages = [
        ModelRequest(
            parts=[
                UserPromptPart("hi"),
                ToolReturnPart("t", {"x": 1}, tool_call_id="call_1"),
            ],
            instructions="ins",
        )
    ]
    params = ModelRequestParameters(
        output_mode="tool",
        allow_text_output=False,
        function_tools=[
            ToolDefinition(
                name="t",
                description="tool",
                parameters_json_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            )
        ],
        output_tools=[
            ToolDefinition(
                name="final",
                description="final",
                parameters_json_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            )
        ],
    )

    await model.request(messages, None, params)
    prompt = app.turn_session_calls[0]["input"]
    assert "Function tools:" in prompt
    assert "Output tools" in prompt
    assert "Conversation so far:" in prompt


@pytest.mark.asyncio
async def test_codex_model_request_stream_yields_response():
    app = FakeAppServerClient(
        notifications=[],
        final_turn={
            "id": "turn-stream-response",
            "usage": {"inputTokens": 1, "cachedInputTokens": 2, "outputTokens": 3},
            "finalResponse": _envelope_json({"tool_calls": [], "final": "hello"}),
        },
        thread_id="thread-123",
    )
    model = CodexModel(app_server=app)

    messages = [ModelRequest(parts=[UserPromptPart("hi")])]
    params = ModelRequestParameters(output_mode="text", allow_text_output=True)

    async with model.request_stream(messages, None, params) as streamed:
        events = [event async for event in streamed]
        response = streamed.get()

    assert any(isinstance(event, PartStartEvent) for event in events)
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], TextPart)
    assert response.parts[0].content == "hello"
    assert response.provider_name == "openai"
    assert response.provider_details == {"thread_id": "thread-123"}


@pytest.mark.asyncio
async def test_codex_model_request_stream_accepts_run_context_argument():
    app = FakeAppServerClient(
        notifications=[],
        final_turn={
            "id": "turn-run-context",
            "usage": {"inputTokens": 1, "outputTokens": 1},
            "finalResponse": _envelope_json({"tool_calls": [], "final": "hello"}),
        },
    )
    model = CodexModel(app_server=app)

    messages = [ModelRequest(parts=[UserPromptPart("hi")])]
    params = ModelRequestParameters(output_mode="text", allow_text_output=True)

    async with model.request_stream(messages, None, params, object()) as streamed:
        _ = [event async for event in streamed]
        response = streamed.get()

    assert len(response.parts) == 1
    assert isinstance(response.parts[0], TextPart)
    assert response.parts[0].content == "hello"


@pytest.mark.asyncio
async def test_agent_run_stream_passes_run_context_to_codex_model():
    class CapturingCodexModel(CodexModel):
        def __init__(self, *, app_server):
            super().__init__(app_server=app_server)
            self.last_run_context = None

        @asynccontextmanager
        async def request_stream(
            self,
            messages,
            model_settings,
            model_request_parameters,
            run_context=None,
        ):
            self.last_run_context = run_context
            async with super().request_stream(
                messages,
                model_settings,
                model_request_parameters,
                run_context,
            ) as streamed:
                yield streamed

    app = FakeAppServerClient(
        notifications=[],
        final_turn={
            "id": "turn-agent-run-context",
            "usage": {"inputTokens": 1, "outputTokens": 1},
            "finalResponse": _envelope_json({"tool_calls": [], "final": "hello"}),
        },
    )
    model = CapturingCodexModel(app_server=app)
    agent = Agent(model=model, output_type=str)

    async with agent.run_stream("hi") as result:
        streamed_output = await result.get_output()

    assert streamed_output == "hello"
    assert model.last_run_context is not None


@pytest.mark.asyncio
async def test_agent_run_stream_events_works_with_codex_model():
    app = FakeAppServerClient(
        notifications=[
            AppServerNotification(
                method="item/updated",
                params={"item": {"id": "m1", "type": "agent_message", "text": "hel"}},
            ),
            AppServerNotification(
                method="item/updated",
                params={"item": {"id": "m1", "type": "agent_message", "text": "hello"}},
            ),
        ],
        final_turn={
            "id": "turn-stream-events",
            "usage": {"inputTokens": 1, "outputTokens": 1},
            "finalResponse": "hello",
        },
    )
    agent = Agent(model=CodexModel(app_server=app), output_type=str)

    events = [event async for event in agent.run_stream_events("hi")]

    assert any(isinstance(event, PartStartEvent) for event in events)
    assert any(isinstance(event, AgentRunResultEvent) for event in events)
    result_event = next(
        event for event in events if isinstance(event, AgentRunResultEvent)
    )
    assert result_event.result.output == "hello"


def test_codex_model_can_construct_codex_from_options():
    CodexModel(codex_options=CodexOptions(codex_path_override="codex-binary"))


def test_codex_model_rejects_legacy_codex_argument():
    with pytest.raises(TypeError):
        CodexModel(**{"codex": object()})


@pytest.mark.asyncio
async def test_streamed_response_emits_tool_calls_and_skips_unknown_parts():
    """Streamed responses should emit events for tool calls and ignore other parts."""
    resp = CodexStreamedResponse(
        model_request_parameters=ModelRequestParameters(
            output_mode="text",
            allow_text_output=True,
        ),
        model_name="m",
        provider_name="openai",
        parts=[
            ToolCallPart("t", args='{"x":1}', tool_call_id="call_1"),
            ThinkingPart("..."),
            TextPart("hello"),
        ],
        thread_id="thread-123",
        usage=RequestUsage(),
    )
    events = [event async for event in resp]
    assert events
    assert resp.provider_url is None


@pytest.mark.asyncio
async def test_streamed_response_supports_iterator_text_delta_api(
    monkeypatch: pytest.MonkeyPatch,
):
    """Accept both old (single-event) and new (iterator) text-delta manager APIs."""
    resp = CodexStreamedResponse(
        model_request_parameters=ModelRequestParameters(
            output_mode="text",
            allow_text_output=True,
        ),
        model_name="m",
        provider_name="openai",
        parts=[TextPart("hello")],
        thread_id="thread-123",
        usage=RequestUsage(),
    )

    original_handle_text_delta = resp._parts_manager.handle_text_delta

    def _iterator_handle_text_delta(*, vendor_part_id, content):
        result = original_handle_text_delta(
            vendor_part_id=vendor_part_id,
            content=content,
        )
        if hasattr(result, "event_kind"):
            return iter([result])
        return result

    monkeypatch.setattr(
        resp._parts_manager, "handle_text_delta", _iterator_handle_text_delta
    )

    events = [event async for event in resp]

    assert any(isinstance(event, PartStartEvent) for event in events)
    model_response = resp.get()
    assert len(model_response.parts) == 1
    assert isinstance(model_response.parts[0], TextPart)
    assert model_response.parts[0].content == "hello"


class FakeAppServerTurnSession:
    def __init__(self, notifications, final_turn):
        self._notifications = list(notifications)
        self._final_turn = dict(final_turn)

    async def notifications(self):
        for notification in self._notifications:
            yield notification

    async def wait(self):
        return self._final_turn


class FakeAppServerClient:
    def __init__(self, *, notifications, final_turn, thread_id="thread-app"):
        self._notifications = list(notifications)
        self._final_turn = dict(final_turn)
        self._thread_id = thread_id
        self.start_calls = 0
        self.close_calls = 0
        self.thread_start_calls = []
        self.turn_session_calls = []

    async def start(self):
        self.start_calls += 1

    async def close(self):
        self.close_calls += 1

    async def thread_start(self, **params):
        self.thread_start_calls.append(dict(params))
        return {"thread": {"id": self._thread_id}}

    async def turn_session(self, thread_id, input, **params):
        self.turn_session_calls.append(
            {"thread_id": thread_id, "input": input, "params": dict(params)}
        )
        return FakeAppServerTurnSession(self._notifications, self._final_turn)


@pytest.mark.asyncio
async def test_codex_model_default_path_uses_app_server_and_returns_tool_calls():
    app = FakeAppServerClient(
        notifications=[],
        final_turn={
            "id": "turn-1",
            "usage": {"inputTokens": 4, "cachedInputTokens": 1, "outputTokens": 2},
            "finalResponse": (
                '{"tool_calls":[{"id":"call_1","name":"add","arguments":"{\\"a\\":1,\\"b\\":2}"}],'
                '"final":""}'
            ),
        },
    )
    model = CodexModel(app_server=app)

    response = await model.request(
        [ModelRequest(parts=[UserPromptPart("calculate")])],
        None,
        ModelRequestParameters(
            output_mode="tool",
            allow_text_output=False,
            function_tools=[
                ToolDefinition(
                    name="add",
                    description="add two numbers",
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                        "additionalProperties": False,
                    },
                )
            ],
        ),
    )

    assert app.start_calls >= 1
    assert len(app.thread_start_calls) == 1
    assert len(app.turn_session_calls) == 1
    assert response.provider_details == {"thread_id": "thread-app"}
    assert _usage_input_tokens(response.usage) == 4
    assert _usage_cached_input_tokens(response.usage) == 1
    assert _usage_output_tokens(response.usage) == 2
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], ToolCallPart)
    assert response.parts[0].tool_name == "add"
    assert response.parts[0].tool_call_id == "call_1"
    assert response.parts[0].args == '{"a":1,"b":2}'


@pytest.mark.asyncio
async def test_codex_model_default_stream_emits_incremental_text_deltas():
    app = FakeAppServerClient(
        notifications=[
            AppServerNotification(
                method="item/updated",
                params={"item": {"id": "m1", "type": "agent_message", "text": "hel"}},
            ),
            AppServerNotification(
                method="item/updated",
                params={"item": {"id": "m1", "type": "agent_message", "text": "hello"}},
            ),
        ],
        final_turn={
            "id": "turn-2",
            "usage": {"inputTokens": 3, "outputTokens": 2},
            "finalResponse": "hello",
        },
    )
    model = CodexModel(app_server=app)

    params = ModelRequestParameters(output_mode="text", allow_text_output=True)
    async with model.request_stream(
        [ModelRequest(parts=[UserPromptPart("say hello")])], None, params
    ) as streamed:
        events = [event async for event in streamed]
        response = streamed.get()

    assert app.start_calls >= 1
    assert len(app.thread_start_calls) == 1
    assert len(app.turn_session_calls) == 1
    assert any(isinstance(event, PartStartEvent) for event in events)
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], TextPart)
    assert response.parts[0].content == "hello"


@pytest.mark.asyncio
async def test_codex_model_default_stream_extracts_text_from_envelope_updates():
    app = FakeAppServerClient(
        notifications=[
            AppServerNotification(
                method="item/updated",
                params={
                    "item": {
                        "id": "m1",
                        "type": "agent_message",
                        "text": _envelope_json({"tool_calls": [], "final": "hel"}),
                    }
                },
            ),
            AppServerNotification(
                method="item/updated",
                params={
                    "item": {
                        "id": "m1",
                        "type": "agent_message",
                        "text": _envelope_json({"tool_calls": [], "final": "hello"}),
                    }
                },
            ),
        ],
        final_turn={
            "id": "turn-envelope-stream",
            "usage": {"inputTokens": 3, "outputTokens": 2},
            "finalResponse": _envelope_json({"tool_calls": [], "final": "hello"}),
        },
    )
    model = CodexModel(app_server=app)

    params = ModelRequestParameters(output_mode="text", allow_text_output=True)
    async with model.request_stream(
        [ModelRequest(parts=[UserPromptPart("say hello")])], None, params
    ) as streamed:
        _ = [event async for event in streamed]
        response = streamed.get()

    assert len(response.parts) == 1
    assert isinstance(response.parts[0], TextPart)
    assert response.parts[0].content == "hello"


def test_handle_notification_streams_tool_calls_from_envelope_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FakeAppServerClient(
        notifications=[],
        final_turn={
            "id": "turn-tool-update",
            "usage": {"inputTokens": 0, "outputTokens": 0},
            "finalResponse": "",
        },
    )
    model = CodexModel(app_server=app)
    streamed = CodexStreamedResponse(
        model_request_parameters=ModelRequestParameters(
            output_mode="tool", allow_text_output=False
        ),
        model_name="codex",
        provider_name="openai",
    )
    state = _TurnAccumulationState()
    tool_calls = []

    def capture_tool_call(**kwargs):
        tool_calls.append(kwargs)

    monkeypatch.setattr(streamed, "push_tool_call", capture_tool_call)

    model._handle_notification(
        notification=AppServerNotification(
            method="item/updated",
            params={
                "item": {
                    "id": "m1",
                    "type": "agent_message",
                    "text": _envelope_json(
                        {
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "name": "add",
                                    "arguments": '{"a":1,"b":2}',
                                }
                            ],
                            "final": "",
                        }
                    ),
                }
            },
        ),
        state=state,
        streamed=streamed,
        allow_stream_text=False,
        stream_raw_text=False,
    )

    assert tool_calls == [
        {
            "vendor_part_id": 0,
            "tool_name": "add",
            "args": '{"a":1,"b":2}',
            "tool_call_id": "call_1",
        }
    ]


@pytest.mark.asyncio
async def test_codex_model_stream_reconciles_with_most_recent_item_update():
    app = FakeAppServerClient(
        notifications=[
            AppServerNotification(
                method="item/updated",
                params={"item": {"id": "m1", "type": "agent_message", "text": "a"}},
            ),
            AppServerNotification(
                method="item/updated",
                params={"item": {"id": "m2", "type": "agent_message", "text": "b"}},
            ),
            AppServerNotification(
                method="item/updated",
                params={"item": {"id": "m1", "type": "agent_message", "text": "abc"}},
            ),
        ],
        final_turn={
            "id": "turn-reconcile",
            "usage": {"inputTokens": 3, "outputTokens": 2},
            "finalResponse": "abcd",
        },
    )
    model = CodexModel(app_server=app)

    params = ModelRequestParameters(output_mode="text", allow_text_output=True)
    async with model.request_stream(
        [ModelRequest(parts=[UserPromptPart("say hello")])], None, params
    ) as streamed:
        _ = [event async for event in streamed]
        response = streamed.get()

    assert [part.content for part in response.parts if isinstance(part, TextPart)] == [
        "abcd"
    ]


@pytest.mark.asyncio
async def test_codex_model_default_run_reuse_mode_resets_on_equal_history_length():
    app = FakeAppServerClient(
        notifications=[],
        final_turn={
            "id": "turn-run",
            "usage": {"inputTokens": 1, "outputTokens": 1},
            "finalResponse": "ok",
        },
    )
    model = CodexModel(app_server=app)
    params = ModelRequestParameters(output_mode="text", allow_text_output=True)

    await model.request([ModelRequest(parts=[UserPromptPart("alpha")])], None, params)
    await model.request([ModelRequest(parts=[UserPromptPart("beta")])], None, params)

    assert len(app.thread_start_calls) == 2
    assert len(app.turn_session_calls) == 2


@pytest.mark.asyncio
async def test_codex_model_always_reuse_mode_reuses_on_equal_history_length():
    app = FakeAppServerClient(
        notifications=[],
        final_turn={
            "id": "turn-always",
            "usage": {"inputTokens": 1, "outputTokens": 1},
            "finalResponse": "ok",
        },
    )
    model = CodexModel(app_server=app, thread_reuse_mode="always")
    params = ModelRequestParameters(output_mode="text", allow_text_output=True)

    await model.request([ModelRequest(parts=[UserPromptPart("alpha")])], None, params)
    await model.request([ModelRequest(parts=[UserPromptPart("beta")])], None, params)

    assert len(app.thread_start_calls) == 1
    assert len(app.turn_session_calls) == 2


def test_app_server_helpers_handle_edge_cases():
    assert _extract_json_object('{"tool_calls":[],"final":"ok"}') == {
        "tool_calls": [],
        "final": "ok",
    }
    assert _extract_json_object('prefix {"tool_calls":[],"final":"ok"} suffix') == {
        "tool_calls": [],
        "final": "ok",
    }
    assert _extract_json_object("not-json") is None

    assert _is_envelope_candidate({"tool_calls": []})
    assert _is_envelope_candidate({"final": ""})
    assert not _is_envelope_candidate([])

    assert _to_int(None) == 0
    assert _to_int("3") == 3
    assert _to_int("not-an-int") == 0

    usage = _usage_from_mapping({"input_tokens": "2", "outputTokens": 4})
    assert usage is not None
    assert _usage_input_tokens(usage) == 2
    assert _usage_output_tokens(usage) == 4
    assert _usage_cached_input_tokens(usage) == 0

    usage_from_turn = _extract_usage_from_turn({"tokenUsage": {"inputTokens": 1}})
    assert usage_from_turn is not None
    assert _usage_input_tokens(usage_from_turn) == 1

    assert _extract_turn_text({"output": {"final": "done"}}) == "done"
    assert (
        _extract_turn_text(
            {"items": [{"type": "agentMessage", "content": "from-agent-message"}]}
        )
        == "from-agent-message"
    )
    assert _extract_turn_text(None) == ""

    notification = AppServerNotification(
        method="item/updated",
        params={
            "item": {"id": "a", "type": "agent_message", "text": "x"},
            "items": [{"id": "b", "type": "agent_message", "text": "y"}],
            "turn": {
                "item": {"id": "c", "type": "agent_message", "text": "z"},
                "items": [{"id": "d", "type": "agent_message", "text": "w"}],
            },
        },
    )
    assert [item["id"] for item in _notification_items(notification)] == [
        "a",
        "b",
        "c",
        "d",
    ]

    failure = AppServerNotification(
        method="turn/failed",
        params={"error": {"message": "boom"}},
    )
    assert _extract_turn_failure_message(failure) == "boom"

    fallback_failure = AppServerNotification(method="turn/failed", params={})
    assert _extract_turn_failure_message(fallback_failure) == "Turn failed"


@pytest.mark.asyncio
async def test_codex_model_default_stream_falls_back_to_final_turn_text():
    app = FakeAppServerClient(
        notifications=[],
        final_turn={
            "id": "turn-fallback",
            "usage": {"inputTokens": 1, "outputTokens": 1},
            "finalResponse": "hello-from-final-turn",
        },
    )
    model = CodexModel(app_server=app)
    params = ModelRequestParameters(output_mode="text", allow_text_output=True)

    async with model.request_stream(
        [ModelRequest(parts=[UserPromptPart("say hello")])], None, params
    ) as streamed:
        events = [event async for event in streamed]
        response = streamed.get()

    assert any(isinstance(event, PartStartEvent) for event in events)
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], TextPart)
    assert response.parts[0].content == "hello-from-final-turn"


@pytest.mark.asyncio
async def test_codex_model_stream_raises_turn_failed_error():
    app = FakeAppServerClient(
        notifications=[
            AppServerNotification(
                method="turn/failed",
                params={"error": {"message": "approval denied"}},
            )
        ],
        final_turn={
            "id": "turn-failed",
            "usage": {"inputTokens": 1, "outputTokens": 0},
            "finalResponse": "",
        },
    )
    model = CodexModel(app_server=app)
    params = ModelRequestParameters(output_mode="text", allow_text_output=True)

    with pytest.raises(TurnFailedError, match="approval denied"):
        async with model.request_stream(
            [ModelRequest(parts=[UserPromptPart("fail")])], None, params
        ) as streamed:
            _ = [event async for event in streamed]


@pytest.mark.asyncio
async def test_codex_model_close_closes_owned_app_server(monkeypatch):
    created = []

    class OwnedFakeAppServer(FakeAppServerClient):
        def __init__(self, options):
            super().__init__(
                notifications=[],
                final_turn={
                    "id": "turn-owned",
                    "usage": {"inputTokens": 1, "outputTokens": 1},
                    "finalResponse": "ok",
                },
            )
            self.options = options
            created.append(self)

    module = importlib.import_module("codex_sdk.integrations.pydantic_ai_model")
    monkeypatch.setattr(module, "AppServerClient", OwnedFakeAppServer)

    model = CodexModel(app_server_options=AppServerOptions())
    params = ModelRequestParameters(output_mode="text", allow_text_output=True)
    await model.request([ModelRequest(parts=[UserPromptPart("hi")])], None, params)

    assert created and created[0].start_calls >= 1

    await model.close()
    assert created[0].close_calls == 1

    with pytest.raises(CodexError, match="closed"):
        await model.request(
            [ModelRequest(parts=[UserPromptPart("again")])], None, params
        )


@pytest.mark.asyncio
async def test_codex_model_close_does_not_close_external_app_server():
    app = FakeAppServerClient(
        notifications=[],
        final_turn={
            "id": "turn-external",
            "usage": {"inputTokens": 1, "outputTokens": 1},
            "finalResponse": "ok",
        },
    )
    model = CodexModel(app_server=app)
    params = ModelRequestParameters(output_mode="text", allow_text_output=True)

    await model.request([ModelRequest(parts=[UserPromptPart("hi")])], None, params)
    assert app.start_calls >= 1

    await model.close()
    assert app.close_calls == 0

    with pytest.raises(CodexError, match="closed"):
        await model.request(
            [ModelRequest(parts=[UserPromptPart("again")])], None, params
        )


def test_codex_model_rejects_invalid_profile_values():
    with pytest.raises(CodexError, match="performance_profile"):
        CodexModel(performance_profile="fast")

    with pytest.raises(CodexError, match="thread_reuse_mode"):
        CodexModel(thread_reuse_mode="sometimes")


@pytest.mark.asyncio
async def test_codex_model_thread_start_params_include_extended_options():
    thread_options = ThreadOptions(
        model="gpt-5",
        sandbox_mode="workspace-write",
        working_directory="/tmp",
        skip_git_repo_check=False,
        model_instructions_file="/tmp/instructions.md",
        model_personality="friendly",
        max_threads=3,
        network_access_enabled=True,
        web_search_mode="live",
        web_search_enabled=True,
        web_search_cached_enabled=True,
        shell_snapshot_enabled=True,
        background_terminals_enabled=True,
        apply_patch_freeform_enabled=True,
        exec_policy_enabled=True,
        remote_models_enabled=True,
        collaboration_modes_enabled=True,
        connectors_enabled=True,
        responses_websockets_enabled=True,
        request_compression_enabled=True,
        feature_overrides={"experimental": True},
        approval_policy="on-request",
        additional_directories=["/tmp/a", "/tmp/b"],
        config_overrides={"feature": "on"},
    )
    app = FakeAppServerClient(
        notifications=[],
        final_turn={
            "id": "turn-opts",
            "usage": {"inputTokens": 1, "outputTokens": 1},
            "finalResponse": "ok",
        },
    )
    model = CodexModel(
        app_server=app,
        thread_options=thread_options,
        performance_profile="max",
    )
    params = ModelRequestParameters(output_mode="text", allow_text_output=True)

    await model.request([ModelRequest(parts=[UserPromptPart("opts")])], None, params)

    sent = app.thread_start_calls[0]
    assert sent["model"] == "gpt-5"
    assert sent["cwd"] == "/tmp"
    assert sent["sandbox_mode"] == "workspace-write"
    assert sent["skip_git_repo_check"] is False
    assert sent["model_instructions_file"] == "/tmp/instructions.md"
    assert sent["model_personality"] == "friendly"
    assert sent["max_threads"] == 3
    assert sent["network_access_enabled"] is True
    assert sent["web_search_mode"] == "live"
    assert sent["web_search_enabled"] is True
    assert sent["web_search_cached_enabled"] is True
    assert sent["shell_snapshot_enabled"] is True
    assert sent["background_terminals_enabled"] is True
    assert sent["apply_patch_freeform_enabled"] is True
    assert sent["exec_policy_enabled"] is True
    assert sent["remote_models_enabled"] is True
    assert sent["collaboration_modes_enabled"] is True
    assert sent["connectors_enabled"] is True
    assert sent["responses_websockets_enabled"] is True
    assert sent["request_compression_enabled"] is True
    assert sent["feature_overrides"] == {"experimental": True}
    assert sent["approval_policy"] == "on-request"
    assert sent["additional_directories"] == ["/tmp/a", "/tmp/b"]
    assert sent["config_overrides"] == {"feature": "on"}


@pytest.mark.asyncio
async def test_codex_model_raises_when_thread_start_missing_id():
    class MissingIdAppServer(FakeAppServerClient):
        async def thread_start(self, **params):
            self.thread_start_calls.append(dict(params))
            return {"thread": {}}

    app = MissingIdAppServer(
        notifications=[],
        final_turn={"id": "turn", "usage": {"inputTokens": 1, "outputTokens": 1}},
    )
    model = CodexModel(app_server=app)
    params = ModelRequestParameters(output_mode="text", allow_text_output=True)

    with pytest.raises(CodexError, match="thread/start response missing thread id"):
        await model.request(
            [ModelRequest(parts=[UserPromptPart("oops")])], None, params
        )
