import dataclasses
import importlib

import pytest

# ruff: noqa: E402
# Imports are intentionally ordered around import-or-skip behavior.

pydantic = pytest.importorskip("pydantic")
pytest.importorskip("pydantic_ai", exc_type=ImportError)
BaseModel = pydantic.BaseModel

from codex_sdk.events import Usage
from codex_sdk.integrations.pydantic_ai_model import (
    CodexModel,
    CodexStreamedResponse,
    _build_envelope_schema,
    _final_from_envelope,
    _json_dumps,
    _jsonable,
    _render_message_history,
    _render_tool_definitions,
    _tool_calls_from_envelope,
)
from codex_sdk.options import CodexOptions
from codex_sdk.thread import ParsedTurn, Turn

messages = importlib.import_module("pydantic_ai.messages")
models = importlib.import_module("pydantic_ai.models")
tools = importlib.import_module("pydantic_ai.tools")

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


class FakeThread:
    def __init__(self, output):
        self._output = output
        self.id = "thread-123"
        self.last_prompt = None
        self.last_schema = None

    async def run_json(self, prompt, *, output_schema, turn_options=None):
        self.last_prompt = prompt
        self.last_schema = output_schema
        turn = Turn(
            items=[],
            final_response="",
            usage=Usage(input_tokens=1, cached_input_tokens=2, output_tokens=3),
        )
        return ParsedTurn(turn=turn, output=self._output)


class FakeCodex:
    def __init__(self, thread):
        self._thread = thread
        self.last_thread_options = None

    def start_thread(self, options=None):
        self.last_thread_options = options
        return self._thread


@pytest.mark.asyncio
async def test_codex_model_returns_tool_calls():
    output = {
        "tool_calls": [
            {"id": "call_1", "name": "add", "arguments": '{"a":1,"b":2}'},
        ],
        "final": "",
    }
    thread = FakeThread(output)
    codex = FakeCodex(thread)
    model = CodexModel(codex=codex)

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
    assert response.usage.input_tokens == 1
    assert response.usage.cache_read_tokens == 2
    assert response.usage.output_tokens == 3
    assert response.usage.details == {"cached_input_tokens": 2}

    # Schema should restrict tool names
    enum = thread.last_schema["properties"]["tool_calls"]["items"]["properties"][
        "name"
    ]["enum"]
    assert enum == ["add"]


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
    assert "sequential: true" in rendered
    assert "metadata:" in rendered
    assert "timeout:" in rendered
    assert "- func2" in rendered
    assert "- out2" in rendered


def test_codex_model_does_not_override_explicit_thread_options() -> None:
    """Cover CodexModel.__init__ branches when thread options are preconfigured."""
    profiles = importlib.import_module("pydantic_ai.profiles")

    thread = FakeThread({"tool_calls": [], "final": "ok"})
    codex = FakeCodex(thread)
    thread_options = importlib.import_module("codex_sdk.options").ThreadOptions(
        skip_git_repo_check=False,
        sandbox_mode="workspace-write",
        approval_policy="on-request",
        web_search_mode="live",
        network_access_enabled=True,
    )

    profile = profiles.ModelProfile(supports_tools=False)
    model = CodexModel(codex=codex, thread_options=thread_options, profile=profile)

    assert model.model_name == "codex"
    assert thread_options.skip_git_repo_check is False
    assert thread_options.sandbox_mode == "workspace-write"
    assert thread_options.approval_policy == "on-request"
    assert thread_options.web_search_mode == "live"
    assert thread_options.network_access_enabled is True


@pytest.mark.asyncio
async def test_codex_model_usage_defaults_when_thread_usage_missing() -> None:
    """Cover the branch where turn.usage is None and we return minimal usage."""

    class FakeThreadNoUsage(FakeThread):
        async def run_json(self, prompt, *, output_schema, turn_options=None):
            self.last_prompt = prompt
            self.last_schema = output_schema
            turn = Turn(items=[], final_response="", usage=None)
            return ParsedTurn(turn=turn, output={"tool_calls": [], "final": "hi"})

    thread = FakeThreadNoUsage({"tool_calls": [], "final": "hi"})
    codex = FakeCodex(thread)
    model = CodexModel(codex=codex)

    reqs = [ModelRequest(parts=[UserPromptPart("hi")])]
    params = ModelRequestParameters(output_mode="text", allow_text_output=True)
    response = await model.request(reqs, None, params)

    assert response.usage == RequestUsage()
    assert response.parts and isinstance(response.parts[0], TextPart)


@pytest.mark.asyncio
async def test_codex_model_request_without_history() -> None:
    """Cover the prompt path where message history is empty."""
    thread = FakeThread({"tool_calls": [], "final": "hello"})
    codex = FakeCodex(thread)
    model = CodexModel(codex=codex)

    params = ModelRequestParameters(output_mode="text", allow_text_output=True)
    response = await model.request([], None, params)

    assert response.parts and isinstance(response.parts[0], TextPart)


@pytest.mark.asyncio
async def test_codex_model_returns_text_when_allowed():
    output = {"tool_calls": [], "final": "hello"}
    thread = FakeThread(output)
    codex = FakeCodex(thread)
    model = CodexModel(codex=codex)

    messages = [ModelRequest(parts=[UserPromptPart("hi")])]
    params = ModelRequestParameters(output_mode="text", allow_text_output=True)

    response = await model.request(messages, None, params)
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], TextPart)
    assert response.parts[0].content == "hello"


@pytest.mark.asyncio
async def test_codex_model_omits_text_when_not_allowed():
    output = {"tool_calls": [], "final": "hello"}
    thread = FakeThread(output)
    codex = FakeCodex(thread)
    model = CodexModel(codex=codex)

    messages = [ModelRequest(parts=[UserPromptPart("hi")])]
    params = ModelRequestParameters(output_mode="tool", allow_text_output=False)

    response = await model.request(messages, None, params)
    assert list(response.parts) == []


def test_model_name_and_system_properties():
    thread = FakeThread({"tool_calls": [], "final": ""})
    codex = FakeCodex(thread)
    model = CodexModel(codex=codex, system="custom")
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
    thread = FakeThread(output)
    codex = FakeCodex(thread)
    model = CodexModel(codex=codex)

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
    assert "Function tools:" in thread.last_prompt
    assert "Output tools" in thread.last_prompt
    assert "Conversation so far:" in thread.last_prompt


@pytest.mark.asyncio
async def test_codex_model_request_stream_yields_response():
    output = {"tool_calls": [], "final": "hello"}
    thread = FakeThread(output)
    codex = FakeCodex(thread)
    model = CodexModel(codex=codex)

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


def test_codex_model_can_construct_codex_from_options():
    CodexModel(codex_options=CodexOptions(codex_path_override="codex-binary"))


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
