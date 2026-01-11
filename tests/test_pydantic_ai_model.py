import dataclasses

import pytest

pydantic = pytest.importorskip("pydantic")
pytest.importorskip("pydantic_ai")
BaseModel = pydantic.BaseModel

from pydantic_ai.messages import (  # type: ignore[import-not-found]
    BuiltinToolCallPart,
    FilePart,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters  # type: ignore[import-not-found]
from pydantic_ai.tools import ToolDefinition  # type: ignore[import-not-found]

from codex_sdk.events import Usage
from codex_sdk.integrations.pydantic_ai_model import (
    CodexModel,
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

    # Schema should restrict tool names
    enum = thread.last_schema["properties"]["tool_calls"]["items"]["properties"][
        "name"
    ]["enum"]
    assert enum == ["add"]


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
                sequential=True,
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
                sequential=True,
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
    assert "sequential: true" in manifest
    assert "Output tools" in manifest
    assert "- final" in manifest


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
    history = _render_message_history(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart("system"),
                    UserPromptPart("user"),
                    UserPromptPart([{"type": "text", "text": "hi"}]),
                    ToolReturnPart("t", {"x": 1}, tool_call_id="call_1"),
                    RetryPromptPart("retry"),
                    FilePart(b"abc"),
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


def test_codex_model_can_construct_codex_from_options():
    CodexModel(codex_options=CodexOptions(codex_path_override="codex-binary"))
