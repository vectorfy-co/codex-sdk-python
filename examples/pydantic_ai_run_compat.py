"""Minimal harness for non-stream Agent.run parity with CodexModel."""

import asyncio
import json
import re

from pydantic_ai import Agent

from codex_sdk.events import Usage
from codex_sdk.integrations.pydantic_ai_model import CodexModel
from codex_sdk.thread import ParsedTurn, Turn


class _FakeThread:
    def __init__(self) -> None:
        self.id = "thread-compat-run"

    async def run_json(self, prompt, *, output_schema, turn_options=None):
        tool_names = (
            output_schema.get("properties", {})
            .get("tool_calls", {})
            .get("items", {})
            .get("properties", {})
            .get("name", {})
            .get("enum", [])
        )
        output = {"tool_calls": [], "final": "run ok"}
        if tool_names:
            first_tool = tool_names[0]
            args_json = _build_args_json(prompt, first_tool)
            output = {
                "tool_calls": [
                    {"id": "call_1", "name": first_tool, "arguments": args_json}
                ],
                "final": "",
            }
        turn = Turn(
            items=[],
            final_response="",
            usage=Usage(input_tokens=1, cached_input_tokens=0, output_tokens=1),
        )
        return ParsedTurn(turn=turn, output=output)


class _FakeCodex:
    def __init__(self) -> None:
        self._thread = _FakeThread()

    def start_thread(self, options=None):
        return self._thread


def _build_args_json(prompt: str, tool_name: str) -> str:
    pattern = re.compile(
        rf"- {re.escape(tool_name)}\\n(?:  .*\\n)*?  parameters_json_schema: (.+)"
    )
    match = pattern.search(prompt)
    if not match:
        return "{}"
    try:
        schema = json.loads(match.group(1))
    except json.JSONDecodeError:
        return "{}"
    value = _example_value(schema)
    if isinstance(value, dict):
        return json.dumps(value, separators=(",", ":"))
    return "{}"


def _example_value(schema):
    schema_type = schema.get("type")
    if schema_type == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        result = {}
        keys = required or list(properties.keys())[:1]
        for key in keys:
            child_schema = properties.get(key, {"type": "string"})
            result[key] = _example_value(child_schema)
        return result
    if schema_type == "array":
        return []
    if schema_type == "integer":
        return 1
    if schema_type == "number":
        return 1.0
    if schema_type == "boolean":
        return True
    return "ok"


async def _main() -> None:
    agent = Agent(model=CodexModel(codex=_FakeCodex()), output_type=str)
    result = await agent.run("ping")
    assert result.output == "run ok"
    print("run compatible:", result.output)


if __name__ == "__main__":
    asyncio.run(_main())
