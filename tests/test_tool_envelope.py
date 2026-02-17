from __future__ import annotations

import pytest

from codex_sdk.tool_envelope import (
    ToolCallEnvelope,
    ToolPlan,
    ToolPlanValidationError,
    _matches_type,
    _normalize_tool_choice,
    _require_type,
    _validate_any_of,
    _validate_array_schema,
    _validate_json_schema,
    _validate_object_schema,
    build_envelope_schema,
    parse_tool_plan,
    validate_tool_plan,
)


def test_build_envelope_schema_restricts_tool_names() -> None:
    schema = build_envelope_schema(["read", "grep"])
    enum = schema["properties"]["tool_calls"]["items"]["properties"]["name"]["enum"]
    assert enum == ["read", "grep"]


def test_parse_tool_plan_tool_calls() -> None:
    plan = parse_tool_plan(
        {
            "tool_calls": [
                {"id": "call_1", "name": "read", "arguments": '{"path":"README.md"}'}
            ],
            "final": "",
        }
    )

    assert plan.kind == "tool_calls"
    assert plan.calls[0].tool_call_id == "call_1"
    assert plan.calls[0].tool_name == "read"


def test_parse_tool_plan_final() -> None:
    plan = parse_tool_plan({"tool_calls": [], "final": "done"})
    assert plan == ToolPlan(kind="final", content="done")


def test_parse_tool_plan_rejects_mixed_output() -> None:
    with pytest.raises(ToolPlanValidationError) as exc:
        parse_tool_plan(
            {
                "tool_calls": [{"id": "1", "name": "x", "arguments": "{}"}],
                "final": "text",
            }
        )

    assert exc.value.code == "invalid_envelope"


def test_validate_tool_plan_normalizes_args_and_ids() -> None:
    plan = ToolPlan(
        kind="tool_calls",
        calls=(
            ToolCallEnvelope(
                tool_call_id="",
                tool_name="read",
                arguments_json='{"path":"README.md","max_lines":10}',
            ),
        ),
    )

    validated = validate_tool_plan(
        plan,
        tool_schemas={
            "read": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_lines": {"type": "integer"},
                },
                "required": ["path"],
                "additionalProperties": False,
            }
        },
        fallback_id_prefix="call_chatcmpl_abc",
    )

    assert validated.kind == "tool_calls"
    assert validated.calls[0].tool_call_id == "call_chatcmpl_abc_0"
    assert validated.calls[0].arguments_json == '{"max_lines":10,"path":"README.md"}'


def test_validate_tool_plan_enforces_tool_choice_none() -> None:
    plan = ToolPlan(
        kind="tool_calls",
        calls=(
            ToolCallEnvelope(tool_call_id="1", tool_name="read", arguments_json="{}"),
        ),
    )

    with pytest.raises(ToolPlanValidationError) as exc:
        validate_tool_plan(
            plan, tool_schemas={"read": {"type": "object"}}, tool_choice="none"
        )

    assert exc.value.code == "tool_choice_mismatch"


def test_validate_tool_plan_enforces_required_choice() -> None:
    with pytest.raises(ToolPlanValidationError) as exc:
        validate_tool_plan(
            ToolPlan(kind="final", content="done"),
            tool_schemas={"read": {"type": "object"}},
            tool_choice="required",
        )

    assert exc.value.code == "tool_choice_mismatch"


def test_validate_tool_plan_enforces_parallel_flag() -> None:
    plan = ToolPlan(
        kind="tool_calls",
        calls=(
            ToolCallEnvelope(tool_call_id="1", tool_name="read", arguments_json="{}"),
            ToolCallEnvelope(tool_call_id="2", tool_name="read", arguments_json="{}"),
        ),
    )

    with pytest.raises(ToolPlanValidationError) as exc:
        validate_tool_plan(
            plan,
            tool_schemas={"read": {"type": "object"}},
            parallel_tool_calls=False,
        )

    assert exc.value.code == "parallel_tool_calls_disabled"


def test_validate_tool_plan_rejects_bad_arguments_schema() -> None:
    plan = ToolPlan(
        kind="tool_calls",
        calls=(
            ToolCallEnvelope(
                tool_call_id="1",
                tool_name="read",
                arguments_json='{"path":123}',
            ),
        ),
    )

    with pytest.raises(ToolPlanValidationError) as exc:
        validate_tool_plan(
            plan,
            tool_schemas={
                "read": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                }
            },
        )

    assert exc.value.code == "invalid_tool_arguments"


def test_validate_tool_plan_rejects_non_object_arguments() -> None:
    plan = ToolPlan(
        kind="tool_calls",
        calls=(
            ToolCallEnvelope(tool_call_id="1", tool_name="read", arguments_json="[]"),
        ),
    )

    with pytest.raises(ToolPlanValidationError) as exc:
        validate_tool_plan(
            plan,
            tool_schemas={"read": {"type": "object"}},
        )

    assert exc.value.code == "invalid_tool_arguments"


@pytest.mark.parametrize(
    "payload",
    [
        "bad",
        {"tool_calls": []},
        {"tool_calls": "bad", "final": ""},
        {"tool_calls": [], "final": 1},
        {"tool_calls": [1], "final": ""},
        {"tool_calls": [{"id": 1, "name": "x", "arguments": "{}"}], "final": ""},
        {"tool_calls": [{"id": "1", "name": 3, "arguments": "{}"}], "final": ""},
        {"tool_calls": [{"id": "1", "name": "x", "arguments": 3}], "final": ""},
    ],
)
def test_parse_tool_plan_rejects_invalid_shapes(payload: object) -> None:
    with pytest.raises(ToolPlanValidationError) as exc:
        parse_tool_plan(payload)
    assert exc.value.code == "invalid_envelope"


def test_validate_tool_plan_final_passthrough_for_auto_choice() -> None:
    plan = ToolPlan(kind="final", content="done")
    assert validate_tool_plan(plan, tool_schemas={"read": {"type": "object"}}) == plan


def test_validate_tool_plan_rejects_empty_tool_call_list() -> None:
    with pytest.raises(ToolPlanValidationError) as exc:
        validate_tool_plan(ToolPlan(kind="tool_calls", calls=()), tool_schemas={})
    assert exc.value.code == "invalid_envelope"


def test_validate_tool_plan_enforces_max_tool_calls() -> None:
    plan = ToolPlan(
        kind="tool_calls",
        calls=(
            ToolCallEnvelope(tool_call_id="1", tool_name="read", arguments_json="{}"),
            ToolCallEnvelope(tool_call_id="2", tool_name="read", arguments_json="{}"),
        ),
    )
    with pytest.raises(ToolPlanValidationError) as exc:
        validate_tool_plan(
            plan, tool_schemas={"read": {"type": "object"}}, max_tool_calls=1
        )
    assert exc.value.code == "too_many_tool_calls"


def test_validate_tool_plan_rejects_empty_tool_name() -> None:
    plan = ToolPlan(
        kind="tool_calls",
        calls=(
            ToolCallEnvelope(tool_call_id="1", tool_name="  ", arguments_json="{}"),
        ),
    )
    with pytest.raises(ToolPlanValidationError) as exc:
        validate_tool_plan(plan, tool_schemas={"read": {"type": "object"}})
    assert exc.value.code == "unknown_tool"


def test_validate_tool_plan_rejects_function_choice_mismatch() -> None:
    plan = ToolPlan(
        kind="tool_calls",
        calls=(
            ToolCallEnvelope(tool_call_id="1", tool_name="read", arguments_json="{}"),
        ),
    )
    with pytest.raises(ToolPlanValidationError) as exc:
        validate_tool_plan(
            plan,
            tool_schemas={"read": {"type": "object"}},
            tool_choice={"type": "function", "function": {"name": "write"}},
        )
    assert exc.value.code == "tool_choice_mismatch"


def test_validate_tool_plan_rejects_unknown_tool() -> None:
    plan = ToolPlan(
        kind="tool_calls",
        calls=(
            ToolCallEnvelope(
                tool_call_id="1",
                tool_name="unknown",
                arguments_json="{}",
            ),
        ),
    )
    with pytest.raises(ToolPlanValidationError) as exc:
        validate_tool_plan(plan, tool_schemas={"read": {"type": "object"}})
    assert exc.value.code == "unknown_tool"


def test_validate_tool_plan_rejects_invalid_json_arguments() -> None:
    plan = ToolPlan(
        kind="tool_calls",
        calls=(
            ToolCallEnvelope(
                tool_call_id="1",
                tool_name="read",
                arguments_json='{"oops":',
            ),
        ),
    )
    with pytest.raises(ToolPlanValidationError) as exc:
        validate_tool_plan(plan, tool_schemas={"read": {"type": "object"}})
    assert exc.value.code == "invalid_tool_arguments"


def test_validate_tool_plan_can_skip_schema_validation() -> None:
    plan = ToolPlan(
        kind="tool_calls",
        calls=(
            ToolCallEnvelope(
                tool_call_id="1",
                tool_name="read",
                arguments_json='{"path": 123}',
            ),
        ),
    )
    validated = validate_tool_plan(
        plan,
        tool_schemas={
            "read": {"type": "object", "properties": {"path": {"type": "string"}}}
        },
        strict_schema_validation=False,
    )
    assert validated.calls[0].arguments_json == '{"path":123}'


@pytest.mark.parametrize(
    "choice",
    [
        "invalid",
        123,
        {"type": "tool"},
        {"type": "function", "function": "x"},
        {"type": "function", "function": {"name": "  "}},
    ],
)
def test_normalize_tool_choice_rejects_invalid_values(choice: object) -> None:
    with pytest.raises(ToolPlanValidationError) as exc:
        _normalize_tool_choice(choice)
    assert exc.value.code == "invalid_tool_choice"


def test_validate_json_schema_handles_non_mapping_schema() -> None:
    _validate_json_schema({"x": 1}, "not-a-schema", path="root")


def test_validate_json_schema_rejects_enum_const_and_union() -> None:
    with pytest.raises(ToolPlanValidationError):
        _validate_json_schema("c", {"enum": ["a", "b"]}, path="root")
    with pytest.raises(ToolPlanValidationError):
        _validate_json_schema("b", {"const": "a"}, path="root")
    with pytest.raises(ToolPlanValidationError):
        _validate_json_schema(1, {"type": ["string", "null"]}, path="root")
    _validate_json_schema("ok", {"type": ["string", "null"]}, path="root")


def test_validate_json_schema_anyof_oneof_allof_paths() -> None:
    with pytest.raises(ToolPlanValidationError):
        _validate_json_schema(
            "a", {"anyOf": [{"const": "b"}, {"const": "c"}]}, path="x"
        )

    with pytest.raises(ToolPlanValidationError):
        _validate_json_schema(
            "a", {"oneOf": [{"const": "b"}, {"const": "c"}]}, path="x"
        )

    with pytest.raises(ToolPlanValidationError):
        _validate_json_schema(
            "a", {"oneOf": [{"type": "string"}, {"const": "a"}]}, path="x"
        )

    with pytest.raises(ToolPlanValidationError):
        _validate_json_schema(
            "a", {"allOf": [{"type": "string"}, {"const": "b"}]}, path="x"
        )


def test_validate_json_schema_object_validation_paths() -> None:
    with pytest.raises(ToolPlanValidationError):
        _validate_json_schema("x", {"type": "object"}, path="obj")

    with pytest.raises(ToolPlanValidationError):
        _validate_json_schema(
            {},
            {"type": "object", "required": ["a"]},
            path="obj",
        )

    with pytest.raises(ToolPlanValidationError):
        _validate_json_schema(
            {"a": 1, "b": 2},
            {
                "type": "object",
                "properties": {"a": {"type": "integer"}},
                "additionalProperties": False,
            },
            path="obj",
        )

    with pytest.raises(ToolPlanValidationError):
        _validate_json_schema(
            {"a": 1, "b": "x"},
            {
                "type": "object",
                "properties": {"a": {"type": "integer"}},
                "additionalProperties": {"type": "integer"},
            },
            path="obj",
        )

    _validate_json_schema(
        {"a": 1, "b": 2},
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "additionalProperties": {"type": "integer"},
        },
        path="obj",
    )


def test_validate_json_schema_array_validation_paths() -> None:
    with pytest.raises(ToolPlanValidationError):
        _validate_json_schema("x", {"type": "array"}, path="arr")

    with pytest.raises(ToolPlanValidationError):
        _validate_json_schema([1], {"type": "array", "minItems": 2}, path="arr")

    with pytest.raises(ToolPlanValidationError):
        _validate_json_schema([1, 2], {"type": "array", "maxItems": 1}, path="arr")

    with pytest.raises(ToolPlanValidationError):
        _validate_json_schema(
            [1, "x"], {"type": "array", "items": {"type": "integer"}}, path="arr"
        )


def test_validate_json_schema_union_runs_object_constraints() -> None:
    with pytest.raises(ToolPlanValidationError):
        _validate_json_schema(
            {},
            {
                "type": ["object", "null"],
                "required": ["a"],
            },
            path="obj",
        )

    with pytest.raises(ToolPlanValidationError):
        _validate_json_schema(
            {"a": 1, "b": "x"},
            {
                "type": ["object", "null"],
                "properties": {"a": {"type": "integer"}},
                "additionalProperties": {"type": "integer"},
            },
            path="obj",
        )


def test_validate_json_schema_union_runs_array_constraints() -> None:
    with pytest.raises(ToolPlanValidationError):
        _validate_json_schema(
            [1, "x"],
            {"type": ["array", "null"], "items": {"type": "integer"}},
            path="arr",
        )

    with pytest.raises(ToolPlanValidationError):
        _validate_json_schema(
            [1],
            {"type": ["array", "null"], "minItems": 2},
            path="arr",
        )


def test_validate_any_of_false_and_true() -> None:
    assert _validate_any_of("a", [{"const": "b"}], path="root") is False
    assert _validate_any_of("a", [{"const": "a"}], path="root") is True


def test_matches_type_matrix_and_require_type_errors() -> None:
    assert _matches_type("x", 1) is False
    assert _matches_type(None, "null") is True
    assert _matches_type(True, "boolean") is True
    assert _matches_type(3, "integer") is True
    assert _matches_type(3.2, "number") is True
    assert _matches_type("x", "string") is True
    assert _matches_type([], "array") is True
    assert _matches_type({}, "object") is True
    assert _matches_type("x", "unknown") is True

    with pytest.raises(ToolPlanValidationError):
        _require_type(1, "string", path="root")


def test_direct_object_and_array_schema_non_matching_types() -> None:
    with pytest.raises(ToolPlanValidationError):
        _validate_object_schema("x", {"type": "object"}, path="obj")
    with pytest.raises(ToolPlanValidationError):
        _validate_array_schema("x", {"type": "array"}, path="arr")
