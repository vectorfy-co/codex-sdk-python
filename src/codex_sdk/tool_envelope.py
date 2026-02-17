"""Shared helpers for model-driven tool-calling envelopes.

This module standardizes how Codex structured output is interpreted for
host-managed tool loops.
"""

from __future__ import annotations

import json
import logging
from base64 import b64encode
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolCallEnvelope:
    """One planned tool call emitted by the model envelope."""

    tool_call_id: str
    tool_name: str
    arguments_json: str


@dataclass(frozen=True)
class ToolPlan:
    """Parsed planner output.

    kind:
      - ``tool_calls`` when one or more tool calls are requested.
      - ``final`` when no tool call is requested and final text is provided.
    """

    kind: Literal["tool_calls", "final"]
    calls: Tuple[ToolCallEnvelope, ...] = ()
    content: str = ""


class ToolPlanValidationError(ValueError):
    """Raised for malformed envelope output or invalid tool-call plans."""

    def __init__(self, code: str, message: str):
        """Initialize a validation error with a stable error code."""
        super().__init__(message)
        self.code = code
        self.message = message


ToolChoice = Optional[Union[str, Mapping[str, Any]]]


def jsonable(value: Any) -> Any:
    """Convert common rich objects into JSON-serializable values."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)

    if hasattr(value, "model_dump") and callable(value.model_dump):
        return value.model_dump(mode="json")

    if isinstance(value, bytes):
        return {"type": "bytes", "base64": b64encode(value).decode("ascii")}

    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]

    return value


def json_dumps(value: Any) -> str:
    """Canonical JSON dump used in prompts and normalized tool arguments."""
    try:
        return json.dumps(
            jsonable(value),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except TypeError as exc:
        logger.error(
            "Failed to serialize value to JSON",
            extra={"value_type": type(value).__name__, "error": str(exc)},
        )
        raise


def build_envelope_schema(tool_names: Sequence[str]) -> Dict[str, Any]:
    """Build constrained schema for planner output.

    Output shape:
      {"tool_calls": [...], "final": "..."}
    """
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


def parse_tool_plan(output: Any) -> ToolPlan:
    """Parse raw envelope output into a normalized ``ToolPlan``."""
    if not isinstance(output, dict):
        raise ToolPlanValidationError(
            "invalid_envelope", "Planner output must be a JSON object."
        )

    if "tool_calls" not in output or "final" not in output:
        raise ToolPlanValidationError(
            "invalid_envelope",
            "Planner output must contain both `tool_calls` and `final`.",
        )

    raw_calls = output.get("tool_calls")
    final = output.get("final")
    if not isinstance(raw_calls, list):
        raise ToolPlanValidationError(
            "invalid_envelope", "`tool_calls` must be an array."
        )
    if not isinstance(final, str):
        raise ToolPlanValidationError("invalid_envelope", "`final` must be a string.")

    calls = []
    for index, call in enumerate(raw_calls):
        if not isinstance(call, dict):
            raise ToolPlanValidationError(
                "invalid_envelope", f"tool_calls[{index}] must be an object."
            )

        tool_call_id = call.get("id")
        tool_name = call.get("name")
        arguments = call.get("arguments")
        if not isinstance(tool_call_id, str):
            raise ToolPlanValidationError(
                "invalid_envelope", f"tool_calls[{index}].id must be a string."
            )
        if not isinstance(tool_name, str):
            raise ToolPlanValidationError(
                "invalid_envelope", f"tool_calls[{index}].name must be a string."
            )
        if not isinstance(arguments, str):
            raise ToolPlanValidationError(
                "invalid_envelope",
                f"tool_calls[{index}].arguments must be a JSON string.",
            )

        calls.append(
            ToolCallEnvelope(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments_json=arguments,
            )
        )

    if calls and final.strip():
        raise ToolPlanValidationError(
            "invalid_envelope",
            "Planner output cannot contain both non-empty `final` and `tool_calls`.",
        )

    if calls:
        return ToolPlan(kind="tool_calls", calls=tuple(calls), content="")

    return ToolPlan(kind="final", calls=(), content=final)


def validate_tool_plan(
    plan: ToolPlan,
    *,
    tool_schemas: Mapping[str, Any],
    tool_choice: ToolChoice = None,
    parallel_tool_calls: Optional[bool] = None,
    max_tool_calls: Optional[int] = None,
    strict_schema_validation: bool = True,
    fallback_id_prefix: str = "call",
) -> ToolPlan:
    """Validate and normalize a parsed tool plan.

    Normalizations:
    - Rewrites argument JSON into canonical form.
    - Replaces missing/blank tool-call IDs with deterministic fallback IDs.
    """
    choice_mode, choice_name = _normalize_tool_choice(tool_choice)

    if plan.kind == "final":
        if choice_mode in {"required", "function"}:
            raise ToolPlanValidationError(
                "tool_choice_mismatch",
                "Model returned final text while tool_choice requires a tool call.",
            )
        return plan

    calls = list(plan.calls)
    if not calls:
        raise ToolPlanValidationError(
            "invalid_envelope", "`tool_calls` plan must contain at least one call."
        )

    if choice_mode == "none":
        raise ToolPlanValidationError(
            "tool_choice_mismatch",
            "Model emitted tool calls while tool_choice is `none`.",
        )

    if max_tool_calls is not None and len(calls) > max_tool_calls:
        raise ToolPlanValidationError(
            "too_many_tool_calls",
            f"Model emitted {len(calls)} tool calls (max {max_tool_calls}).",
        )

    if parallel_tool_calls is False and len(calls) > 1:
        raise ToolPlanValidationError(
            "parallel_tool_calls_disabled",
            "Model emitted multiple tool calls while parallel_tool_calls is false.",
        )

    normalized_calls = []
    for index, call in enumerate(calls):
        call_id = (
            call.tool_call_id.strip() if isinstance(call.tool_call_id, str) else ""
        )
        if not call_id:
            call_id = f"{fallback_id_prefix}_{index}"

        tool_name = call.tool_name.strip()
        if not tool_name:
            raise ToolPlanValidationError(
                "unknown_tool", f"tool_calls[{index}] has empty tool name."
            )

        if choice_mode == "function" and choice_name and tool_name != choice_name:
            raise ToolPlanValidationError(
                "tool_choice_mismatch",
                (
                    "Model emitted tool "
                    f"`{tool_name}` while tool_choice requires `{choice_name}`."
                ),
            )

        schema = tool_schemas.get(tool_name)
        if schema is None:
            raise ToolPlanValidationError(
                "unknown_tool", f"Model requested undeclared tool `{tool_name}`."
            )

        try:
            parsed_args = json.loads(call.arguments_json)
        except json.JSONDecodeError as exc:
            raise ToolPlanValidationError(
                "invalid_tool_arguments",
                f"Tool `{tool_name}` arguments are not valid JSON: {exc.msg}",
            ) from exc

        if not isinstance(parsed_args, dict):
            raise ToolPlanValidationError(
                "invalid_tool_arguments",
                f"Tool `{tool_name}` arguments must decode to a JSON object.",
            )

        if strict_schema_validation:
            _validate_json_schema(parsed_args, schema, path=f"tool:{tool_name}")

        normalized_calls.append(
            ToolCallEnvelope(
                tool_call_id=call_id,
                tool_name=tool_name,
                arguments_json=json_dumps(parsed_args),
            )
        )

    return ToolPlan(kind="tool_calls", calls=tuple(normalized_calls), content="")


def _normalize_tool_choice(tool_choice: ToolChoice) -> Tuple[str, Optional[str]]:
    """Normalize tool choice into a `(mode, function_name)` tuple."""
    if tool_choice is None:
        return "auto", None

    if isinstance(tool_choice, str):
        mode = tool_choice.strip().lower()
        if mode in {"auto", "none", "required"}:
            return mode, None
        raise ToolPlanValidationError(
            "invalid_tool_choice",
            f"Unsupported tool_choice string `{tool_choice}`.",
        )

    if not isinstance(tool_choice, Mapping):
        raise ToolPlanValidationError(
            "invalid_tool_choice", "tool_choice must be a string or object."
        )

    choice_type = tool_choice.get("type")
    if choice_type != "function":
        raise ToolPlanValidationError(
            "invalid_tool_choice",
            "tool_choice object must use type=`function`.",
        )

    function = tool_choice.get("function")
    if not isinstance(function, Mapping):
        raise ToolPlanValidationError(
            "invalid_tool_choice", "tool_choice.function must be an object."
        )

    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ToolPlanValidationError(
            "invalid_tool_choice",
            "tool_choice.function.name must be a non-empty string.",
        )

    return "function", name.strip()


def _validate_json_schema(value: Any, schema: Any, *, path: str) -> None:
    """Validate a value against a subset of JSON Schema used for tool arguments."""
    if not isinstance(schema, Mapping):
        return

    # Generic enum support.
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        raise ToolPlanValidationError(
            "invalid_tool_arguments",
            f"{path}: value {value!r} is not in enum {enum_values!r}.",
        )

    if "const" in schema and value != schema["const"]:
        raise ToolPlanValidationError(
            "invalid_tool_arguments",
            f"{path}: value {value!r} does not match const {schema['const']!r}.",
        )

    schema_type = schema.get("type")
    resolved_type: Optional[str] = None
    if isinstance(schema_type, list):
        # Support basic union form, e.g. ["string", "null"].
        matched_types = [t for t in schema_type if _matches_type(value, t)]
        if matched_types:
            if "object" in matched_types and isinstance(value, dict):
                resolved_type = "object"
            elif "array" in matched_types and isinstance(value, list):
                resolved_type = "array"
            else:
                first = matched_types[0]
                resolved_type = first if isinstance(first, str) else None
        else:
            raise ToolPlanValidationError(
                "invalid_tool_arguments",
                f"{path}: expected one of types {schema_type!r}.",
            )
    elif isinstance(schema_type, str):
        _require_type(value, schema_type, path=path)
        resolved_type = schema_type

    # Composite schemas.
    any_of = schema.get("anyOf")
    if isinstance(any_of, list) and any_of:
        if not _validate_any_of(value, any_of, path=path):
            raise ToolPlanValidationError(
                "invalid_tool_arguments", f"{path}: no anyOf branch matched."
            )

    one_of = schema.get("oneOf")
    if isinstance(one_of, list) and one_of:
        matches = 0
        for branch in one_of:
            try:
                _validate_json_schema(value, branch, path=path)
            except ToolPlanValidationError:
                continue
            matches += 1
        if matches != 1:
            raise ToolPlanValidationError(
                "invalid_tool_arguments",
                f"{path}: expected exactly one oneOf branch match, got {matches}.",
            )

    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        for branch in all_of:
            _validate_json_schema(value, branch, path=path)

    if resolved_type == "object":
        _validate_object_schema(value, schema, path=path)
    elif resolved_type == "array":
        _validate_array_schema(value, schema, path=path)


def _validate_any_of(value: Any, branches: Sequence[Any], *, path: str) -> bool:
    """Return `True` when at least one branch validates without errors."""
    for branch in branches:
        try:
            _validate_json_schema(value, branch, path=path)
        except ToolPlanValidationError:
            continue
        return True
    return False


def _validate_object_schema(
    value: Any, schema: Mapping[str, Any], *, path: str
) -> None:
    """Validate object constraints (`required`, properties, additionalProperties)."""
    if not isinstance(value, dict):
        raise ToolPlanValidationError(
            "invalid_tool_arguments", f"{path}: expected object."
        )

    required = schema.get("required")
    if isinstance(required, list):
        for field in required:
            if isinstance(field, str) and field not in value:
                raise ToolPlanValidationError(
                    "invalid_tool_arguments",
                    f"{path}: missing required field `{field}`.",
                )

    properties = schema.get("properties")
    property_schemas: Mapping[str, Any] = (
        properties if isinstance(properties, Mapping) else {}
    )

    additional_properties = schema.get("additionalProperties", True)
    if additional_properties is False:
        unknown = [key for key in value.keys() if key not in property_schemas]
        if unknown:
            raise ToolPlanValidationError(
                "invalid_tool_arguments",
                f"{path}: unexpected fields {unknown!r}.",
            )

    for key, item in value.items():
        prop_schema = property_schemas.get(key)
        if prop_schema is not None:
            _validate_json_schema(item, prop_schema, path=f"{path}.{key}")
        elif isinstance(additional_properties, Mapping):
            _validate_json_schema(
                item,
                additional_properties,
                path=f"{path}.{key}",
            )


def _validate_array_schema(value: Any, schema: Mapping[str, Any], *, path: str) -> None:
    """Validate array constraints (`items`, `minItems`, `maxItems`)."""
    if not isinstance(value, list):
        raise ToolPlanValidationError(
            "invalid_tool_arguments", f"{path}: expected array."
        )

    items_schema = schema.get("items")
    if items_schema is not None:
        for index, item in enumerate(value):
            _validate_json_schema(item, items_schema, path=f"{path}[{index}]")

    min_items = schema.get("minItems")
    if isinstance(min_items, int) and len(value) < min_items:
        raise ToolPlanValidationError(
            "invalid_tool_arguments",
            f"{path}: expected at least {min_items} items.",
        )

    max_items = schema.get("maxItems")
    if isinstance(max_items, int) and len(value) > max_items:
        raise ToolPlanValidationError(
            "invalid_tool_arguments",
            f"{path}: expected at most {max_items} items.",
        )


def _matches_type(value: Any, schema_type: Any) -> bool:
    """Return whether a value satisfies a JSON-schema primitive type label."""
    if not isinstance(schema_type, str):
        return False
    if schema_type == "null":
        return value is None
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return (isinstance(value, int) and not isinstance(value, bool)) or isinstance(
            value, float
        )
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "object":
        return isinstance(value, dict)
    return True


def _require_type(value: Any, schema_type: str, *, path: str) -> None:
    """Raise when a value does not satisfy a required JSON-schema type."""
    if not _matches_type(value, schema_type):
        raise ToolPlanValidationError(
            "invalid_tool_arguments", f"{path}: expected type `{schema_type}`."
        )
