"""Helpers for encoding Codex CLI --config overrides."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

ConfigOverrides = Mapping[str, Any]

_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def merge_config_overrides(
    *overrides: Optional[ConfigOverrides],
) -> Optional[Dict[str, Any]]:
    """Merge config override mappings, later entries winning on conflicts."""
    merged: Dict[str, Any] = {}
    for mapping in overrides:
        if not mapping:
            continue
        merged.update(mapping)
    return merged or None


def encode_config_overrides(overrides: ConfigOverrides) -> List[str]:
    """Encode overrides to CLI-friendly key=value strings.

    Values are serialized as TOML literals (inline tables/arrays where needed).
    """
    encoded: List[str] = []
    for key, value in overrides.items():
        encoded.append(f"{key}={_toml_literal(value)}")
    return encoded


def _toml_key(key: Any) -> str:
    if not isinstance(key, str):
        raise TypeError("TOML inline table keys must be strings")
    if _KEY_RE.match(key):
        return key
    return json.dumps(key)


def _toml_literal(value: Any) -> str:
    if value is None:
        raise TypeError("None is not a valid TOML literal")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite floats are not valid TOML literals")
        return repr(value)
    if isinstance(value, Path):
        return json.dumps(str(value))
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_literal(item) for item in value) + "]"
    if isinstance(value, Mapping):
        inner = ", ".join(
            f"{_toml_key(k)} = {_toml_literal(v)}" for k, v in value.items()
        )
        return "{ " + inner + " }"

    raise TypeError(f"Unsupported config override value: {type(value).__name__}")
