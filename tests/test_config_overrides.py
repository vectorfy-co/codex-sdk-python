from pathlib import Path

import pytest

from codex_sdk.config_overrides import encode_config_overrides, merge_config_overrides


def test_merge_config_overrides():
    assert merge_config_overrides() is None
    assert merge_config_overrides(None, {}) is None
    merged = merge_config_overrides({"a": 1}, {"b": 2}, {"a": 3})
    assert merged == {"a": 3, "b": 2}


def test_encode_config_overrides_literals(tmp_path: Path):
    overrides = {
        "analytics.enabled": True,
        "count": 2,
        "ratio": 1.5,
        "name": "value",
        "path": tmp_path / "file.txt",
        "list": [1, "two"],
        "nested": {"simple": "ok", "complex key": 3},
    }
    encoded = encode_config_overrides(overrides)

    assert "analytics.enabled=true" in encoded
    assert "count=2" in encoded
    assert "ratio=1.5" in encoded
    assert 'name="value"' in encoded
    assert f'path="{tmp_path / "file.txt"}"' in encoded
    assert 'list=[1, "two"]' in encoded
    assert 'nested={ simple = "ok", "complex key" = 3 }' in encoded


def test_encode_config_overrides_rejects_invalid_values():
    with pytest.raises(TypeError):
        encode_config_overrides({"bad": None})
    with pytest.raises(ValueError):
        encode_config_overrides({"bad": float("nan")})
    with pytest.raises(TypeError):
        encode_config_overrides({"bad": object()})


def test_encode_config_overrides_rejects_non_string_mapping_keys():
    with pytest.raises(TypeError):
        encode_config_overrides({"nested": {1: "nope"}})  # type: ignore[dict-item]
