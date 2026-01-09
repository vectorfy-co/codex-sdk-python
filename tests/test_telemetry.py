import builtins
import sys
from types import ModuleType
from typing import Any, Dict, List

import pytest

import codex_sdk.telemetry as telemetry


def test_maybe_logfire_returns_none_when_import_fails(monkeypatch: pytest.MonkeyPatch):
    original_import = builtins.__import__

    def fake_import(
        name: str,
        globals: Dict[str, Any] | None = None,
        locals: Dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ):
        if name == "logfire":
            raise ImportError("no logfire")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert telemetry._maybe_logfire() is None


def test_maybe_logfire_returns_none_when_config_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_logfire = ModuleType("logfire")
    fake_logfire.DEFAULT_LOGFIRE_INSTANCE = object()
    monkeypatch.setitem(sys.modules, "logfire", fake_logfire)
    assert telemetry._maybe_logfire() is None


def test_maybe_logfire_returns_none_on_exception(monkeypatch: pytest.MonkeyPatch):
    class BrokenInstance:
        @property
        def config(self) -> Any:
            raise RuntimeError("boom")

    fake_logfire = ModuleType("logfire")
    fake_logfire.DEFAULT_LOGFIRE_INSTANCE = BrokenInstance()
    monkeypatch.setitem(sys.modules, "logfire", fake_logfire)
    assert telemetry._maybe_logfire() is None


def test_span_uses_logfire_when_initialized(monkeypatch: pytest.MonkeyPatch):
    calls: List[Dict[str, Any]] = []

    class FakeConfig:
        _initialized = True

    class FakeInstance:
        config = FakeConfig()

    class FakeSpan:
        def __init__(self, name: str, attributes: Dict[str, Any]):
            self._name = name
            self._attributes = attributes

        def __enter__(self) -> None:
            calls.append({"name": self._name, "attributes": dict(self._attributes)})

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    fake_logfire = ModuleType("logfire")
    fake_logfire.DEFAULT_LOGFIRE_INSTANCE = FakeInstance()

    def span(name: str, **attributes: Any) -> FakeSpan:
        return FakeSpan(name, attributes)

    fake_logfire.span = span  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "logfire", fake_logfire)

    with telemetry.span("test_span", answer=42):
        pass

    assert calls == [{"name": "test_span", "attributes": {"answer": 42}}]
