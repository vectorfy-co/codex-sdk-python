"""Optional Logfire instrumentation helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional


def _maybe_logfire() -> Optional[Any]:
    try:
        import logfire
    except ImportError:
        return None

    try:
        instance = getattr(logfire, "DEFAULT_LOGFIRE_INSTANCE", None)
        config = getattr(instance, "config", None)
        if config is None:
            return None
        if getattr(config, "_initialized", False):
            return logfire
    except Exception:
        return None

    return None


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    logfire = _maybe_logfire()
    if logfire is None:
        yield
        return

    with logfire.span(name, **attributes):
        yield
