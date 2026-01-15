"""Utilities for reacting to streamed Codex events."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from .events import ThreadEvent
from .items import ThreadItem

Hook = Callable[[Any], Any]


@dataclass
class ThreadHooks:
    """Callbacks for streamed thread events.

    Each hook can be sync or async. If a hook raises, the turn will stop.
    """

    on_event: Optional[Hook] = None
    on_thread_started: Optional[Hook] = None
    on_turn_started: Optional[Hook] = None
    on_turn_completed: Optional[Hook] = None
    on_turn_failed: Optional[Hook] = None
    on_item_started: Optional[Hook] = None
    on_item_updated: Optional[Hook] = None
    on_item_completed: Optional[Hook] = None
    on_error: Optional[Hook] = None

    # Item-specific hooks
    on_item: Optional[Hook] = None
    on_item_type: Optional[Mapping[str, Hook]] = None


async def dispatch_event(hooks: ThreadHooks, event: ThreadEvent) -> None:
    """Dispatch a single event to the configured hooks."""
    await _maybe_call(hooks.on_event, event)

    if event.type == "thread.started":
        await _maybe_call(hooks.on_thread_started, event)
    elif event.type == "turn.started":
        await _maybe_call(hooks.on_turn_started, event)
    elif event.type == "turn.completed":
        await _maybe_call(hooks.on_turn_completed, event)
    elif event.type == "turn.failed":
        await _maybe_call(hooks.on_turn_failed, event)
    elif event.type == "item.started":
        await _maybe_call(hooks.on_item_started, event)
        await _dispatch_item_hooks(hooks, event.item)
    elif event.type == "item.updated":
        await _maybe_call(hooks.on_item_updated, event)
        await _dispatch_item_hooks(hooks, event.item)
    elif event.type == "item.completed":
        await _maybe_call(hooks.on_item_completed, event)
        await _dispatch_item_hooks(hooks, event.item)
    elif event.type == "error":
        await _maybe_call(hooks.on_error, event)


async def _dispatch_item_hooks(hooks: ThreadHooks, item: ThreadItem) -> None:
    await _maybe_call(hooks.on_item, item)
    if hooks.on_item_type:
        hook = hooks.on_item_type.get(item.type)
        await _maybe_call(hook, item)


async def _maybe_call(func: Optional[Hook], arg: Any) -> None:
    if func is None:
        return
    result = func(arg)
    if inspect.isawaitable(result):
        await result
