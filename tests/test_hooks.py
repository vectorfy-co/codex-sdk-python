import asyncio

import pytest

from codex_sdk.events import (
    ItemCompletedEvent,
    ItemStartedEvent,
    ItemUpdatedEvent,
    ThreadError,
    ThreadErrorEvent,
    ThreadStartedEvent,
    TurnCompletedEvent,
    TurnFailedEvent,
    TurnStartedEvent,
    Usage,
)
from codex_sdk.hooks import ThreadHooks, dispatch_event
from codex_sdk.items import AgentMessageItem


@pytest.mark.asyncio
async def test_dispatch_event_calls_hooks():
    calls = []
    seen = asyncio.Event()

    async def on_event(event) -> None:
        calls.append(("event", event.type))
        seen.set()

    hooks = ThreadHooks(
        on_event=on_event,
        on_thread_started=lambda event: calls.append(("thread", event.thread_id)),
        on_turn_started=lambda event: calls.append(("turn_started", event.type)),
        on_turn_completed=lambda event: calls.append(("turn_completed", event.type)),
        on_turn_failed=lambda event: calls.append(("turn_failed", event.type)),
        on_item_started=lambda event: calls.append(("item_started", event.item.type)),
        on_item_updated=lambda event: calls.append(("item_updated", event.item.type)),
        on_item_completed=lambda event: calls.append(
            ("item_completed", event.item.type)
        ),
        on_item=lambda item: calls.append(("item", item.type)),
        on_item_type={
            "agent_message": lambda item: calls.append(("item_type", item.type))
        },
        on_error=lambda event: calls.append(("error", event.type)),
    )

    item = AgentMessageItem(id="i1", type="agent_message", text="hi")
    events = [
        ThreadStartedEvent(type="thread.started", thread_id="thread-1"),
        TurnStartedEvent(type="turn.started"),
        TurnCompletedEvent(type="turn.completed", usage=Usage(1, 0, 2)),
        TurnFailedEvent(type="turn.failed", error=ThreadError(message="boom")),
        ItemStartedEvent(type="item.started", item=item),
        ItemUpdatedEvent(type="item.updated", item=item),
        ItemCompletedEvent(type="item.completed", item=item),
        ThreadErrorEvent(type="error", message="bad"),
    ]

    for event in events:
        await dispatch_event(hooks, event)

    assert seen.is_set()
    assert ("thread", "thread-1") in calls
    assert ("turn_started", "turn.started") in calls
    assert ("turn_completed", "turn.completed") in calls
    assert ("turn_failed", "turn.failed") in calls
    assert calls.count(("item", "agent_message")) == 3
    assert calls.count(("item_type", "agent_message")) == 3
    assert ("error", "error") in calls


@pytest.mark.asyncio
async def test_dispatch_event_ignores_unknown_event_types() -> None:
    """Cover the fallthrough path when the event type doesn't match any branch."""

    class DummyEvent:
        type = "not-a-real-event"

    seen = []
    hooks = ThreadHooks(on_event=lambda event: seen.append(event.type))
    await dispatch_event(hooks, DummyEvent())
    assert seen == ["not-a-real-event"]
