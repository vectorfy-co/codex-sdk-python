"""
Event types for the Codex SDK.

Based on event types from codex-rs/exec/src/exec_events.rs
"""

from dataclasses import dataclass
from typing import Literal, Union

from .items import ThreadItem


@dataclass
class Usage:
    """Describes the usage of tokens during a turn."""

    # The number of input tokens used during the turn
    input_tokens: int

    # The number of cached input tokens used during the turn
    cached_input_tokens: int

    # The number of output tokens used during the turn
    output_tokens: int


@dataclass
class ThreadError:
    """Fatal error emitted by the stream."""

    message: str


@dataclass
class ThreadStartedEvent:
    """Emitted when a new thread is started as the first event."""

    type: Literal["thread.started"]

    # The identifier of the new thread. Can be used to resume the thread later.
    thread_id: str


@dataclass
class TurnStartedEvent:
    """
    Emitted when a turn is started by sending a new prompt to the model.
    A turn encompasses all events that happen while the agent is processing the prompt.
    """

    type: Literal["turn.started"]


@dataclass
class TurnCompletedEvent:
    """Emitted when a turn is completed. Typically right after the assistant's response."""

    type: Literal["turn.completed"]
    usage: Usage


@dataclass
class TurnFailedEvent:
    """Indicates that a turn failed with an error."""

    type: Literal["turn.failed"]
    error: ThreadError


@dataclass
class ItemStartedEvent:
    """Emitted when a new item is added to the thread. Typically the item is initially "in progress"."""

    type: Literal["item.started"]
    item: ThreadItem


@dataclass
class ItemUpdatedEvent:
    """Emitted when an item is updated."""

    type: Literal["item.updated"]
    item: ThreadItem


@dataclass
class ItemCompletedEvent:
    """Signals that an item has reached a terminal state—either success or failure."""

    type: Literal["item.completed"]
    item: ThreadItem


@dataclass
class ThreadErrorEvent:
    """Represents an unrecoverable error emitted directly by the event stream."""

    type: Literal["error"]
    message: str


# Top-level JSONL events emitted by codex exec
ThreadEvent = Union[
    ThreadStartedEvent,
    TurnStartedEvent,
    TurnCompletedEvent,
    TurnFailedEvent,
    ItemStartedEvent,
    ItemUpdatedEvent,
    ItemCompletedEvent,
    ThreadErrorEvent,
]
