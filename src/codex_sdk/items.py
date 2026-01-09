"""
Item types for the Codex SDK.

Based on item types from codex-rs/exec/src/exec_events.rs
"""

from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Union

# The status of a command execution
CommandExecutionStatus = Literal["in_progress", "completed", "failed", "declined"]

# Indicates the type of the file change
PatchChangeKind = Literal["add", "delete", "update"]

# The status of a file change
PatchApplyStatus = Literal["in_progress", "completed", "failed"]

# The status of an MCP tool call
McpToolCallStatus = Literal["in_progress", "completed", "failed"]


@dataclass
class FileUpdateChange:
    """A set of file changes by the agent."""

    path: str
    kind: PatchChangeKind


@dataclass
class TodoItem:
    """An item in the agent's to-do list."""

    text: str
    completed: bool


@dataclass
class CommandExecutionItem:
    """A command executed by the agent."""

    id: str
    type: Literal["command_execution"]

    # The command line executed by the agent
    command: str

    # Aggregated stdout and stderr captured while the command was running
    aggregated_output: str

    # Set when the command exits; omitted while still running
    exit_code: Optional[int] = None

    # Current status of the command execution
    status: CommandExecutionStatus = "in_progress"


@dataclass
class FileChangeItem:
    """A set of file changes by the agent. Emitted once the patch succeeds or fails."""

    id: str
    type: Literal["file_change"]

    # Individual file changes that comprise the patch
    changes: List[FileUpdateChange]

    # Whether the patch ultimately succeeded or failed
    status: PatchApplyStatus


@dataclass
class McpToolCallItem:
    """
    Represents a call to an MCP tool. The item starts when the invocation is dispatched
    and completes when the MCP server reports success or failure.
    """

    id: str
    type: Literal["mcp_tool_call"]

    # Name of the MCP server handling the request
    server: str

    # The tool invoked on the MCP server
    tool: str

    # Current status of the tool invocation
    status: McpToolCallStatus

    # Arguments forwarded to the tool invocation
    arguments: Any = None

    # Result payload returned by the MCP server for successful calls
    result: Optional["McpToolCallItemResult"] = None

    # Error message reported for failed calls
    error: Optional["McpToolCallItemError"] = None


@dataclass
class McpToolCallItemResult:
    content: List[Any]
    structured_content: Optional[Any] = None


@dataclass
class McpToolCallItemError:
    message: str


@dataclass
class AgentMessageItem:
    """Response from the agent. Either natural-language text or JSON when structured output is requested."""

    id: str
    type: Literal["agent_message"]

    # Either natural-language text or JSON when structured output is requested
    text: str


@dataclass
class ReasoningItem:
    """Agent's reasoning summary."""

    id: str
    type: Literal["reasoning"]
    text: str


@dataclass
class WebSearchItem:
    """Captures a web search request. Completes when results are returned to the agent."""

    id: str
    type: Literal["web_search"]
    query: str


@dataclass
class ErrorItem:
    """Describes a non-fatal error surfaced as an item."""

    id: str
    type: Literal["error"]
    message: str


@dataclass
class TodoListItem:
    """
    Tracks the agent's running to-do list. Starts when the plan is issued, updates as steps change,
    and completes when the turn ends.
    """

    id: str
    type: Literal["todo_list"]
    items: List[TodoItem]


# Canonical union of thread items and their type-specific payloads
ThreadItem = Union[
    AgentMessageItem,
    ReasoningItem,
    CommandExecutionItem,
    FileChangeItem,
    McpToolCallItem,
    WebSearchItem,
    TodoListItem,
    ErrorItem,
]
