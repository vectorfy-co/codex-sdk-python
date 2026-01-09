#!/usr/bin/env python3
"""
Streaming example of the Codex Python SDK.

This example demonstrates how to:
1. Use streaming to get real-time updates
2. Process different types of events
3. Handle streaming responses
"""

import asyncio

from codex_sdk import Codex


async def main():
    """Main function demonstrating streaming usage."""
    print("Starting Codex SDK streaming example...")

    # Create a Codex instance
    codex = Codex()

    # Start a new thread
    thread = codex.start_thread()
    print(f"Started thread with ID: {thread.id}")

    # Run a command with streaming
    print(
        "Running command with streaming: 'Create a Python function to calculate fibonacci numbers'"
    )

    result = await thread.run_streamed(
        "Create a Python function to calculate fibonacci numbers"
    )

    # Process events as they come in
    async for event in result.events:
        if event.type == "thread.started":
            print(f"Thread started with ID: {event.thread_id}")
        elif event.type == "turn.started":
            print("Turn started...")
        elif event.type == "item.started":
            print(f"Item started: {event.item.type} (ID: {event.item.id})")
        elif event.type == "item.updated":
            print(f"Item updated: {event.item.type} (ID: {event.item.id})")
        elif event.type == "item.completed":
            print(f"Item completed: {event.item.type} (ID: {event.item.id})")

            # Show details for specific item types
            if event.item.type == "command_execution":
                print(f"  Command: {event.item.command}")
                print(f"  Status: {event.item.status}")
                if event.item.exit_code is not None:
                    print(f"  Exit code: {event.item.exit_code}")
            elif event.item.type == "file_change":
                print(f"  Changes: {len(event.item.changes)} files")
                for change in event.item.changes:
                    print(f"    {change.kind}: {change.path}")
            elif event.item.type == "agent_message":
                print(f"  Message: {event.item.text[:100]}...")
        elif event.type == "turn.completed":
            print(f"Turn completed. Usage: {event.usage}")
        elif event.type == "turn.failed":
            print(f"Turn failed: {event.error.message}")
        elif event.type == "error":
            print(f"Error: {event.message}")

    print("\nStreaming completed!")


if __name__ == "__main__":
    asyncio.run(main())
