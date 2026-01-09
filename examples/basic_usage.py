#!/usr/bin/env python3
"""
Basic usage example of the Codex Python SDK.

This example demonstrates how to:
1. Create a Codex instance
2. Start a thread
3. Run a simple command
4. Process the results
"""

import asyncio

from codex_sdk import Codex


async def main():
    """Main function demonstrating basic SDK usage."""
    print("Starting Codex SDK basic usage example...")

    # Create a Codex instance
    codex = Codex()

    # Start a new thread
    thread = codex.start_thread()
    print(f"Started thread with ID: {thread.id}")

    # Run a simple command
    print("Running command: 'List the files in the current directory'")
    turn = await thread.run("List the files in the current directory")

    # Print the results
    print(f"\nFinal response: {turn.final_response}")
    print(f"\nNumber of items: {len(turn.items)}")

    # Print details about each item
    for i, item in enumerate(turn.items):
        print(f"\nItem {i + 1}:")
        print(f"  Type: {item.type}")
        print(f"  ID: {item.id}")

        if item.type == "command_execution":
            print(f"  Command: {item.command}")
            print(f"  Status: {item.status}")
            if item.exit_code is not None:
                print(f"  Exit code: {item.exit_code}")
        elif item.type == "agent_message":
            print(f"  Text: {item.text[:100]}...")

    # Continue the conversation
    print("\n" + "=" * 50)
    print("Continuing the conversation...")

    next_turn = await thread.run(
        "Now create a simple Python script that prints 'Hello, World!'"
    )

    print(f"\nFinal response: {next_turn.final_response}")
    print(f"Number of items: {len(next_turn.items)}")


if __name__ == "__main__":
    asyncio.run(main())
