#!/usr/bin/env python3
"""
Thread resume example of the Codex Python SDK.

This example demonstrates how to:
1. Start a thread and get its ID
2. Resume a thread using its ID
3. Continue a conversation across sessions
"""

import asyncio
import os

from codex_sdk import Codex


async def main():
    """Main function demonstrating thread resume usage."""
    print("Starting Codex SDK thread resume example...")

    # Create a Codex instance
    codex = Codex()

    # Check if we have a saved thread ID
    saved_thread_id = os.environ.get("CODEX_THREAD_ID")

    if saved_thread_id:
        print(f"Resuming thread with ID: {saved_thread_id}")
        thread = codex.resume_thread(saved_thread_id)
    else:
        print("Starting new thread...")
        thread = codex.start_thread()

        # Save the thread ID for future use
        if thread.id:
            print(f"Save this thread ID for future use: {thread.id}")
            print(
                "You can set CODEX_THREAD_ID environment variable to resume this thread later"
            )

    print(f"Current thread ID: {thread.id}")

    # Run a command
    print("\nRunning command: 'What is the current working directory?'")
    turn = await thread.run("What is the current working directory?")

    print(f"Response: {turn.final_response}")

    # Continue the conversation
    print("\nContinuing conversation...")
    next_turn = await thread.run("Now list the files in that directory")

    print(f"Response: {next_turn.final_response}")

    # Show how to save thread ID for next time
    if thread.id:
        print("\nTo resume this thread later, set:")
        print(f"export CODEX_THREAD_ID={thread.id}")
        print("Then run this script again.")


if __name__ == "__main__":
    asyncio.run(main())
