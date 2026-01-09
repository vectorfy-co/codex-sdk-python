#!/usr/bin/env python3
"""
Permission Levels and Sandbox Modes Example

This example demonstrates how to use different permission levels and sandbox modes
to control what Codex can and cannot do in your Python applications.
"""

import asyncio

from codex_sdk import Codex, ThreadOptions


async def test_read_only_sandbox():
    """Test with read-only sandbox - most restrictive."""
    print("=" * 60)
    print("READ-ONLY SANDBOX (Most Restrictive)")
    print("=" * 60)

    codex = Codex()
    thread_options = ThreadOptions(
        sandbox_mode="read-only", working_directory=".", skip_git_repo_check=True
    )

    thread = codex.start_thread(thread_options)
    print(f"Thread ID: {thread.id}")

    # Try to create a file (should fail)
    print("\n1. Trying to create a file (should fail):")
    turn = await thread.run("Create a test file called 'test.txt' with some content")

    print("Response:")
    try:
        print(
            turn.final_response[:300] + "..."
            if len(turn.final_response) > 300
            else turn.final_response
        )
    except UnicodeEncodeError:
        safe_response = turn.final_response.encode("utf-8", errors="replace").decode(
            "utf-8"
        )
        print(
            safe_response[:300] + "..." if len(safe_response) > 300 else safe_response
        )

    # Try to list files (should work)
    print("\n2. Trying to list files (should work):")
    turn2 = await thread.run("List the files in the current directory")

    print("Response:")
    try:
        print(
            turn2.final_response[:200] + "..."
            if len(turn2.final_response) > 200
            else turn2.final_response
        )
    except UnicodeEncodeError:
        safe_response = turn2.final_response.encode("utf-8", errors="replace").decode(
            "utf-8"
        )
        print(
            safe_response[:200] + "..." if len(safe_response) > 200 else safe_response
        )


async def test_workspace_write_sandbox():
    """Test with workspace-write sandbox - moderate restrictions."""
    print("\n" + "=" * 60)
    print("WORKSPACE-WRITE SANDBOX (Moderate Restrictions)")
    print("=" * 60)

    codex = Codex()
    thread_options = ThreadOptions(
        sandbox_mode="workspace-write", working_directory=".", skip_git_repo_check=True
    )

    thread = codex.start_thread(thread_options)
    print(f"Thread ID: {thread.id}")

    # Try to create a file (might work in workspace)
    print("\n1. Trying to create a file in workspace:")
    turn = await thread.run(
        "Create a test file called 'workspace_test.txt' with content 'Hello from workspace'"
    )

    print("Response:")
    try:
        print(
            turn.final_response[:300] + "..."
            if len(turn.final_response) > 300
            else turn.final_response
        )
    except UnicodeEncodeError:
        safe_response = turn.final_response.encode("utf-8", errors="replace").decode(
            "utf-8"
        )
        print(
            safe_response[:300] + "..." if len(safe_response) > 300 else safe_response
        )

    # Try to access system files (should fail)
    print("\n2. Trying to access system files (should fail):")
    turn2 = await thread.run(
        "Try to read the contents of C:\\Windows\\System32\\drivers\\etc\\hosts"
    )

    print("Response:")
    try:
        print(
            turn2.final_response[:300] + "..."
            if len(turn2.final_response) > 300
            else turn2.final_response
        )
    except UnicodeEncodeError:
        safe_response = turn2.final_response.encode("utf-8", errors="replace").decode(
            "utf-8"
        )
        print(
            safe_response[:300] + "..." if len(safe_response) > 300 else safe_response
        )


async def test_full_access_sandbox():
    """Test with full access sandbox - least restrictive (use with caution!)."""
    print("\n" + "=" * 60)
    print("FULL ACCESS SANDBOX (Least Restrictive - Use with Caution!)")
    print("=" * 60)

    codex = Codex()
    thread_options = ThreadOptions(
        sandbox_mode="danger-full-access",
        working_directory=".",
        skip_git_repo_check=True,
    )

    thread = codex.start_thread(thread_options)
    print(f"Thread ID: {thread.id}")

    # Try to create a file (should work)
    print("\n1. Trying to create a file:")
    turn = await thread.run(
        "Create a test file called 'full_access_test.txt' with content 'Hello from full access'"
    )

    print("Response:")
    try:
        print(
            turn.final_response[:300] + "..."
            if len(turn.final_response) > 300
            else turn.final_response
        )
    except UnicodeEncodeError:
        safe_response = turn.final_response.encode("utf-8", errors="replace").decode(
            "utf-8"
        )
        print(
            safe_response[:300] + "..." if len(safe_response) > 300 else safe_response
        )

    # Try web search (should work with full access)
    print("\n2. Trying web search:")
    turn2 = await thread.run(
        "Search the web for 'Python 3.13 new features' and give me a brief summary"
    )

    print("Response:")
    try:
        print(
            turn2.final_response[:400] + "..."
            if len(turn2.final_response) > 400
            else turn2.final_response
        )
    except UnicodeEncodeError:
        safe_response = turn2.final_response.encode("utf-8", errors="replace").decode(
            "utf-8"
        )
        print(
            safe_response[:400] + "..." if len(safe_response) > 400 else safe_response
        )

    # Show what types of operations were performed
    print(f"\n3. Operations performed: {len(turn2.items)} items")
    command_executions = [
        item for item in turn2.items if item.type == "command_execution"
    ]
    web_searches = [item for item in turn2.items if item.type == "web_search"]
    agent_messages = [item for item in turn2.items if item.type == "agent_message"]

    print(f"   - Command executions: {len(command_executions)}")
    print(f"   - Web searches: {len(web_searches)}")
    print(f"   - Agent messages: {len(agent_messages)}")


async def test_custom_working_directory():
    """Test with a custom working directory."""
    print("\n" + "=" * 60)
    print("CUSTOM WORKING DIRECTORY")
    print("=" * 60)

    codex = Codex()
    thread_options = ThreadOptions(
        sandbox_mode="workspace-write",
        working_directory="examples",  # Restrict to examples directory
        skip_git_repo_check=True,
    )

    thread = codex.start_thread(thread_options)
    print(f"Thread ID: {thread.id}")
    print("Working directory: examples/")

    # Try to list files in the restricted directory
    print("\n1. Listing files in examples directory:")
    turn = await thread.run("List all files in the current directory")

    print("Response:")
    try:
        print(
            turn.final_response[:300] + "..."
            if len(turn.final_response) > 300
            else turn.final_response
        )
    except UnicodeEncodeError:
        safe_response = turn.final_response.encode("utf-8", errors="replace").decode(
            "utf-8"
        )
        print(
            safe_response[:300] + "..." if len(safe_response) > 300 else safe_response
        )

    # Try to access parent directory (should be restricted)
    print("\n2. Trying to access parent directory (should be restricted):")
    turn2 = await thread.run("Try to list files in the parent directory (..)")

    print("Response:")
    try:
        print(
            turn2.final_response[:300] + "..."
            if len(turn2.final_response) > 300
            else turn2.final_response
        )
    except UnicodeEncodeError:
        safe_response = turn2.final_response.encode("utf-8", errors="replace").decode(
            "utf-8"
        )
        print(
            safe_response[:300] + "..." if len(safe_response) > 300 else safe_response
        )


async def demonstrate_security_levels():
    """Demonstrate different security levels and their use cases."""
    print("\n" + "=" * 60)
    print("SECURITY LEVELS SUMMARY")
    print("=" * 60)

    print(
        """
SANDBOX MODES:

1. READ-ONLY:
   - Can read files and directories
   - Can execute commands to gather information
   - Cannot write files or modify system
   - Cannot access network
   - Use case: Code analysis, documentation generation

2. WORKSPACE-WRITE:
   - Can read and write files in workspace
   - Can execute commands within workspace
   - Cannot access system files outside workspace
   - Limited network access
   - Use case: Development tasks, file processing

3. DANGER-FULL-ACCESS:
   - Can read/write files anywhere (with permissions)
   - Can execute system commands
   - Can access network (web search, APIs)
   - Can modify system settings
   - Use case: System administration, web scraping
   - WARNING: Use with extreme caution!

WORKING DIRECTORY:
   - Restricts file operations to specified directory
   - Adds extra layer of security
   - Useful for isolating Codex to specific project folders

RECOMMENDATIONS:
   - Start with 'read-only' for analysis tasks
   - Use 'workspace-write' for development
   - Only use 'danger-full-access' when absolutely necessary
   - Always specify a working directory when possible
   - Test with restricted permissions first
"""
    )


async def main():
    """Main function to run all permission level tests."""
    print("Codex Python SDK - Permission Levels and Sandbox Modes")
    print("=" * 60)

    try:
        await test_read_only_sandbox()
        await test_workspace_write_sandbox()
        await test_full_access_sandbox()
        await test_custom_working_directory()
        await demonstrate_security_levels()

        print("\n" + "=" * 60)
        print("PERMISSION LEVELS TEST COMPLETED!")
        print("=" * 60)
        print(
            """
Key Takeaways:
- Use appropriate sandbox modes for your use case
- Start with restrictive permissions and increase as needed
- Working directory provides additional security
- Full access should be used sparingly and with caution
- Test your applications with different permission levels
"""
        )

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
