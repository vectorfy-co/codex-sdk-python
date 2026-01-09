#!/usr/bin/env python3
"""
PydanticAI tool handoff example.

This example registers Codex as a PydanticAI tool and asks the agent to delegate a coding task.

Requirements:
  - uv add "codex-sdk-python[pydantic-ai]"
  - codex login   (or set CODEX_API_KEY)
"""

import asyncio

from pydantic_ai import Agent

from codex_sdk import ThreadOptions
from codex_sdk.integrations.pydantic_ai import codex_handoff_tool


async def main() -> None:
    tool = codex_handoff_tool(
        thread_options=ThreadOptions(
            sandbox_mode="workspace-write",
            skip_git_repo_check=True,
            working_directory=".",
        ),
        include_items=True,
        items_limit=20,
    )

    agent = Agent(
        "openai:gpt-5",
        tools=[tool],
        system_prompt=(
            "You can delegate implementation details to the `codex_handoff` tool. "
            "Use it when you need repository-aware edits, command execution, or detailed patches."
        ),
    )

    result = await agent.run(
        "Use the codex_handoff tool to scan this repository and suggest one small, safe DX improvement."
    )
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
