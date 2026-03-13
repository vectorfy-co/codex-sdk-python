#!/usr/bin/env python3
"""
Example: stream live text from Codex through the PydanticAI model provider.

This is the user-facing streaming variant of the CodexModel integration. It is
useful for terminal UIs and web UIs where you want to render the model output
incrementally as Codex emits agent-message updates.

Requires:
  uv add "codex-sdk-python[pydantic-ai]"
  codex login   (or set CODEX_API_KEY)
"""

from __future__ import annotations

import asyncio

from pydantic_ai import Agent

from codex_sdk.integrations.pydantic_ai_model import CodexModel
from codex_sdk.options import ThreadOptions


async def main() -> None:
    model = CodexModel(
        thread_options=ThreadOptions(
            model="gpt-5",
            sandbox_mode="read-only",
            skip_git_repo_check=True,
        )
    )

    agent = Agent(model, output_type=str)

    async with agent.run_stream(
        "In two short sentences, explain why the sky appears blue."
    ) as result:
        async for chunk in result.stream_text(delta=True, debounce_by=None):
            print(chunk, end="", flush=True)
        print()

        final_output = await result.get_output()
        print("\nFinal output:", final_output)


if __name__ == "__main__":
    asyncio.run(main())
