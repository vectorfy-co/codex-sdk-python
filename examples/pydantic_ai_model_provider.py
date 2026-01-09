"""
Example: use Codex as a PydanticAI model provider.

This lets PydanticAI manage tools + validation, while Codex generates tool-call
plans and final responses via `codex exec --output-schema`.

Requires:
  uv add "codex-sdk-python[pydantic-ai]"
"""

from __future__ import annotations

from pydantic_ai import Agent, Tool

from codex_sdk.integrations.pydantic_ai_model import CodexModel
from codex_sdk.options import ThreadOptions


def add(a: int, b: int) -> int:
    return a + b


def main() -> None:
    model = CodexModel(
        thread_options=ThreadOptions(
            # Pick a model family that doesn't aggressively auto-use Codex tools.
            model="gpt-5",
            sandbox_mode="read-only",
            skip_git_repo_check=True,
        )
    )

    agent = Agent(
        model,
        tools=[Tool(add)],
    )

    result = agent.run_sync("What's 19 + 23? Use the add tool.")
    print(result.output)


if __name__ == "__main__":
    main()
