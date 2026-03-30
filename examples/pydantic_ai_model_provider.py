"""
Example: use Codex as a PydanticAI model provider.

This lets PydanticAI manage tools + validation, while Codex generates tool-call
plans and final responses through the app-server-backed CodexModel runtime.

Requires:
  uv add "codex-sdk-python[pydantic-ai]"
  codex login   (or set CODEX_API_KEY)
"""

from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai import Agent, Tool

from codex_sdk import ThreadHooks
from codex_sdk.integrations.pydantic_ai_model import CodexModel
from codex_sdk.options import ThreadOptions


class MathAnswer(BaseModel):
    result: int
    explanation: str


def add(a: int, b: int) -> int:
    return a + b


def main() -> None:
    model = CodexModel(
        thread_options=ThreadOptions(
            # Pick a model family that doesn't aggressively auto-use Codex tools.
            model="gpt-5.4",
            sandbox_mode="read-only",
            skip_git_repo_check=True,
        ),
        hooks=ThreadHooks(
            on_turn_started=lambda _event: print("[codex] turn started"),
            on_turn_completed=lambda event: print(
                f"[codex] output tokens: {event.usage.output_tokens}"
            ),
        ),
    )

    agent = Agent(
        model,
        tools=[Tool(add)],
        output_type=MathAnswer,
    )

    result = agent.run_sync(
        "What's 19 + 23? Use the add tool and return a structured answer.",
        model_settings={"thinking": "low"},
    )
    print(result.output.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
