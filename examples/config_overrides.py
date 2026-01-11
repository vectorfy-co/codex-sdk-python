"""Use config overrides for analytics/notify and OTEL opt-in.

Note: OTEL exporters are configured in ~/.codex/config.toml. This example toggles
analytics and notify via CLI overrides; you still need exporters configured in
config.toml for metrics to leave the machine.
"""

import asyncio
from codex_sdk import Codex, CodexOptions, ThreadOptions


async def main() -> None:
    codex = Codex(
        CodexOptions(
            config_overrides={
                "analytics.enabled": True,
                "notify": ["python3", "/absolute/path/to/examples/notify_hook.py"],
            }
        )
    )
    thread = codex.start_thread(ThreadOptions(model="gpt-5-codex-high"))
    turn = await thread.run("Summarize the repository and suggest next steps.")
    print(turn.final_response)


if __name__ == "__main__":
    asyncio.run(main())
