"""Explicit skill invocation via app-server input."""

import asyncio

from codex_sdk import AppServerClient, AppServerOptions


async def main() -> None:
    async with AppServerClient(AppServerOptions()) as app:
        start = await app.thread_start(model="gpt-5-codex-high", cwd=".")
        thread_id = start["thread"]["id"]

        await app.turn_start(
            thread_id,
            [
                {"type": "text", "text": "Use $my-skill and summarize the repo."},
                {"type": "skill", "name": "my-skill", "path": "/path/to/SKILL.md"},
            ],
        )

        async for notification in app.notifications():
            print(notification.method)
            if notification.method == "turn/completed":
                break


if __name__ == "__main__":
    asyncio.run(main())
