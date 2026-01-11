"""Fork a thread using the app-server API."""

import asyncio

from codex_sdk import AppServerClient, AppServerOptions


async def main() -> None:
    async with AppServerClient(AppServerOptions()) as app:
        start = await app.thread_start(model="gpt-5-codex-high", cwd=".")
        thread_id = start["thread"]["id"]

        await app.turn_start(thread_id, "Write a short plan for refactoring.")

        async for notification in app.notifications():
            if notification.method == "turn/completed":
                break

        forked = await app.thread_fork(thread_id)
        print("Forked thread:", forked["thread"]["id"])


if __name__ == "__main__":
    asyncio.run(main())
