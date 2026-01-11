"""Basic Codex app-server usage with notifications."""

import asyncio

from codex_sdk import AppServerClient, AppServerOptions


async def main() -> None:
    async with AppServerClient(AppServerOptions()) as app:
        thread_resp = await app.thread_start(model="gpt-5-codex-high", cwd=".")
        thread_id = thread_resp["thread"]["id"]

        await app.turn_start(
            thread_id,
            "Summarize the changes in this repository and suggest next steps.",
        )

        async for notification in app.notifications():
            print(notification.method, notification.params)
            if notification.method == "turn/completed":
                break


if __name__ == "__main__":
    asyncio.run(main())
