"""Run a single app-server turn with automatic approvals."""

import asyncio

from codex_sdk import AppServerClient, AppServerOptions, ApprovalDecisions


async def main() -> None:
    async with AppServerClient(AppServerOptions()) as app:
        thread = await app.thread_start(model="gpt-5-codex-high", cwd=".")
        thread_id = thread["thread"]["id"]

        session = await app.turn_session(
            thread_id,
            "Run tests and summarize failures.",
            approvals=ApprovalDecisions(
                command_execution="accept",
                file_change="decline",
            ),
        )

        async for notification in session.notifications():
            print(notification.method, notification.params)

        final_turn = await session.wait()
        print(final_turn)


if __name__ == "__main__":
    asyncio.run(main())
