"""Handle approval requests from app-server turns."""

import asyncio

from codex_sdk import AppServerClient, AppServerOptions, CodexAppServerError


async def main() -> None:
    async with AppServerClient(AppServerOptions()) as app:
        start = await app.thread_start(model="gpt-5-codex-high", cwd=".")
        thread_id = start["thread"]["id"]

        await app.turn_start(thread_id, "List the repo and explain the structure.")

        async for req in app.requests():
            if req is None:
                break

            try:
                if req.method == "item/commandExecution/requestApproval":
                    await app.respond(req.id, {"decision": "accept"})
                elif req.method == "item/fileChange/requestApproval":
                    await app.respond(req.id, {"decision": "accept"})
            except CodexAppServerError as exc:
                print("Approval error:", exc)


if __name__ == "__main__":
    asyncio.run(main())
