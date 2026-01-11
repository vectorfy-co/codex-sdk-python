"""Run a turn while reacting to streamed events."""

import asyncio

from codex_sdk import Codex, ThreadHooks


async def main() -> None:
    codex = Codex()
    thread = codex.start_thread()

    async def on_event(event) -> None:
        print("event:", event.type)

    hooks = ThreadHooks(
        on_event=on_event,
        on_item_type={
            "command_execution": lambda item: print("command:", item.command),
            "file_change": lambda item: print("file change:", item.status),
        },
    )

    turn = await thread.run_with_hooks(
        "Run the test suite and summarize any failures.", hooks=hooks
    )
    print(turn.final_response)


if __name__ == "__main__":
    asyncio.run(main())
