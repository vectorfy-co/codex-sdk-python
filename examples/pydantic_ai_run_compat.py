"""Minimal harness for non-stream Agent.run parity with app-server CodexModel."""

import asyncio

from pydantic_ai import Agent

from codex_sdk.app_server import AppServerNotification
from codex_sdk.integrations.pydantic_ai_model import CodexModel


class _FakeAppServerTurnSession:
    def __init__(self, notifications, final_turn):
        self._notifications = list(notifications)
        self._final_turn = dict(final_turn)

    async def notifications(self):
        for notification in self._notifications:
            yield notification

    async def wait(self):
        return self._final_turn


class _FakeAppServerClient:
    async def start(self):
        return None

    async def close(self):
        return None

    async def thread_start(self, **_params):
        return {"thread": {"id": "thread-compat-run"}}

    async def turn_session(self, _thread_id, _input, **_params):
        return _FakeAppServerTurnSession(
            notifications=[
                AppServerNotification(
                    method="item/updated",
                    params={
                        "item": {
                            "id": "msg-1",
                            "type": "agent_message",
                            "text": "run ok",
                        }
                    },
                )
            ],
            final_turn={
                "id": "turn-compat-run",
                "usage": {"inputTokens": 1, "outputTokens": 1},
                "finalResponse": "run ok",
            },
        )


async def _main() -> None:
    agent = Agent(model=CodexModel(app_server=_FakeAppServerClient()), output_type=str)
    result = await agent.run("ping")
    assert result.output == "run ok"
    print("run compatible:", result.output)


if __name__ == "__main__":
    asyncio.run(_main())
