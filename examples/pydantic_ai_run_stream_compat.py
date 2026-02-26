"""Minimal harness reproducing Agent.run_stream + app-server CodexModel compatibility."""

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
        return {"thread": {"id": "thread-compat-stream"}}

    async def turn_session(self, _thread_id, _input, **_params):
        return _FakeAppServerTurnSession(
            notifications=[
                AppServerNotification(
                    method="item/updated",
                    params={
                        "item": {
                            "id": "msg-1",
                            "type": "agent_message",
                            "text": "stream ",
                        }
                    },
                ),
                AppServerNotification(
                    method="item/updated",
                    params={
                        "item": {
                            "id": "msg-1",
                            "type": "agent_message",
                            "text": "stream ok",
                        }
                    },
                ),
            ],
            final_turn={
                "id": "turn-compat-stream",
                "usage": {"inputTokens": 1, "outputTokens": 1},
                "finalResponse": "stream ok",
            },
        )


async def _main() -> None:
    agent = Agent(model=CodexModel(app_server=_FakeAppServerClient()), output_type=str)
    try:
        async with agent.run_stream("ping") as result:
            output = await result.get_output()
    except Exception as exc:
        raise AssertionError(
            f"run_stream failed with {type(exc).__name__}: {exc}"
        ) from exc

    assert output == "stream ok"
    print("run_stream compatible:", output)


if __name__ == "__main__":
    asyncio.run(_main())
