import asyncio

import pytest

from codex_sdk.app_server import (
    ApprovalDecisions,
    AppServerClient,
    AppServerNotification,
    AppServerOptions,
    AppServerRequest,
    AppServerTurnSession,
)
from codex_sdk.exceptions import CodexError


class FakeAppServerClient:
    def __init__(self) -> None:
        self.notifications: asyncio.Queue = asyncio.Queue()
        self.requests: asyncio.Queue = asyncio.Queue()
        self.respond_calls = []

    async def next_notification(self):
        return await self.notifications.get()

    async def next_request(self):
        return await self.requests.get()

    async def respond(self, request_id, result=None, *, error=None) -> None:
        self.respond_calls.append((request_id, result, error))


@pytest.mark.asyncio
async def test_turn_session_auto_approves_requests():
    client = FakeAppServerClient()
    session = AppServerTurnSession(
        client,
        thread_id="thr_1",
        turn_id="turn_1",
        approvals=ApprovalDecisions(
            command_execution="accept",
            file_change="accept_for_session",
        ),
    )

    await session.start()
    await client.requests.put(
        AppServerRequest(
            id=1,
            method="item/commandExecution/requestApproval",
            params={"threadId": "thr_1", "turnId": "turn_1"},
        )
    )
    await client.requests.put(
        AppServerRequest(
            id=2,
            method="item/fileChange/requestApproval",
            params={"threadId": "thr_1", "turnId": "turn_1"},
        )
    )
    await client.notifications.put(
        AppServerNotification(
            method="turn/completed",
            params={"turn": {"id": "turn_1"}},
        )
    )

    final_turn = await session.wait()
    assert final_turn == {"id": "turn_1"}

    decisions = {call[0]: call[1] for call in client.respond_calls}
    assert decisions[1] == {"decision": "accept"}
    assert decisions[2] == {"decision": "acceptForSession"}


@pytest.mark.asyncio
async def test_turn_session_exposes_unhandled_requests():
    client = FakeAppServerClient()
    session = AppServerTurnSession(
        client, thread_id="thr_1", turn_id="turn_1", approvals=ApprovalDecisions()
    )

    await session.start()
    await client.requests.put(
        AppServerRequest(
            id=5,
            method="item/commandExecution/requestApproval",
            params={"threadId": "thr_1", "turnId": "turn_1"},
        )
    )
    await client.notifications.put(
        AppServerNotification(
            method="turn/completed",
            params={"turn": {"id": "turn_1"}},
        )
    )

    pending = await session.next_request()
    assert pending is not None
    assert pending.id == 5

    await session.wait()


@pytest.mark.asyncio
async def test_turn_session_execpolicy_amendment():
    client = FakeAppServerClient()
    session = AppServerTurnSession(
        client,
        thread_id="thr_1",
        turn_id="turn_2",
        approvals=ApprovalDecisions(
            command_execution="accept_with_execpolicy_amendment",
            execpolicy_amendment={"command": ["ls", "-la"]},
        ),
    )

    await session.start()
    await client.requests.put(
        AppServerRequest(
            id=7,
            method="item/commandExecution/requestApproval",
            params={"threadId": "thr_1", "turnId": "turn_2"},
        )
    )
    await client.notifications.put(
        AppServerNotification(method="turn/completed", params={"id": "turn_2"})
    )

    await session.wait()
    assert client.respond_calls[0][1] == {
        "decision": {
            "acceptWithExecpolicyAmendment": {
                "execpolicyAmendment": {"command": ["ls", "-la"]}
            }
        }
    }


@pytest.mark.asyncio
async def test_app_server_client_turn_session_requires_turn_id(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_turn_start(self, *_args, **_kwargs):
        return {"turn": {}}

    monkeypatch.setattr(AppServerClient, "turn_start", fake_turn_start)

    client = AppServerClient(AppServerOptions(auto_initialize=False))
    with pytest.raises(CodexError):
        await client.turn_session("thr_1", "hi")


@pytest.mark.asyncio
async def test_app_server_client_turn_session_starts_session(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_turn_start(self, *_args, **_kwargs):
        return {"turn": {"id": "turn_9"}}

    async def fake_start(self):
        return None

    monkeypatch.setattr(AppServerClient, "turn_start", fake_turn_start)
    monkeypatch.setattr(AppServerTurnSession, "start", fake_start)

    client = AppServerClient(AppServerOptions(auto_initialize=False))
    session = await client.turn_session("thr_1", "hi")
    assert session.turn_id == "turn_9"


@pytest.mark.asyncio
async def test_turn_session_context_manager_and_generators():
    client = FakeAppServerClient()
    async with AppServerTurnSession(client, thread_id="thr", turn_id="turn") as session:

        async def collect_notifications():
            seen = []
            async for note in session.notifications():
                seen.append(note.method)
            return seen

        async def collect_requests():
            seen = []
            async for req in session.requests():
                seen.append(req.method)
            return seen

        notes_task = asyncio.create_task(collect_notifications())
        reqs_task = asyncio.create_task(collect_requests())

        await client.requests.put(
            AppServerRequest(
                id=1,
                method="item/fileChange/requestApproval",
                params={"threadId": "thr", "turnId": "turn"},
            )
        )
        await client.notifications.put(
            AppServerNotification(
                method="turn/started", params={"turn": {"id": "turn"}}
            )
        )
        await client.notifications.put(
            AppServerNotification(
                method="turn/completed", params={"turn": {"id": "turn"}}
            )
        )

        await session.wait()
        assert await notes_task == ["turn/started", "turn/completed"]
        assert await reqs_task == ["item/fileChange/requestApproval"]


@pytest.mark.asyncio
async def test_turn_session_next_notification_and_double_close():
    client = FakeAppServerClient()
    session = AppServerTurnSession(client, thread_id="thr", turn_id="turn")
    await session.start()
    await client.notifications.put(
        AppServerNotification(method="turn/completed", params={"turn": {"id": "turn"}})
    )
    note = await session.next_notification()
    assert note is not None
    assert note.method == "turn/completed"
    await session.close()
    await session.close()


@pytest.mark.asyncio
async def test_turn_session_ignores_other_thread_requests():
    client = FakeAppServerClient()
    session = AppServerTurnSession(
        client,
        thread_id="thr",
        turn_id="turn",
        approvals=ApprovalDecisions(command_execution="accept"),
    )
    await session.start()
    await client.requests.put(
        AppServerRequest(
            id=9,
            method="item/commandExecution/requestApproval",
            params={"threadId": "other", "turnId": "turn"},
        )
    )
    await client.notifications.put(
        AppServerNotification(method="turn/completed", params={"turn": {"id": "turn"}})
    )
    pending = await session.next_request()
    assert pending is not None
    assert pending.id == 9


@pytest.mark.asyncio
async def test_turn_session_handles_none_sentinels():
    client = FakeAppServerClient()
    session = AppServerTurnSession(client, thread_id="thr", turn_id="turn")
    await session.start()
    await client.notifications.put(None)
    await client.requests.put(None)
    result = await session.wait()
    assert result is None


@pytest.mark.asyncio
async def test_turn_session_file_change_decision_none():
    client = FakeAppServerClient()
    session = AppServerTurnSession(
        client,
        thread_id="thr",
        turn_id="turn",
        approvals=ApprovalDecisions(command_execution="accept"),
    )
    await session.start()
    await client.requests.put(
        AppServerRequest(
            id=2,
            method="item/fileChange/requestApproval",
            params={"threadId": "thr", "turnId": "turn"},
        )
    )
    await client.notifications.put(
        AppServerNotification(method="turn/completed", params={"turn": {"id": "turn"}})
    )
    pending = await session.next_request()
    assert pending is not None
    assert pending.method == "item/fileChange/requestApproval"


def test_turn_session_is_turn_completed_false():
    client = FakeAppServerClient()
    session = AppServerTurnSession(client, thread_id="thr", turn_id="turn")
    note = AppServerNotification(method="turn/started", params={"turn": {"id": "turn"}})
    assert session._is_turn_completed(note) is False
