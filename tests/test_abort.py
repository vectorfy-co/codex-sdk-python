import asyncio

import pytest

from codex_sdk.abort import AbortController


@pytest.mark.asyncio
async def test_abort_signal_wait_completes():
    controller = AbortController()
    waiter = asyncio.create_task(controller.signal.wait())
    await asyncio.sleep(0)
    controller.abort("stop")
    await asyncio.wait_for(waiter, timeout=1.0)
    assert controller.signal.aborted is True
    assert controller.signal.reason == "stop"
