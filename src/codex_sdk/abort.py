"""AbortController and AbortSignal helpers for cancelling Codex runs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional, Union

AbortReason = Union[str, BaseException]


@dataclass
class AbortSignal:
    """Signal object used to abort a running operation."""

    _event: asyncio.Event
    _reason: Optional[AbortReason] = None

    @property
    def aborted(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> Optional[AbortReason]:
        return self._reason

    async def wait(self) -> None:
        await self._event.wait()


class AbortController:
    """Controller used to trigger cancellation for an AbortSignal."""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self.signal = AbortSignal(self._event)

    def abort(self, reason: Optional[AbortReason] = None) -> None:
        self.signal._reason = reason
        self._event.set()
