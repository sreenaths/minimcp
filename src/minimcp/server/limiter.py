import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from anyio import CancelScope, CapacityLimiter, current_time

logger = logging.getLogger(__name__)


class TimeLimiter:
    _timeout: float
    _scope: CancelScope

    def __init__(self, timeout: float):
        self._timeout = float(timeout)
        self._scope = CancelScope()
        self.reset()

    def reset(self) -> None:
        self._scope.deadline = current_time() + self._timeout

    def __enter__(self):
        self._scope.__enter__()
        return self

    def __exit__(self, *args):
        return self._scope.__exit__(*args)


class Limiter:
    _idle_timeout: int

    def __init__(self, idle_timeout: int, max_concurrency: int) -> None:
        self._idle_timeout = idle_timeout
        self._capacity_limiter = CapacityLimiter(max_concurrency)

    @asynccontextmanager
    async def __call__(self) -> AsyncGenerator[TimeLimiter, None]:
        async with self._capacity_limiter:
            with TimeLimiter(self._idle_timeout) as time_limiter:
                yield time_limiter
