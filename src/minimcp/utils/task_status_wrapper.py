from typing import Generic, TypeVar

from anyio.abc import TaskStatus

T = TypeVar("T")


class TaskStatusWrapper(Generic[T]):
    _inner: TaskStatus[T]
    _is_set: bool

    def __init__(self, inner: TaskStatus[T]) -> None:
        self._inner = inner
        self._is_set = False

    def set(self, value: T) -> bool:
        # Not using a lock as set is synchronous and not likely to be called concurrently
        # Could add a threading lock, but could be overkill.
        if not self._is_set:
            self._is_set = True
            self._inner.started(value)
            return True
        return False

    @property
    def is_set(self) -> bool:
        return self._is_set
