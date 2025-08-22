from collections.abc import Awaitable, Callable
from typing import Any

Message = dict[str, Any]

MessageSender = Callable[[Message | None], Awaitable[None]]

TransportHandler = Callable[[Message, MessageSender], Awaitable[Message | None]]
