from collections.abc import Awaitable, Callable

from minimcp.server.responder import Responder
from minimcp.server.types import Message

TransportRequestHandler = Callable[[Message, Responder], Awaitable[Message | None]]
