from collections.abc import Awaitable, Callable
from enum import Enum

Message = str


class NoMessage(Enum):
    """
    Represents handler responses without any message.
    https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#sending-messages-to-the-server
    """

    NOTIFICATION = "notification"  # Response to a client notification
    RESPONSE = "response"  # Response to a client request


Send = Callable[[Message], Awaitable[None]]
