from collections.abc import Awaitable, Callable
from enum import Enum

Message = str

Send = Callable[[Message], Awaitable[None]]


class ResponseType(Enum):
    """
    Represents handler responses that are not JSON-RPC messages.
    """

    NOTIFICATION = "notification"  # Response to a client notification
    RESPONSE = "response"  # Response to a client request
