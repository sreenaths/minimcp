import logging
from collections.abc import Awaitable, Callable
from operator import attrgetter

import mcp.types as types

from minimcp.server import json_rpc
from minimcp.server.types import Message
from minimcp.utils.model import to_json

logger = logging.getLogger(__name__)


class Responder:
    _request: Message
    _progress_token: types.ProgressToken | None

    _send: Callable[[Message], Awaitable[None]]

    def __init__(self, request: Message, send: Callable[[Message], Awaitable[None]]):
        self._request = request
        self._progress_token = self._get_progress_token(request)

        self._send = send

    def _get_progress_token(self, request: Message) -> types.ProgressToken | None:
        try:
            client_request = types.ClientRequest.model_validate(request)
            return attrgetter("params.meta.progressToken")(client_request.root)
        except Exception as e:
            logger.debug("Failed to get progress token: %s", e)
            return None

    async def report_progress(
        self, progress: float, total: float | None = None, message: str | None = None
    ) -> types.ProgressToken | None:
        """Report progress for the current operation.

        Args:
            progress: Current progress value e.g. 24
            total: Optional total value e.g. 100
            message: Optional message to display to the
            user

        Returns:
            The progress token if notification was sent. If not None.
        """

        if self._progress_token is None:
            return None

        notification = types.ServerNotification(
            types.ProgressNotification(
                method="notifications/progress",  # TODO: Remove once python-sdk/pull/1292 is merged.
                params=types.ProgressNotificationParams(
                    progressToken=self._progress_token,
                    progress=progress,
                    total=total,
                    message=message,
                ),
            )
        )

        await self.send_notification(notification)

        return self._progress_token

    async def send_notification(self, notification: types.ServerNotification) -> None:
        """
        Send a notification, which is a one-way message that does not expect
        a response.
        """
        logger.debug("Sending notification: %s", notification)
        rpc_msg = json_rpc.build_notification_message(notification)

        # Just call the sender with the message and let transport layer handle the rest.
        await self._send(to_json(rpc_msg))
