import json
import logging
from operator import attrgetter

import mcp.types as types

from minimcp.server import json_rpc
from minimcp.server.limiter import TimeLimiter
from minimcp.server.types import Message, Send
from minimcp.utils.model import to_json

logger = logging.getLogger(__name__)


class Responder:
    _request: Message
    _progress_token: types.ProgressToken | None
    _time_limiter: TimeLimiter

    _send: Send

    def __init__(self, request: Message, send: Send, time_limiter: TimeLimiter):
        self._request = request
        self._progress_token = self._get_progress_token(request)

        self._send = send
        self._time_limiter = time_limiter

    def _get_progress_token(self, request: Message) -> types.ProgressToken | None:
        try:
            client_request = types.ClientRequest.model_validate(json.loads(request))
            return attrgetter("params.meta.progressToken")(client_request.root)
        except Exception:
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
            logger.warning("report_progress failed: Progress token is not available.")
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

        # Reset time limiter
        self._time_limiter.reset()

        # Just call the sender with the message and let transport layer handle the rest.
        await self._send(to_json(rpc_msg))
