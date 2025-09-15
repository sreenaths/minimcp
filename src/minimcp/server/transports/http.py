import logging
from collections.abc import Awaitable, Callable, Mapping
from http import HTTPStatus

from minimcp.server.transports.http_transport_base import CONTENT_TYPE_JSON, HTTPResult, HTTPTransportBase
from minimcp.server.types import Message, NoMessage

HTTPRequestHandler = Callable[[Message], Awaitable[Message | NoMessage]]

logger = logging.getLogger(__name__)


class HTTPTransport(HTTPTransportBase):
    async def dispatch(
        self, handler: HTTPRequestHandler, method: str, headers: Mapping[str, str], body: str
    ) -> HTTPResult:
        if method == "POST":
            return await self._handle_post_request(handler, headers, body)
        else:
            return self._handle_unsupported_request({"POST"})

    async def _handle_post_request(
        self, handler: HTTPRequestHandler, headers: Mapping[str, str], body: str
    ) -> HTTPResult:
        logger.debug("Handling POST request. Headers: %s, Body: %s", headers, body)

        if result := self._check_accept_headers(headers, {CONTENT_TYPE_JSON}):
            return result
        if result := self._check_content_type(headers):
            return result
        if result := self._validate_protocol_version(headers, body):
            return result
        if result := self._validate_request_body(body):
            return result

        response = await handler(body)
        logger.debug("Handling completed. Response: %s", response)

        if isinstance(response, NoMessage):
            return HTTPResult(HTTPStatus.ACCEPTED)

        return HTTPResult(HTTPStatus.OK, response, CONTENT_TYPE_JSON)
