from collections.abc import Awaitable, Callable, Mapping
from http import HTTPStatus

from minimcp.server.transports.http_transport_base import CONTENT_TYPE_JSON, HTTPResult, HTTPTransportBase
from minimcp.server.types import Message, NoMessage

HTTPRequestHandler = Callable[[Message], Awaitable[Message | NoMessage]]


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
        if result := self._check_accept_headers(headers, {CONTENT_TYPE_JSON}):
            return result
        if result := self._check_content_type(headers):
            return result
        if result := self._validate_protocol_version(headers, body):
            return result

        response = await handler(body)

        if isinstance(response, NoMessage):
            return HTTPResult(HTTPStatus.ACCEPTED)

        status_code = self._get_status_code(response)
        return HTTPResult(status_code, response, CONTENT_TYPE_JSON)
