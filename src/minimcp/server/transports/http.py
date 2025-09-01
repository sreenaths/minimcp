import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

import mcp.types as types

from minimcp.server.types import Message, NoMessage

CONTENT_TYPE_JSON = "application/json"


@dataclass
class HTTPResult:
    status_code: HTTPStatus
    content: Any | None = None
    media_type: str | None = None
    headers: Mapping[str, str] | None = None


JSON_RPC_TO_HTTP_STATUS_CODES: dict[int, HTTPStatus] = {
    types.PARSE_ERROR: HTTPStatus.BAD_REQUEST,
    types.INVALID_REQUEST: HTTPStatus.BAD_REQUEST,
    types.INVALID_PARAMS: HTTPStatus.BAD_REQUEST,
    types.METHOD_NOT_FOUND: HTTPStatus.NOT_FOUND,
    types.INTERNAL_ERROR: HTTPStatus.INTERNAL_SERVER_ERROR,
}

HTTPRequestHandler = Callable[[Message], Awaitable[Message | NoMessage]]


class HTTPTransport:
    def _get_status_code(self, msg: Message) -> HTTPStatus:
        """
        Get the HTTP status code for a JSON-RPC message.
        """

        response_dict = json.loads(msg)

        if "error" not in response_dict:
            return HTTPStatus.OK

        json_rpc_error_code = response_dict["error"].get("code", 0)
        return JSON_RPC_TO_HTTP_STATUS_CODES.get(json_rpc_error_code, HTTPStatus.INTERNAL_SERVER_ERROR)

    async def dispatch(
        self, handler: HTTPRequestHandler, method: str, headers: Mapping[str, str], body: str
    ) -> HTTPResult:
        if method == "POST":
            # _check_accept_headers
            # _check_content_type
            # _validate_protocol_version header

            response = await handler(body)

            if isinstance(response, NoMessage):
                return HTTPResult(HTTPStatus.ACCEPTED)

            status_code = self._get_status_code(response)
            return HTTPResult(status_code, response, CONTENT_TYPE_JSON)
        else:
            return HTTPResult(HTTPStatus.METHOD_NOT_ALLOWED)


HTTP_TRANSPORT = HTTPTransport()
