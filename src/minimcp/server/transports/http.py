import json
from collections.abc import Awaitable, Callable
from http import HTTPStatus

import mcp.types as types
from starlette.requests import Request
from starlette.responses import Response

from minimcp.server.types import Message

MEDIA_TYPE = "application/json; charset=utf-8"

JSON_RPC_TO_HTTP_STATUS_CODES: dict[int, HTTPStatus] = {
    types.PARSE_ERROR: HTTPStatus.BAD_REQUEST,
    types.INVALID_REQUEST: HTTPStatus.BAD_REQUEST,
    types.INVALID_PARAMS: HTTPStatus.BAD_REQUEST,
    types.METHOD_NOT_FOUND: HTTPStatus.NOT_FOUND,
    types.INTERNAL_ERROR: HTTPStatus.INTERNAL_SERVER_ERROR,
}


def get_response_http_status(response: Message | None) -> HTTPStatus:
    """
    Get the HTTP status code for a response.
    """

    # -- Notification --
    if response is None:  # This is a notification
        return HTTPStatus.ACCEPTED

    # -- Response --
    response_dict = json.loads(response)

    if "error" not in response_dict:
        return HTTPStatus.OK

    code = response_dict["error"].get("code", 0)
    return JSON_RPC_TO_HTTP_STATUS_CODES.get(code, HTTPStatus.INTERNAL_SERVER_ERROR)


async def starlette_http_transport(request: Request, handler: Callable[[Message], Awaitable[Message | None]]):
    msg = await request.body()
    msg_str = msg.decode("utf-8")

    response = await handler(msg_str)

    return Response(content=response, status_code=get_response_http_status(response), media_type=MEDIA_TYPE)
