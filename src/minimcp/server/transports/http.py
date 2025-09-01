import json
from collections.abc import Awaitable, Callable
from http import HTTPStatus

import mcp.types as types
from starlette.requests import Request
from starlette.responses import Response

from minimcp.server.types import Message, NoMessage

MEDIA_TYPE = "application/json; charset=utf-8"

JSON_RPC_TO_HTTP_STATUS_CODES: dict[int, HTTPStatus] = {
    types.PARSE_ERROR: HTTPStatus.BAD_REQUEST,
    types.INVALID_REQUEST: HTTPStatus.BAD_REQUEST,
    types.INVALID_PARAMS: HTTPStatus.BAD_REQUEST,
    types.METHOD_NOT_FOUND: HTTPStatus.NOT_FOUND,
    types.INTERNAL_ERROR: HTTPStatus.INTERNAL_SERVER_ERROR,
}


def http_status_from_message(response: Message) -> HTTPStatus:
    """
    Get the HTTP status code for a JSON-RPC message.
    """

    response_dict = json.loads(response)

    if "error" not in response_dict:
        return HTTPStatus.OK

    json_rpc_error_code = response_dict["error"].get("code", 0)
    return JSON_RPC_TO_HTTP_STATUS_CODES.get(json_rpc_error_code, HTTPStatus.INTERNAL_SERVER_ERROR)


async def starlette_http_transport(request: Request, handler: Callable[[Message], Awaitable[Message | NoMessage]]):
    msg = await request.body()
    msg_str = msg.decode("utf-8")

    response = await handler(msg_str)

    if isinstance(response, NoMessage):
        return Response(status_code=HTTPStatus.ACCEPTED)

    return Response(content=response, status_code=http_status_from_message(response), media_type=MEDIA_TYPE)
