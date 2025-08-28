import json
from http import HTTPStatus

import mcp.types as types

from minimcp.server.types import Message

MEDIA_TYPE = "application/json; charset=utf-8"


def get_response_http_code(response: Message | None) -> int:
    if response is None:  # This is a notification
        return HTTPStatus.ACCEPTED

    response_dict = json.loads(response)

    if "error" not in response_dict:
        return HTTPStatus.OK

    code = response_dict["error"].get("code")
    match code:
        case types.PARSE_ERROR | types.INVALID_REQUEST | types.INVALID_PARAMS:
            return HTTPStatus.BAD_REQUEST

        case types.METHOD_NOT_FOUND:
            return HTTPStatus.NOT_FOUND

        case types.INTERNAL_ERROR:
            return HTTPStatus.INTERNAL_SERVER_ERROR

    return HTTPStatus.INTERNAL_SERVER_ERROR
