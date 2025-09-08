import json
from collections.abc import Mapping
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

import mcp.types as types
from mcp.shared.version import SUPPORTED_PROTOCOL_VERSIONS

from minimcp.server import json_rpc
from minimcp.server.types import Message


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

MCP_PROTOCOL_VERSION_HEADER = "MCP-Protocol-Version"

CONTENT_TYPE_JSON = "application/json"


class HTTPTransportBase:
    def _get_status_code(self, msg: Message) -> HTTPStatus:
        """
        Get the HTTP status code for a JSON-RPC message.
        """
        response_dict = json.loads(msg)

        if not isinstance(response_dict, dict):
            return HTTPStatus.INTERNAL_SERVER_ERROR

        if "error" not in response_dict:
            return HTTPStatus.OK

        json_rpc_error_code = response_dict["error"].get("code", 0)
        return JSON_RPC_TO_HTTP_STATUS_CODES.get(json_rpc_error_code, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_unsupported_request(self, supported_methods: set[str]) -> HTTPResult:
        headers = {
            "Content-Type": CONTENT_TYPE_JSON,
            "Allow": ", ".join(supported_methods),
        }

        return HTTPResult(HTTPStatus.METHOD_NOT_ALLOWED, headers=headers)

    def _check_accept_headers(self, headers: Mapping[str, str], needed_content_types: set[str]) -> HTTPResult | None:
        accept_header = headers.get("Accept", "")
        accepted_types = [t.split(";")[0].strip().lower() for t in accept_header.split(",")]

        if not needed_content_types.issubset(accepted_types):
            return self._build_error_result(
                HTTPStatus.NOT_ACCEPTABLE, "Not Acceptable: Client must accept " + " and ".join(needed_content_types)
            )

        return None

    def _check_content_type(self, headers: Mapping[str, str]) -> HTTPResult | None:
        content_type = headers.get("Content-Type", "")
        content_type = content_type.split(";")[0].strip().lower()

        if content_type != CONTENT_TYPE_JSON:
            return self._build_error_result(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Unsupported Media Type: Content-Type must be " + CONTENT_TYPE_JSON
            )

        return None

    def _validate_protocol_version(self, headers: Mapping[str, str], body: str) -> HTTPResult | None:
        request_obj = json.loads(body)
        if isinstance(request_obj, dict) and request_obj.get("method") == "initialize":
            # Ignore protocol version validation for initialize request
            return None

        # If no protocol version provided, assume default version as per the specification
        # https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#protocol-version-header
        protocol_version = headers.get(MCP_PROTOCOL_VERSION_HEADER, types.DEFAULT_NEGOTIATED_VERSION)

        # Check if the protocol version is supported
        if protocol_version not in SUPPORTED_PROTOCOL_VERSIONS:
            supported_versions = ", ".join(SUPPORTED_PROTOCOL_VERSIONS)
            return self._build_error_result(
                HTTPStatus.BAD_REQUEST,
                f"Bad Request: Unsupported protocol version: {protocol_version}. "
                + f"Supported versions: {supported_versions}",
            )

        return None

    def _build_error_result(self, status_code: HTTPStatus, err_msg: str) -> HTTPResult:
        err = ValueError(err_msg)
        content = json_rpc.build_error_message(types.INVALID_REQUEST, "", err)
        return HTTPResult(status_code, content, CONTENT_TYPE_JSON)
