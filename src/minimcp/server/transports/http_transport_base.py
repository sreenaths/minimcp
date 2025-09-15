import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from http import HTTPStatus
from json.decoder import JSONDecodeError

import mcp.types as types
from anyio.streams.memory import MemoryObjectReceiveStream
from mcp.shared.version import SUPPORTED_PROTOCOL_VERSIONS

from minimcp.server import json_rpc
from minimcp.server.types import Message, NoMessage
from minimcp.utils.model import to_json


@dataclass
class HTTPResult:
    status_code: HTTPStatus
    content: Message | NoMessage | MemoryObjectReceiveStream[Message] | None = None
    media_type: str | None = None
    headers: Mapping[str, str] | None = None


MCP_PROTOCOL_VERSION_HEADER = "MCP-Protocol-Version"

CONTENT_TYPE_JSON = "application/json"

logger = logging.getLogger(__name__)


class HTTPTransportBase:
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
                HTTPStatus.NOT_ACCEPTABLE,
                types.INVALID_REQUEST,
                "Not Acceptable: Client must accept " + " and ".join(needed_content_types),
            )

        return None

    def _check_content_type(self, headers: Mapping[str, str]) -> HTTPResult | None:
        content_type = headers.get("Content-Type", "")
        content_type = content_type.split(";")[0].strip().lower()

        if content_type != CONTENT_TYPE_JSON:
            return self._build_error_result(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                types.INVALID_REQUEST,
                "Unsupported Media Type: Content-Type must be " + CONTENT_TYPE_JSON,
            )

        return None

    def _validate_protocol_version(self, headers: Mapping[str, str], body: str) -> HTTPResult | None:
        try:
            request_obj = json.loads(body)
            if isinstance(request_obj, dict) and request_obj.get("method") == "initialize":
                # Ignore protocol version validation for initialize request
                return None
        except JSONDecodeError:
            logger.debug("JSONDecodeError: Ignoring for the handler to return correct error response.")
            pass

        # If no protocol version provided, assume default version as per the specification
        # https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#protocol-version-header
        protocol_version = headers.get(MCP_PROTOCOL_VERSION_HEADER, types.DEFAULT_NEGOTIATED_VERSION)

        # Check if the protocol version is supported
        if protocol_version not in SUPPORTED_PROTOCOL_VERSIONS:
            supported_versions = ", ".join(SUPPORTED_PROTOCOL_VERSIONS)
            return self._build_error_result(
                HTTPStatus.BAD_REQUEST,
                types.INVALID_REQUEST,
                f"Bad Request: Unsupported protocol version: {protocol_version}. "
                + f"Supported versions: {supported_versions}",
            )

        return None

    def _validate_request_body(self, body: str) -> HTTPResult | None:
        try:
            request_obj = json.loads(body)
        except JSONDecodeError:
            return self._build_error_result(HTTPStatus.BAD_REQUEST, types.PARSE_ERROR, "Bad Request: Invalid JSON")

        if not isinstance(request_obj, dict):
            return self._build_error_result(
                HTTPStatus.BAD_REQUEST, types.PARSE_ERROR, "Bad Request: Invalid JSON - not a dictionary"
            )

        if "jsonrpc" not in request_obj:
            return self._build_error_result(
                HTTPStatus.BAD_REQUEST, types.INVALID_PARAMS, "Bad Request: Invalid JSON - Not a JSON-RPC message"
            )

        if request_obj["jsonrpc"] != json_rpc.JSON_RPC_VERSION:
            return self._build_error_result(
                HTTPStatus.BAD_REQUEST, types.INVALID_PARAMS, "Bad Request: Invalid JSON - Not a JSON-RPC 2.0 message"
            )

        return None

    def _build_error_result(self, status_code: HTTPStatus, err_code: int, err_msg: str) -> HTTPResult:
        """
        Build an error result with the given status code, JSON-RPC error code, and error message.
        """
        err = ValueError(err_msg)
        content = to_json(json_rpc.build_error_message(err_code, "", err))

        logger.debug("Building error result with HTTP status code %s and error message %s", status_code, err_msg)
        return HTTPResult(status_code, content, CONTENT_TYPE_JSON)
