import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

import anyio
import mcp.types as types
from anyio.abc import TaskGroup
from anyio.streams.memory import MemoryObjectReceiveStream
from mcp.shared.version import SUPPORTED_PROTOCOL_VERSIONS

from minimcp.server import json_rpc
from minimcp.server.types import Message, NoMessage, Send

logger = logging.getLogger(__name__)


MCP_PROTOCOL_VERSION_HEADER = "MCP-Protocol-Version"

CONTENT_TYPE_JSON = "application/json"
CONTENT_TYPE_SSE = "text/event-stream"

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "Content-Type": CONTENT_TYPE_SSE,
}


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

StreamableHTTPRequestHandler = Callable[[Message, Send], Awaitable[Message | NoMessage]]


class StreamableHTTPTransport:
    _stack: AsyncExitStack
    _tg: TaskGroup | None

    def __init__(self):
        self._stack = AsyncExitStack()
        self._tg = None

    async def start(self) -> "StreamableHTTPTransport":
        return await self.__aenter__()

    async def aclose(self):
        return await self.__aexit__(None, None, None)

    async def __aenter__(self) -> "StreamableHTTPTransport":
        await self._stack.__aenter__()
        self._tg = await self._stack.enter_async_context(anyio.create_task_group())
        return self

    async def __aexit__(self, exc_type, exc, tb):
        result = await self._stack.__aexit__(exc_type, exc, tb)
        self._tg = None
        self._stack = AsyncExitStack()
        return result

    async def dispatch(
        self, handler: StreamableHTTPRequestHandler, method: str, headers: Mapping[str, str], body: str
    ) -> HTTPResult:
        if method == "POST":
            return await self._handle_post_request(handler, headers, body)
        else:
            return self._handle_unsupported_request({"POST"})

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

    async def _task_runner(
        self, handler: StreamableHTTPRequestHandler, body: str, task_status=anyio.TASK_STATUS_IGNORED
    ):
        send_stream, recv_stream = anyio.create_memory_object_stream[Message | NoMessage](1)
        try:
            await self._stack.enter_async_context(send_stream)
            await self._stack.enter_async_context(recv_stream)

            streaming = False

            async def send(msg: Message):
                nonlocal streaming
                if not streaming:
                    streaming = True
                    task_status.started(recv_stream)
                await send_stream.send(msg)

            response = await handler(body, send)

            if streaming:
                await send_stream.send(response)
            else:
                task_status.started(response)
        except Exception as e:
            logger.error("Error in task runner: %s", e)
            # Pre-start failures propagate to self._tg.start(...) and raises
            # Post-start failures, must be handled in handler
            # All unhandled exceptions cancels the TaskGroup and propagates
            raise
        finally:
            await send_stream.aclose()

    async def _handle_post_request(
        self, handler: StreamableHTTPRequestHandler, headers: Mapping[str, str], body: str
    ) -> HTTPResult:
        if result := self._check_accept_headers(headers, {CONTENT_TYPE_JSON, CONTENT_TYPE_SSE}):
            return result
        if result := self._check_content_type(headers):
            return result
        if result := self._validate_protocol_version(headers, body):
            return result

        if self._tg is None:
            raise RuntimeError("StreamableHTTPTransport was not started")

        response = await self._tg.start(self._task_runner, handler, body)

        if isinstance(response, NoMessage):
            return HTTPResult(HTTPStatus.ACCEPTED)

        if isinstance(response, MemoryObjectReceiveStream):
            return HTTPResult(HTTPStatus.OK, response, headers=SSE_HEADERS)

        status_code = self._get_status_code(response)
        return HTTPResult(status_code, response, CONTENT_TYPE_JSON)

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
