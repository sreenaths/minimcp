import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

import anyio
import mcp.types as types
from anyio.abc import TaskGroup, TaskStatus
from anyio.streams.memory import MemoryObjectReceiveStream
from mcp.shared.version import SUPPORTED_PROTOCOL_VERSIONS

from minimcp.server import json_rpc
from minimcp.server.types import Message, NoMessage, Send
from minimcp.utils.task_status_wrapper import TaskStatusWrapper

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


HandlerTaskResponse = Message | NoMessage | MemoryObjectReceiveStream[Message]


# Context manager for suppressing expected stream errors
suppress_stream_errors = suppress(anyio.BrokenResourceError, anyio.ClosedResourceError)


# TODO: Add resumability based on Last-Event-ID header
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

    async def _runner(
        self, handler: StreamableHTTPRequestHandler, body: str, task_status: TaskStatus[HandlerTaskResponse]
    ):
        task_status_wrapper = TaskStatusWrapper(task_status)

        send_stream, recv_stream = anyio.create_memory_object_stream[Message](1)
        await self._stack.enter_async_context(recv_stream)

        async def send(msg: Message) -> None:
            task_status_wrapper.set(recv_stream)
            with suppress_stream_errors:
                await send_stream.send(msg)

        async with send_stream:
            response = await handler(body, send)

            if task_status_wrapper.set(response):
                # recv_stream will not be used, close it to free resources
                with suppress_stream_errors:
                    await recv_stream.aclose()
            elif not isinstance(response, NoMessage):
                # Stream was set as status, send the response to the stream
                with suppress_stream_errors:
                    await send_stream.send(response)

        # Exception Handling:
        # 1. Exceptions before task_status is set:
        #    - Propagate directly to the caller of self._tg.start(...)
        #    - This includes setup errors (stream creation, context manager entry)
        # 2. Exceptions inside handler:
        #    - Handler exceptions should be caught and handled by the handler itself
        #    - Unhandled handler exceptions must cancel the transport and propagate
        # 3. Exceptions after handler returns:
        #    - Specific stream errors are handled gracefully
        #    - Others are unexpected, hence they must cancel the transport and propagate

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

        try:
            response = await self._tg.start(self._runner, handler, body)
        except Exception as e:
            # Unexpected exception in _runner after task started - transport is now broken
            # Cancel the transport to ensure clean state
            await self.aclose()
            raise RuntimeError("Transport cancelled due to unexpected exception in runner") from e

        if isinstance(response, MemoryObjectReceiveStream):
            return HTTPResult(HTTPStatus.OK, response, headers=SSE_HEADERS)

        if isinstance(response, NoMessage):
            return HTTPResult(HTTPStatus.ACCEPTED)

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
