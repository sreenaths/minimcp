import logging
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack, suppress
from http import HTTPStatus

import anyio
import mcp.types as types
from anyio.abc import TaskGroup, TaskStatus
from anyio.streams.memory import MemoryObjectReceiveStream

from minimcp.server import json_rpc
from minimcp.server.transports.http_transport_base import CONTENT_TYPE_JSON, HTTPResult, HTTPTransportBase
from minimcp.server.types import Message, NoMessage, Send
from minimcp.utils.model import to_json
from minimcp.utils.task_status_wrapper import TaskStatusWrapper

logger = logging.getLogger(__name__)


CONTENT_TYPE_SSE = "text/event-stream"

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "Content-Type": CONTENT_TYPE_SSE,
}

StreamableHTTPRequestHandler = Callable[[Message, Send], Awaitable[Message | NoMessage]]


# Context manager for suppressing expected stream errors
suppress_stream_errors = suppress(anyio.BrokenResourceError, anyio.ClosedResourceError)


# TODO: Add resumability based on Last-Event-ID header
class StreamableHTTPTransport(HTTPTransportBase):
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

    async def _runner(
        self,
        handler: StreamableHTTPRequestHandler,
        body: str,
        task_status: TaskStatus[Message | NoMessage | MemoryObjectReceiveStream[Message]],
    ):
        task_status_wrapper = TaskStatusWrapper(task_status)

        send_stream, recv_stream = anyio.create_memory_object_stream[Message](1)
        await self._stack.enter_async_context(recv_stream)

        async def send(msg: Message) -> None:
            task_status_wrapper.set(recv_stream)
            with suppress_stream_errors:
                await send_stream.send(msg)

        try:
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

        except Exception as e:
            logger.exception("Exception while handling request in StreamableHTTPTransport, generating error response")
            # Generate a proper JSON-RPC error response
            error_json = to_json(json_rpc.build_error_message(types.INTERNAL_ERROR, "", e))

            if not task_status_wrapper.set(error_json):
                # Try to send error through the stream if it exists and is still open
                with suppress_stream_errors:
                    await send_stream.send(error_json)

        # Exception Handling Strategy:
        # 1. Setup exceptions (stream creation, context manager): Propagate to caller (before try/except)
        # 2. Handler exceptions: Caught and converted to JSON-RPC error responses
        # 3. Cleanup exceptions: Handled gracefully with suppress_stream_errors
        # This prevents TaskGroup crashes while maintaining proper error reporting

    async def _handle_post_request(
        self, handler: StreamableHTTPRequestHandler, headers: Mapping[str, str], body: str
    ) -> HTTPResult:
        logger.debug("Handling POST request. Headers: %s, Body: %s", headers, body)

        if result := self._check_accept_headers(headers, {CONTENT_TYPE_JSON, CONTENT_TYPE_SSE}):
            return result
        if result := self._check_content_type(headers):
            return result
        if result := self._validate_protocol_version(headers, body):
            return result
        if result := self._validate_request_body(body):
            return result

        if self._tg is None:
            raise RuntimeError("StreamableHTTPTransport was not started")

        response = await self._tg.start(self._runner, handler, body)
        logger.debug("Handling completed. Response: %s", response)

        if isinstance(response, MemoryObjectReceiveStream):
            return HTTPResult(HTTPStatus.OK, response, headers=SSE_HEADERS)

        if isinstance(response, NoMessage):
            return HTTPResult(HTTPStatus.ACCEPTED)

        return HTTPResult(HTTPStatus.OK, response, CONTENT_TYPE_JSON)
