import logging
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack, suppress
from http import HTTPStatus

import anyio
from anyio.abc import TaskGroup, TaskStatus
from anyio.streams.memory import MemoryObjectReceiveStream

from minimcp.server.transports.http_transport_base import CONTENT_TYPE_JSON, HTTPResult, HTTPTransportBase
from minimcp.server.types import Message, NoMessage, Send
from minimcp.utils.task_status_wrapper import TaskStatusWrapper

logger = logging.getLogger(__name__)


CONTENT_TYPE_SSE = "text/event-stream"

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "Content-Type": CONTENT_TYPE_SSE,
}

StreamableHTTPRequestHandler = Callable[[Message, Send], Awaitable[Message | NoMessage]]


HandlerTaskResponse = Message | NoMessage | MemoryObjectReceiveStream[Message]


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
