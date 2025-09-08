from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from anyio.streams.memory import MemoryObjectReceiveStream
from sse_starlette.sse import EventSourceResponse
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import Response

from minimcp.server.transports.http import HTTPRequestHandler, HTTPTransport
from minimcp.server.transports.streamable_http import StreamableHTTPRequestHandler, StreamableHTTPTransport


# --- HTTP Transport ---
async def http_transport(handler: HTTPRequestHandler, request: Request) -> Response:
    msg = await request.body()
    msg_str = msg.decode("utf-8")

    transport = HTTPTransport()
    result = await transport.dispatch(handler, request.method, request.headers, msg_str)

    return Response(
        content=result.content, status_code=result.status_code, media_type=result.media_type, headers=result.headers
    )


# --- Streamable HTTP Transport ---
TRANSPORT_STATE_OBJ_KEY = "_streamable_http_transport"


@asynccontextmanager
async def streamable_http_lifespan(app: Starlette) -> AsyncGenerator[None, None]:
    async with StreamableHTTPTransport() as transport:
        setattr(app.state, TRANSPORT_STATE_OBJ_KEY, transport)
        yield


async def streamable_http_transport(
    handler: StreamableHTTPRequestHandler, request: Request, ping: int = 15
) -> Response:
    msg = await request.body()
    msg_str = msg.decode("utf-8")

    transport: StreamableHTTPTransport | None = getattr(request.state, TRANSPORT_STATE_OBJ_KEY, None)
    close_transport = None
    if transport is None:
        # No application-level StreamableHTTPTransport found; create a new transport instance for this request.
        transport = StreamableHTTPTransport()
        await transport.start()
        # Use a BackgroundTask to ensure the transport is properly closed after the response is served.
        close_transport = BackgroundTask(transport.aclose)

    result = await transport.dispatch(handler, request.method, request.headers, msg_str)

    if isinstance(result.content, MemoryObjectReceiveStream):
        return EventSourceResponse(
            content=result.content,
            ping=ping,
            headers=result.headers,
            background=close_transport,
        )

    return Response(
        content=result.content,
        status_code=result.status_code,
        media_type=result.media_type,
        headers=result.headers,
        background=close_transport,
    )
