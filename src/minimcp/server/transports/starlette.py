from anyio.streams.memory import MemoryObjectReceiveStream
from sse_starlette.sse import EventSourceResponse
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import Response

from minimcp.server.transports.http import HTTPRequestHandler, HTTPTransport
from minimcp.server.transports.streamable_http import StreamableHTTPRequestHandler, StreamableHTTPTransport


async def http_transport(handler: HTTPRequestHandler, request: Request) -> Response:
    msg = await request.body()
    msg_str = msg.decode("utf-8")

    transport = HTTPTransport()
    result = await transport.dispatch(handler, request.method, request.headers, msg_str)

    return Response(
        content=result.content, status_code=result.status_code, media_type=result.media_type, headers=result.headers
    )


async def streamable_http_transport(handler: StreamableHTTPRequestHandler, request: Request) -> Response:
    msg = await request.body()
    msg_str = msg.decode("utf-8")

    transport = StreamableHTTPTransport()
    await transport.start()
    close_transport = BackgroundTask(transport.aclose)

    result = await transport.dispatch(handler, request.method, request.headers, msg_str)

    if isinstance(result.content, MemoryObjectReceiveStream):
        return EventSourceResponse(
            content=result.content,
            ping=15,
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
