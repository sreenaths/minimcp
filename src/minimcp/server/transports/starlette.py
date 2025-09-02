from starlette.requests import Request
from starlette.responses import Response

from minimcp.server.transports.http import HTTP_TRANSPORT, HTTPRequestHandler


async def http_transport(handler: HTTPRequestHandler, request: Request):
    msg = await request.body()
    msg_str = msg.decode("utf-8")

    result = await HTTP_TRANSPORT.dispatch(handler, request.method, request.headers, msg_str)

    return Response(
        content=result.content, status_code=result.status_code, media_type=result.media_type, headers=result.headers
    )
