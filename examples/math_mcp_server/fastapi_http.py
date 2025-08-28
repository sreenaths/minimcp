import logging
import sys

from fastapi import FastAPI, Request
from fastapi.responses import Response

from minimcp.server.transports.http import MEDIA_TYPE, get_response_http_code

from .math_mcp import math_mcp

# Configure logging globally for the demo server
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger(__name__)


app = FastAPI()


@app.post("/mcp")
async def handle_mcp_request(request: Request):
    msg = await request.body()
    response = await math_mcp.handle(msg.decode("utf-8"))
    return Response(content=response, status_code=get_response_http_code(response), media_type=MEDIA_TYPE)
