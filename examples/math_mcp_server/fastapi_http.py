import logging
import sys

from fastapi import FastAPI, Request

from minimcp.server.transports.http import starlette_http_transport

from .math_mcp import math_mcp

# Configure logging globally for the demo server
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger(__name__)


app = FastAPI()


@app.post("/mcp")
async def handle_mcp_request(request: Request):
    return await starlette_http_transport(request, math_mcp.handle)
