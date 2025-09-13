import logging
import sys

import anyio
from fastapi import FastAPI, Request
from pydantic import Field

from minimcp import MiniMCP, starlette

# Configure logging globally for the demo server
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger(__name__)


app = FastAPI()

math_mcp = MiniMCP(name="MathServer - Streamable HTTP", version="0.1.0")


@math_mcp.tool(description="Add two numbers")
async def add(
    a: float = Field(description="The first float number"), b: float = Field(description="The second float number")
) -> float:
    responder = math_mcp.context.get_responder()
    await responder.report_progress(0.1, message="Adding numbers")
    await anyio.sleep(1)
    await responder.report_progress(0.4, message="Adding numbers")
    await anyio.sleep(1)
    await responder.report_progress(0.7, message="Adding numbers")
    await anyio.sleep(1)
    return a + b


@math_mcp.tool(description="Subtract two numbers")
def subtract(
    a: float = Field(description="The first float number"), b: float = Field(description="The second float number")
) -> float:
    return a - b


@app.post("/mcp")
async def handle_mcp_request(request: Request):
    return await starlette.streamable_http_transport(math_mcp.handle, request)
