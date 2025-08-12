import logging
import sys

from fastapi import FastAPI, Request
from pydantic import Field

from minimcp.server import MiniMCP

# Configure logging globally for the demo server
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger(__name__)


app = FastAPI()
mcp = MiniMCP(name="MathServer", version="0.1.0")


@mcp.tool(name="add", description="Add two numbers")
def add(
    a: int = Field(..., description="The first number"), b: int = Field(..., description="The second number")
) -> int:
    return a + b


@app.post("/mcp")
async def handle_mcp_request(request: Request):
    msg = await request.json()
    return await mcp.handle(msg)
