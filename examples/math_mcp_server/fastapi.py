import logging
import sys

from fastapi import FastAPI, Request

from .math_mcp import math_mcp

# Configure logging globally for the demo server
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger(__name__)


app = FastAPI()


@app.post("/mcp")
async def handle_mcp_request(request: Request):
    msg = await request.json()
    return await math_mcp.handle(msg)
