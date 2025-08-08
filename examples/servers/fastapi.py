import sys
import logging

from fastapi import FastAPI, Request
from minimcp.server import MiniMCP


# Configure logging globally for the demo server
logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


app = FastAPI()
mcp = MiniMCP(name="WeatherServer", version="0.1.0")


@app.post("/mcp")
async def handle_mcp_request(request: Request):
    msg = await request.json()
    return await mcp.handle(msg)
