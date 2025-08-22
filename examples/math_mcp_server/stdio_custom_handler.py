import logging
import os

import anyio

from minimcp.server.responder import Responder
from minimcp.server.transports.stdio import stdio_transport
from minimcp.server.types import Message

from .math_mcp import math_mcp

# Configure logging globally for the demo server
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.environ.get("MCP_SERVER_LOG_FILE", "mcp_server.log")),
        logging.StreamHandler(),  # Also log to stderr
    ],
)
logger = logging.getLogger(__name__)


async def custom_handler(message: Message, responder: Responder):
    try:
        # You could do setup (like building scope or session, creating custom responder, etc)
        # Custom scope type can be defined while instantiating MiniMCP.
        scope = {"session_id": "123"}
        return await math_mcp.handle(message, responder, scope)
    finally:
        pass
        # You could do teardown (like closing connections, etc) here.


if __name__ == "__main__":
    logger.info("MiniMCP: Started stdio server with custom handler, listening for messages...")
    anyio.run(stdio_transport, custom_handler)
