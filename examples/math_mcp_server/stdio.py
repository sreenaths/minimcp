import logging
import os

import anyio

from minimcp.server.responder import Responder
from minimcp.server.transports.stdio import stdio_transport
from minimcp.server.types import Message, MessageSender

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


async def transport_handler(message: Message, sender: MessageSender):
    # You could do setup (like building scope, creating responder, etc) and teardown in this function.
    responder = Responder(message, sender)
    return await math_mcp.handle(message, responder=responder)


def main():
    anyio.run(stdio_transport, transport_handler)


if __name__ == "__main__":
    main()
