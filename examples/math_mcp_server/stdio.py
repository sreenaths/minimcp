import logging
import os
import sys
from io import TextIOWrapper

import anyio

from minimcp.server import MiniMCP

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


async def stdio_transport(mcp: MiniMCP):
    """
    stdio_transport interfaces MiniMCP with stdio by reading from
    the current process stdin and writing to stdout.
    """

    stdin = anyio.wrap_file(TextIOWrapper(sys.stdin.buffer, encoding="utf-8"))
    stdout = anyio.wrap_file(TextIOWrapper(sys.stdout.buffer, encoding="utf-8"))

    async def handle_message(line: str):
        logger.info("Handling incoming message: %s", line)

        response = await mcp.handles(line)
        if response:
            logger.info("Writing response message to stdio: %s", response)
            await stdout.write(response + "\n")
            await stdout.flush()

    logger.info("MiniMCP: Started stdio server, listening for messages...")
    try:
        # stdio is expected to connect the server with one single client,
        # hence not deploying advanced concurrency management or handling backpressure
        async with anyio.create_task_group() as tg:
            async for line in stdin:
                tg.start_soon(handle_message, line)
    except (KeyboardInterrupt, anyio.get_cancelled_exc_class()):
        print("\nCtrl+C detected, exiting gracefully...")


def main():
    anyio.run(stdio_transport, math_mcp)


if __name__ == "__main__":
    main()
