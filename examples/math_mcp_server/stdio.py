import json
import logging
import os
import sys
from asyncio.exceptions import CancelledError
from io import TextIOWrapper
from typing import Any

import anyio
import mcp.types as types

from minimcp.server import MiniMCP, json_rpc
from minimcp.server.utils import to_dict

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


async def stdio_server(mcp: MiniMCP):
    """
    Server transport for stdio: this communicates with an MCP client by reading
    from the current process' stdin and writing to stdout.
    """

    stdin = anyio.wrap_file(TextIOWrapper(sys.stdin.buffer, encoding="utf-8"))
    stdout = anyio.wrap_file(TextIOWrapper(sys.stdout.buffer, encoding="utf-8"))

    async def write_message(response: dict[str, Any]):
        logger.info("Writing response message to stdio: %s", response)

        msg_json = json.dumps(response)
        await stdout.write(msg_json + "\n")
        await stdout.flush()

    async def handle_message(line: str):
        logger.info("Handling incoming message: %s", line)

        try:
            message = json.loads(line)
            response = await mcp.handle(message)
        except json.JSONDecodeError as e:
            response = to_dict(json_rpc.build_error_message(types.PARSE_ERROR, {}, e))

        if response:
            await write_message(response)

    logger.info("MiniMCP: Started stdio server, listening for messages...")
    try:
        # stdio is expected to connect the server with one single client,
        # hence not deploying advanced concurrency management or handling backpressure
        async with anyio.create_task_group() as tg:
            async for line in stdin:
                tg.start_soon(handle_message, line)
    except (KeyboardInterrupt, CancelledError):
        print("\nCtrl+C detected, exiting gracefully...")


def main():
    anyio.run(stdio_server, math_mcp)


if __name__ == "__main__":
    main()
