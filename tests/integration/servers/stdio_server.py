import logging
import os
import sys
from pathlib import Path

import anyio

from minimcp import stdio_transport

# Add the current directory to Python path to import math_mcp
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from math_mcp import math_mcp  # noqa: E402

# Configure logging for the test server
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.environ.get("MCP_SERVER_LOG_FILE", "stdio_server.log")),
        logging.StreamHandler(sys.stderr),  # Log to stderr to avoid interfering with stdio transport
    ],
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the test math server"""
    logger.info("Test MiniMCP: Started stdio server, listening for messages...")
    anyio.run(stdio_transport, math_mcp.handle)


if __name__ == "__main__":
    main()
