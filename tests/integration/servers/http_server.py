#!/usr/bin/env python3
"""
Test server for HTTP transport integration tests.
"""

import logging
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request

from minimcp import starlette

SERVER_HOST = os.environ.get("TEST_SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("TEST_SERVER_PORT", "30789"))

HEALTH_PATH = "/health"
HTTP_MCP_PATH = "/http-mcp"
STREAMABLE_HTTP_MCP_PATH = "/streamable-http-mcp"


# Add the current directory to Python path to import math_mcp
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from math_mcp import math_mcp  # noqa: E402

# Configure logging for the test server
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.environ.get("MCP_SERVER_LOG_FILE", "test_mcp_server.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# The server name is already set to "TestMathServer" in math_mcp.py

app = FastAPI()


@app.get(HEALTH_PATH)
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "server": "TestMathServer"}


@app.post(HTTP_MCP_PATH)
async def handle_http_mcp_request(request: Request):
    """Handle MCP requests via HTTP transport."""
    return await starlette.http_transport(math_mcp.handle, request)


@app.post(STREAMABLE_HTTP_MCP_PATH)
async def handle_streamable_http_mcp_request(request: Request):
    """Handle MCP requests via Streamable HTTP transport."""
    return await starlette.streamable_http_transport(math_mcp.handle, request)


def main():
    """Main entry point for the test server."""

    logger.info("Starting TestMathServer on %s:%s", SERVER_HOST, SERVER_PORT)

    uvicorn.run(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
