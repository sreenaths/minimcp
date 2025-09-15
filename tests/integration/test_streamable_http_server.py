"""
Integration tests for MCP server using FastMCP StreamableHttpTransport client with streamable HTTP endpoint.

This module imports all tests from test_http_server.py and runs them against the streamable HTTP endpoint.
Additional streamable-specific tests can be added to this file.
"""

import sys
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from servers.http_server import SERVER_HOST, SERVER_PORT, STREAMABLE_HTTP_MCP_PATH  # noqa: E402
from test_http_server import TestHttpServer  # noqa: E402


class TestStreamableHttpServer(TestHttpServer):
    """Test suite for Streamable HTTP server."""

    server_url: str = f"http://{SERVER_HOST}:{SERVER_PORT}{STREAMABLE_HTTP_MCP_PATH}"
    default_headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    # Common tests would be inherited from TestHttpServer
