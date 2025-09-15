"""
Integration tests for MCP server using FastMCP StreamableHttpTransport client with streamable HTTP endpoint.

This module imports all tests from test_http_server.py and runs them against the streamable HTTP endpoint.
Additional streamable-specific tests can be added to this file.
"""

import sys
from pathlib import Path

import pytest
from fastmcp.client import Client, StreamableHttpTransport

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

    @pytest.mark.asyncio
    async def test_add_with_progress_tool(self, mcp_client):
        """Test calling the add_with_progress tool which sends progress notifications."""
        # Test that the tool exists and can be called
        tools = await mcp_client.list_tools()
        tool_names = [tool.name for tool in tools]
        assert "add_with_progress" in tool_names, f"add_with_progress tool not found in {tool_names}"

        # Call the tool with progress reporting
        result = await mcp_client.call_tool("add_with_progress", {"a": 5.0, "b": 3.0})

        # Verify the result
        assert result.is_error is False
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert float(result.content[0].text) == 8.0

    @pytest.mark.asyncio
    async def test_add_with_progress_tool_with_progress_handler(self, start_mcp_http_server):
        """Test calling the add_with_progress tool with a progress handler to capture progress notifications."""

        # Track progress notifications
        progress_notifications = []

        async def progress_handler(progress: float, total: float | None, message: str | None) -> None:
            """Capture progress notifications."""
            progress_notifications.append({"progress": progress, "total": total, "message": message})

        # Create client with progress handler
        transport = StreamableHttpTransport(url=self.server_url)
        client = Client(transport, progress_handler=progress_handler)

        async with client:
            # Test that the tool exists and can be called
            tools = await client.list_tools()
            tool_names = [tool.name for tool in tools]
            assert "add_with_progress" in tool_names, f"add_with_progress tool not found in {tool_names}"

            # Call the tool with progress reporting
            result = await client.call_tool("add_with_progress", {"a": 7.0, "b": 13.0})

            # Verify the result
            assert result.is_error is False
            assert len(result.content) == 1
            assert result.content[0].type == "text"
            assert float(result.content[0].text) == 20.0

            # Verify progress notifications were received
            assert len(progress_notifications) > 0, "No progress notifications were received"

            # Verify we got the expected progress values (0.1, 0.4, 0.7 from the implementation)
            progress_values = [n["progress"] for n in progress_notifications]
            expected_progress = [0.1, 0.4, 0.7]
            assert progress_values == expected_progress, f"Expected progress {expected_progress}, got {progress_values}"
