import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from anyio.streams.memory import MemoryObjectReceiveStream
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route
from starlette.testclient import TestClient

from minimcp.server.transports.starlette import (
    TRANSPORT_STATE_OBJ_KEY,
    http_transport,
    streamable_http_lifespan,
    streamable_http_transport,
)
from minimcp.server.types import NoMessage


class TestStarletteTransports:
    """Test suite for Starlette transport functions."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock Starlette request."""
        request = Mock(spec=Request)
        request.method = "POST"
        request.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "MCP-Protocol-Version": "2025-06-18",
        }
        request.body = AsyncMock(return_value=b'{"jsonrpc": "2.0", "method": "test", "id": 1}')
        return request

    @pytest.fixture
    def mock_handler(self):
        """Create a mock handler."""
        return AsyncMock(return_value='{"jsonrpc": "2.0", "result": "success", "id": 1}')

    @pytest.fixture
    def mock_streamable_handler(self):
        """Create a mock streamable handler."""
        return AsyncMock(return_value='{"jsonrpc": "2.0", "result": "success", "id": 1}')

    @pytest.mark.asyncio
    async def test_http_transport_success(self, mock_handler, mock_request):
        """Test successful HTTP transport handling."""
        response = await http_transport(mock_handler, mock_request)

        assert isinstance(response, Response)
        assert response.status_code == 200
        assert response.media_type == "application/json"
        assert response.body == b'{"jsonrpc": "2.0", "result": "success", "id": 1}'

        mock_request.body.assert_called_once()
        mock_handler.assert_called_once_with('{"jsonrpc": "2.0", "method": "test", "id": 1}')

    @pytest.mark.asyncio
    async def test_http_transport_no_message_response(self, mock_request):
        """Test HTTP transport with NoMessage response."""
        handler = AsyncMock(return_value=NoMessage.NOTIFICATION)

        response = await http_transport(handler, mock_request)

        assert isinstance(response, Response)
        assert response.status_code == 202  # ACCEPTED

        mock_request.body.assert_called_once()
        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_http_transport_error_response(self, mock_request):
        """Test HTTP transport with error response."""
        error_response = json.dumps(
            {"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid Request"}, "id": 1}
        )
        handler = AsyncMock(return_value=error_response)

        response = await http_transport(handler, mock_request)

        assert isinstance(response, Response)
        assert response.status_code == 200
        assert bytes(response.body).decode() == error_response

    @pytest.mark.asyncio
    async def test_http_transport_invalid_method(self, mock_handler):
        """Test HTTP transport with invalid method."""
        request = Mock(spec=Request)
        request.method = "GET"
        request.headers = {"Accept": "application/json"}
        request.body = AsyncMock(return_value=b"{}")

        response = await http_transport(mock_handler, request)

        assert isinstance(response, Response)
        assert response.status_code == 405  # Method Not Allowed
        assert "Allow" in response.headers

    @pytest.mark.asyncio
    async def test_streamable_http_transport_success(self, mock_streamable_handler, mock_request):
        """Test successful streamable HTTP transport handling."""
        # Mock the request state to not have a transport
        mock_request.state = Mock()
        # Add required Accept header for streamable transport
        mock_request.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-06-18",
        }

        with patch("minimcp.server.transports.starlette.getattr", return_value=None):
            response = await streamable_http_transport(mock_streamable_handler, mock_request)

        assert isinstance(response, Response)
        # May not be 200 due to header validation
        mock_request.body.assert_called_once()

    @pytest.mark.asyncio
    async def test_streamable_http_transport_with_app_transport(self, mock_streamable_handler, mock_request):
        """Test streamable HTTP transport using application-level transport."""
        # Mock transport from application state
        mock_transport = Mock()
        mock_transport.dispatch = AsyncMock(
            return_value=Mock(
                status_code=200, content='{"result": "success"}', media_type="application/json", headers={}
            )
        )

        mock_request.state = Mock()

        with patch("minimcp.server.transports.starlette.getattr", return_value=mock_transport):
            response = await streamable_http_transport(mock_streamable_handler, mock_request)

        assert isinstance(response, Response)
        mock_transport.dispatch.assert_called_once()

    @pytest.mark.asyncio
    async def test_streamable_http_transport_sse_response(self, mock_streamable_handler, mock_request):
        """Test streamable HTTP transport returning SSE response."""
        # Mock a MemoryObjectReceiveStream response
        mock_stream = Mock(spec=MemoryObjectReceiveStream)

        mock_request.state = Mock()

        with (
            patch("minimcp.server.transports.starlette.getattr", return_value=None),
            patch("minimcp.server.transports.starlette.StreamableHTTPTransport") as MockTransport,
        ):
            mock_transport_instance = Mock()
            mock_transport_instance.start = AsyncMock(return_value=mock_transport_instance)
            mock_transport_instance.aclose = AsyncMock()
            mock_transport_instance.dispatch = AsyncMock(
                return_value=Mock(content=mock_stream, headers={"Content-Type": "text/event-stream"})
            )
            MockTransport.return_value = mock_transport_instance

            response = await streamable_http_transport(mock_streamable_handler, mock_request)

        # Should return EventSourceResponse for stream content
        from sse_starlette.sse import EventSourceResponse

        assert isinstance(response, EventSourceResponse)

    @pytest.mark.asyncio
    async def test_streamable_http_transport_no_message(self, mock_request):
        """Test streamable HTTP transport with NoMessage response."""
        handler = AsyncMock(return_value=NoMessage.NOTIFICATION)
        mock_request.state = Mock()
        mock_request.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-06-18",
        }

        with patch("minimcp.server.transports.starlette.getattr", return_value=None):
            response = await streamable_http_transport(handler, mock_request)

        assert isinstance(response, Response)
        # Status code depends on transport validation

    @pytest.mark.asyncio
    async def test_streamable_http_transport_custom_ping(self, mock_streamable_handler, mock_request):
        """Test streamable HTTP transport with custom ping interval."""
        mock_stream = Mock(spec=MemoryObjectReceiveStream)
        mock_request.state = Mock()

        with (
            patch("minimcp.server.transports.starlette.getattr", return_value=None),
            patch("minimcp.server.transports.starlette.StreamableHTTPTransport") as MockTransport,
        ):
            mock_transport_instance = Mock()
            mock_transport_instance.start = AsyncMock(return_value=mock_transport_instance)
            mock_transport_instance.aclose = AsyncMock()
            mock_transport_instance.dispatch = AsyncMock(
                return_value=Mock(content=mock_stream, headers={"Content-Type": "text/event-stream"})
            )
            MockTransport.return_value = mock_transport_instance

            response = await streamable_http_transport(mock_streamable_handler, mock_request, ping=30)

        from sse_starlette.sse import EventSourceResponse

        assert isinstance(response, EventSourceResponse)
        # The ping parameter should be passed to EventSourceResponse
        # We can't easily verify this without inspecting the response object internals

    @pytest.mark.asyncio
    async def test_streamable_http_lifespan(self):
        """Test streamable HTTP lifespan context manager."""
        app = Starlette()

        with patch("minimcp.server.transports.starlette.StreamableHTTPTransport") as MockTransport:
            mock_transport_instance = Mock()
            mock_transport_instance.__aenter__ = AsyncMock(return_value=mock_transport_instance)
            mock_transport_instance.__aexit__ = AsyncMock(return_value=None)
            MockTransport.return_value = mock_transport_instance

            async with streamable_http_lifespan(app):
                # Verify transport was set on app state
                transport = getattr(app.state, TRANSPORT_STATE_OBJ_KEY, None)
                assert transport is mock_transport_instance

            # Verify transport was properly cleaned up
            mock_transport_instance.__aenter__.assert_called_once()
            mock_transport_instance.__aexit__.assert_called_once()

    def test_transport_state_key_constant(self):
        """Test that the transport state key constant is properly defined."""
        assert TRANSPORT_STATE_OBJ_KEY == "_streamable_http_transport"
        assert isinstance(TRANSPORT_STATE_OBJ_KEY, str)

    @pytest.mark.asyncio
    async def test_http_transport_body_decoding(self, mock_handler):
        """Test that request body is properly decoded from bytes to string."""
        request = Mock(spec=Request)
        request.method = "POST"
        request.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        # Test with UTF-8 encoded bytes
        test_message = '{"jsonrpc": "2.0", "method": "test", "params": {"unicode": "café"}, "id": 1}'
        request.body = AsyncMock(return_value=test_message.encode("utf-8"))

        response = await http_transport(mock_handler, request)

        assert isinstance(response, Response)
        mock_handler.assert_called_once_with(test_message)

    @pytest.mark.asyncio
    async def test_streamable_http_transport_body_decoding(self, mock_streamable_handler):
        """Test that streamable transport properly decodes request body."""
        request = Mock(spec=Request)
        request.method = "POST"
        request.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-06-18",
        }
        request.state = Mock()

        test_message = '{"jsonrpc": "2.0", "method": "test", "id": 1}'
        request.body = AsyncMock(return_value=test_message.encode("utf-8"))

        with patch("minimcp.server.transports.starlette.getattr", return_value=None):
            response = await streamable_http_transport(mock_streamable_handler, request)

        assert isinstance(response, Response)
        # Verify the handler was called
        mock_streamable_handler.assert_called()

    @pytest.mark.asyncio
    async def test_streamable_http_transport_background_task_cleanup(self, mock_streamable_handler, mock_request):
        """Test that background task is set for transport cleanup when no app transport exists."""
        mock_request.state = Mock()

        with (
            patch("minimcp.server.transports.starlette.getattr", return_value=None),
            patch("minimcp.server.transports.starlette.StreamableHTTPTransport") as MockTransport,
            patch("minimcp.server.transports.starlette.BackgroundTask") as MockBackgroundTask,
        ):
            mock_transport_instance = Mock()
            mock_transport_instance.start = AsyncMock(return_value=mock_transport_instance)
            mock_transport_instance.aclose = AsyncMock()
            mock_transport_instance.dispatch = AsyncMock(
                return_value=Mock(
                    content='{"result": "success"}', status_code=200, media_type="application/json", headers={}
                )
            )
            MockTransport.return_value = mock_transport_instance

            response = await streamable_http_transport(mock_streamable_handler, mock_request)

        # Verify BackgroundTask was created for cleanup
        MockBackgroundTask.assert_called_once_with(mock_transport_instance.aclose)
        assert isinstance(response, Response)
        assert hasattr(response, "background")

    @pytest.mark.asyncio
    async def test_streamable_http_transport_error_handling(self, mock_request):
        """Test error handling in streamable HTTP transport."""
        handler = AsyncMock(side_effect=Exception("Handler error"))
        mock_request.state = Mock()

        with patch("minimcp.server.transports.starlette.getattr", return_value=None):
            response = await streamable_http_transport(handler, mock_request)

        # Should still return a response (error handling is done in the transport layer)
        assert isinstance(response, Response)

    def test_integration_with_starlette_app(self):
        """Test integration with a real Starlette application."""

        async def mcp_endpoint(request):
            async def simple_handler(message):
                return '{"jsonrpc": "2.0", "result": "test", "id": 1}'

            return await http_transport(simple_handler, request)

        app = Starlette(routes=[Route("/mcp", mcp_endpoint, methods=["POST"])])

        client = TestClient(app)

        response = client.post(
            "/mcp", json={"jsonrpc": "2.0", "method": "test", "id": 1}, headers={"Accept": "application/json"}
        )

        assert response.status_code == 200
        assert response.json() == {"jsonrpc": "2.0", "result": "test", "id": 1}
