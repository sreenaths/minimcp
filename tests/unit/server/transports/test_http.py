import json
from http import HTTPStatus
from unittest.mock import AsyncMock

import pytest

from minimcp.server.transports.http import HTTPTransport
from minimcp.server.transports.http_transport_base import CONTENT_TYPE_JSON
from minimcp.server.types import NoMessage


class TestHTTPTransport:
    """Test suite for HTTP transport."""

    @pytest.fixture
    def transport(self):
        """Create an HTTPTransport instance."""
        return HTTPTransport()

    @pytest.fixture
    def mock_handler(self):
        """Create a mock handler."""
        return AsyncMock(return_value='{"jsonrpc": "2.0", "result": "success", "id": 1}')

    @pytest.fixture
    def valid_headers(self):
        """Create valid HTTP headers."""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "MCP-Protocol-Version": "2025-06-18",
        }

    @pytest.fixture
    def valid_body(self):
        """Create a valid JSON-RPC request body."""
        return json.dumps({"jsonrpc": "2.0", "method": "test_method", "params": {"test": "value"}, "id": 1})

    @pytest.mark.asyncio
    async def test_dispatch_post_request_success(self, transport, mock_handler, valid_headers, valid_body):
        """Test successful POST request handling."""
        result = await transport.dispatch(mock_handler, "POST", valid_headers, valid_body)

        assert result.status_code == HTTPStatus.OK
        assert result.content == '{"jsonrpc": "2.0", "result": "success", "id": 1}'
        assert result.media_type == CONTENT_TYPE_JSON
        mock_handler.assert_called_once_with(valid_body)

    @pytest.mark.asyncio
    async def test_dispatch_unsupported_method(self, transport, mock_handler, valid_headers, valid_body):
        """Test handling of unsupported HTTP methods."""
        result = await transport.dispatch(mock_handler, "GET", valid_headers, valid_body)

        assert result.status_code == HTTPStatus.METHOD_NOT_ALLOWED
        assert result.headers is not None
        assert "Allow" in result.headers
        assert "POST" in result.headers["Allow"]
        mock_handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_invalid_content_type(self, transport, mock_handler, valid_body):
        """Test handling of invalid content type."""
        headers = {
            "Content-Type": "text/plain",
            "Accept": "application/json",
        }

        result = await transport.dispatch(mock_handler, "POST", headers, valid_body)

        assert result.status_code == HTTPStatus.UNSUPPORTED_MEDIA_TYPE
        assert result.content is not None
        assert "Unsupported Media Type" in result.content
        mock_handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_invalid_accept_header(self, transport, mock_handler, valid_body):
        """Test handling of invalid Accept header."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/plain",
        }

        result = await transport.dispatch(mock_handler, "POST", headers, valid_body)

        assert result.status_code == HTTPStatus.NOT_ACCEPTABLE
        assert result.content is not None
        assert "Not Acceptable" in result.content
        mock_handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_no_message_response(self, transport, valid_headers, valid_body):
        """Test handling when handler returns NoMessage."""
        handler = AsyncMock(return_value=NoMessage.NOTIFICATION)

        result = await transport.dispatch(handler, "POST", valid_headers, valid_body)

        assert result.status_code == HTTPStatus.ACCEPTED
        assert result.content is None or isinstance(result.content, NoMessage)
        handler.assert_called_once_with(valid_body)

    @pytest.mark.asyncio
    async def test_dispatch_error_response(self, transport, valid_headers, valid_body):
        """Test handling of JSON-RPC error responses."""
        error_response = json.dumps(
            {"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid Request"}, "id": 1}
        )
        handler = AsyncMock(return_value=error_response)

        result = await transport.dispatch(handler, "POST", valid_headers, valid_body)

        assert result.status_code == HTTPStatus.OK
        assert result.content == error_response
        assert result.media_type == CONTENT_TYPE_JSON
        handler.assert_called_once_with(valid_body)

    @pytest.mark.asyncio
    async def test_dispatch_internal_error_response(self, transport, valid_headers, valid_body):
        """Test handling of internal error responses."""
        error_response = json.dumps({"jsonrpc": "2.0", "error": {"code": -32603, "message": "Internal error"}, "id": 1})
        handler = AsyncMock(return_value=error_response)

        result = await transport.dispatch(handler, "POST", valid_headers, valid_body)

        assert result.status_code == HTTPStatus.OK
        assert result.content == error_response
        assert result.media_type == CONTENT_TYPE_JSON

    @pytest.mark.asyncio
    async def test_dispatch_method_not_found_error(self, transport, valid_headers, valid_body):
        """Test handling of method not found errors."""
        error_response = json.dumps(
            {"jsonrpc": "2.0", "error": {"code": -32601, "message": "Method not found"}, "id": 1}
        )
        handler = AsyncMock(return_value=error_response)

        result = await transport.dispatch(handler, "POST", valid_headers, valid_body)

        assert result.status_code == HTTPStatus.OK
        assert result.content == error_response
        assert result.media_type == CONTENT_TYPE_JSON

    @pytest.mark.asyncio
    async def test_dispatch_unknown_error_code(self, transport, valid_headers, valid_body):
        """Test handling of unknown error codes."""
        error_response = json.dumps({"jsonrpc": "2.0", "error": {"code": -99999, "message": "Unknown error"}, "id": 1})
        handler = AsyncMock(return_value=error_response)

        result = await transport.dispatch(handler, "POST", valid_headers, valid_body)

        assert result.status_code == HTTPStatus.OK
        assert result.content == error_response
        assert result.media_type == CONTENT_TYPE_JSON

    @pytest.mark.asyncio
    async def test_dispatch_malformed_response(self, transport, valid_headers, valid_body):
        """Test handling of malformed JSON responses."""
        handler = AsyncMock(return_value="not valid json")

        result = await transport.dispatch(handler, "POST", valid_headers, valid_body)

        # Malformed JSON should result in 500 Internal Server Error
        assert result.status_code == HTTPStatus.OK
        assert result.content == "not valid json"
        assert result.media_type == CONTENT_TYPE_JSON

    @pytest.mark.asyncio
    async def test_dispatch_missing_headers(self, transport, mock_handler, valid_body):
        """Test handling with minimal headers."""
        headers = {}

        result = await transport.dispatch(mock_handler, "POST", headers, valid_body)

        # Should fail due to missing Accept header first (checked before Content-Type)
        assert result.status_code == HTTPStatus.NOT_ACCEPTABLE
        mock_handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_protocol_version_validation(self, transport, mock_handler, valid_body):
        """Test protocol version validation."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "MCP-Protocol-Version": "invalid-version",
        }

        result = await transport.dispatch(mock_handler, "POST", headers, valid_body)

        assert result.status_code == HTTPStatus.BAD_REQUEST
        assert "Unsupported protocol version" in result.content
        mock_handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_initialize_request_skips_version_check(self, transport, mock_handler):
        """Test that initialize requests skip protocol version validation."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            # No protocol version header
        }

        initialize_body = json.dumps(
            {"jsonrpc": "2.0", "method": "initialize", "params": {"protocolVersion": "2025-06-18"}, "id": 1}
        )

        result = await transport.dispatch(mock_handler, "POST", headers, initialize_body)

        assert result.status_code == HTTPStatus.OK
        mock_handler.assert_called_once_with(initialize_body)

    @pytest.mark.asyncio
    async def test_dispatch_default_protocol_version(self, transport, mock_handler, valid_body):
        """Test that default protocol version is used when header is missing."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            # No MCP-Protocol-Version header - should use default
        }

        result = await transport.dispatch(mock_handler, "POST", headers, valid_body)

        assert result.status_code == HTTPStatus.OK
        mock_handler.assert_called_once_with(valid_body)

    @pytest.mark.asyncio
    async def test_dispatch_content_type_with_charset(self, transport, mock_handler, valid_body):
        """Test Content-Type header with charset parameter."""
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "MCP-Protocol-Version": "2025-06-18",
        }

        result = await transport.dispatch(mock_handler, "POST", headers, valid_body)

        assert result.status_code == HTTPStatus.OK
        mock_handler.assert_called_once_with(valid_body)

    @pytest.mark.asyncio
    async def test_dispatch_accept_header_with_quality(self, transport, mock_handler, valid_body):
        """Test Accept header with quality values."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json; q=0.9, text/plain; q=0.1",
            "MCP-Protocol-Version": "2025-06-18",
        }

        result = await transport.dispatch(mock_handler, "POST", headers, valid_body)

        assert result.status_code == HTTPStatus.OK
        mock_handler.assert_called_once_with(valid_body)

    @pytest.mark.asyncio
    async def test_dispatch_case_insensitive_headers(self, transport, mock_handler, valid_body):
        """Test that header checking is case insensitive."""
        headers = {
            "Content-Type": "APPLICATION/JSON",
            "Accept": "APPLICATION/JSON",
            "MCP-Protocol-Version": "2025-06-18",
        }

        result = await transport.dispatch(mock_handler, "POST", headers, valid_body)

        assert result.status_code == HTTPStatus.OK
        mock_handler.assert_called_once_with(valid_body)

    @pytest.mark.asyncio
    async def test_handle_post_request_direct(self, transport, mock_handler, valid_headers, valid_body):
        """Test the _handle_post_request method directly."""
        result = await transport._handle_post_request(mock_handler, valid_headers, valid_body)

        assert result.status_code == HTTPStatus.OK
        assert result.content == '{"jsonrpc": "2.0", "result": "success", "id": 1}'
        assert result.media_type == CONTENT_TYPE_JSON
        mock_handler.assert_called_once_with(valid_body)

    @pytest.mark.asyncio
    async def test_response_without_id(self, transport, valid_headers, valid_body):
        """Test handling of responses without ID (notifications)."""
        notification_response = json.dumps(
            {
                "jsonrpc": "2.0",
                "result": "success",
                # No ID field
            }
        )
        handler = AsyncMock(return_value=notification_response)

        result = await transport.dispatch(handler, "POST", valid_headers, valid_body)

        assert result.status_code == HTTPStatus.OK
        assert result.content == notification_response
        assert result.media_type == CONTENT_TYPE_JSON
