import json
from http import HTTPStatus
from unittest.mock import AsyncMock

import anyio
import pytest
from anyio.streams.memory import MemoryObjectReceiveStream

from minimcp.server.transports.streamable_http import (
    CONTENT_TYPE_SSE,
    SSE_HEADERS,
    StreamableHTTPTransport,
    suppress_stream_errors,
)
from minimcp.server.types import NoMessage


class TestStreamableHTTPTransport:
    """Test suite for StreamableHTTPTransport."""

    @pytest.fixture
    def transport(self):
        """Create a StreamableHTTPTransport instance."""
        return StreamableHTTPTransport()

    @pytest.fixture
    def mock_handler(self):
        """Create a mock streamable handler."""
        return AsyncMock(return_value='{"jsonrpc": "2.0", "result": "success", "id": 1}')

    @pytest.fixture
    def valid_headers(self):
        """Create valid HTTP headers."""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-06-18",
        }

    @pytest.fixture
    def sse_headers(self):
        """Create headers that accept SSE."""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-06-18",
        }

    @pytest.fixture
    def valid_body(self):
        """Create a valid JSON-RPC request body."""
        return json.dumps({"jsonrpc": "2.0", "method": "test_method", "params": {"test": "value"}, "id": 1})

    @pytest.mark.asyncio
    async def test_transport_lifecycle(self, transport):
        """Test transport start and close lifecycle."""
        # Transport should not be started initially
        assert transport._tg is None

        # Start the transport
        started_transport = await transport.start()
        assert started_transport is transport
        assert transport._tg is not None

        # Close the transport
        await transport.aclose()
        assert transport._tg is None

    @pytest.mark.asyncio
    async def test_transport_context_manager(self):
        """Test transport as async context manager."""
        transport = StreamableHTTPTransport()

        async with transport as t:
            assert t is transport
            assert transport._tg is not None

        # Should be cleaned up after exiting context
        assert transport._tg is None

    @pytest.mark.asyncio
    async def test_dispatch_post_request_success(self, transport, mock_handler, valid_headers, valid_body):
        """Test successful POST request handling."""
        async with transport:
            result = await transport.dispatch(mock_handler, "POST", valid_headers, valid_body)

        assert result.status_code == HTTPStatus.OK
        assert result.content == '{"jsonrpc": "2.0", "result": "success", "id": 1}'
        assert result.media_type == "application/json"
        mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_unsupported_method(self, transport, mock_handler, valid_headers, valid_body):
        """Test handling of unsupported HTTP methods."""
        async with transport:
            result = await transport.dispatch(mock_handler, "GET", valid_headers, valid_body)

        assert result.status_code == HTTPStatus.METHOD_NOT_ALLOWED
        assert result.headers is not None
        assert "Allow" in result.headers
        assert "POST" in result.headers["Allow"]
        mock_handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_not_started_error(self, transport, mock_handler, valid_headers, valid_body):
        """Test that dispatch raises error when transport is not started."""
        with pytest.raises(RuntimeError, match="StreamableHTTPTransport was not started"):
            await transport.dispatch(mock_handler, "POST", valid_headers, valid_body)

    @pytest.mark.asyncio
    async def test_dispatch_sse_response(self, transport, sse_headers, valid_body):
        """Test dispatch returning SSE stream response."""

        async def streaming_handler(message, send):
            await send('{"jsonrpc": "2.0", "result": "stream1", "id": 1}')
            await send('{"jsonrpc": "2.0", "result": "stream2", "id": 2}')
            return NoMessage.NOTIFICATION

        async with transport:
            result = await transport.dispatch(streaming_handler, "POST", sse_headers, valid_body)

        assert result.status_code == HTTPStatus.OK
        assert isinstance(result.content, MemoryObjectReceiveStream)
        assert result.headers == SSE_HEADERS

    @pytest.mark.asyncio
    async def test_dispatch_no_message_response(self, transport, valid_headers, valid_body):
        """Test handling when handler returns NoMessage."""
        handler = AsyncMock(return_value=NoMessage.NOTIFICATION)

        async with transport:
            result = await transport.dispatch(handler, "POST", valid_headers, valid_body)

        assert result.status_code == HTTPStatus.ACCEPTED
        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_error_response(self, transport, valid_headers, valid_body):
        """Test handling of JSON-RPC error responses."""
        error_response = json.dumps(
            {"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid Request"}, "id": 1}
        )
        handler = AsyncMock(return_value=error_response)

        async with transport:
            result = await transport.dispatch(handler, "POST", valid_headers, valid_body)

        assert result.status_code == HTTPStatus.OK
        assert result.content == error_response
        assert result.media_type == "application/json"

    @pytest.mark.asyncio
    async def test_dispatch_handler_exception(self, transport, valid_headers, valid_body):
        """Test handling when handler raises an exception."""
        handler = AsyncMock(side_effect=Exception("Handler error"))

        async with transport:
            result = await transport.dispatch(handler, "POST", valid_headers, valid_body)

        # Should return an error response
        assert result.status_code == HTTPStatus.OK
        assert result.content is not None
        assert "error" in result.content
        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_streaming_with_final_response(self, transport, sse_headers, valid_body):
        """Test streaming handler that sends messages and returns a final response."""

        async def streaming_handler(message, send):
            await send('{"jsonrpc": "2.0", "result": "stream1", "id": 1}')
            await send('{"jsonrpc": "2.0", "result": "stream2", "id": 2}')
            return '{"jsonrpc": "2.0", "result": "final", "id": 3}'

        async with transport:
            result = await transport.dispatch(streaming_handler, "POST", sse_headers, valid_body)

        # Should return stream since send was called
        assert isinstance(result.content, MemoryObjectReceiveStream)
        assert result.headers == SSE_HEADERS

    @pytest.mark.asyncio
    async def test_dispatch_accept_both_json_and_sse(self, transport, valid_body):
        """Test dispatch with headers accepting both JSON and SSE."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-06-18",
        }
        handler = AsyncMock(return_value='{"result": "success"}')

        async with transport:
            result = await transport.dispatch(handler, "POST", headers, valid_body)

        assert result.status_code == HTTPStatus.OK
        assert result.content == '{"result": "success"}'
        assert result.media_type == "application/json"

    @pytest.mark.asyncio
    async def test_dispatch_invalid_accept_header(self, transport, valid_body):
        """Test dispatch with invalid Accept header."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/plain",
            "MCP-Protocol-Version": "2025-06-18",
        }
        handler = AsyncMock()

        async with transport:
            result = await transport.dispatch(handler, "POST", headers, valid_body)

        assert result.status_code == HTTPStatus.NOT_ACCEPTABLE
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_invalid_content_type(self, transport, valid_body):
        """Test dispatch with invalid Content-Type."""
        headers = {
            "Content-Type": "text/plain",
            "Accept": "application/json, text/event-stream",
        }
        handler = AsyncMock()

        async with transport:
            result = await transport.dispatch(handler, "POST", headers, valid_body)

        assert result.status_code == HTTPStatus.UNSUPPORTED_MEDIA_TYPE
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_protocol_version_validation(self, transport, valid_body):
        """Test protocol version validation."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "invalid-version",
        }
        handler = AsyncMock()

        async with transport:
            result = await transport.dispatch(handler, "POST", headers, valid_body)

        assert result.status_code == HTTPStatus.BAD_REQUEST
        assert "Unsupported protocol version" in result.content
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_runner_task_status_handling(self, transport, valid_headers, valid_body):
        """Test the _runner method's task status handling."""
        handler = AsyncMock(return_value='{"result": "success"}')

        async with transport:
            # Test the runner directly
            async def test_runner():
                async with anyio.create_task_group() as tg:
                    result = await tg.start(transport._runner, handler, valid_body)
                    return result

            result = await test_runner()

        assert result == '{"result": "success"}'
        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_runner_streaming_task_status(self, transport, valid_body):
        """Test the _runner method with streaming handler."""

        async def streaming_handler(message, send):
            await send('{"result": "stream"}')
            return NoMessage.NOTIFICATION

        async with transport:

            async def test_runner():
                async with anyio.create_task_group() as tg:
                    result = await tg.start(transport._runner, streaming_handler, valid_body)
                    return result

            result = await test_runner()

        assert isinstance(result, MemoryObjectReceiveStream)

    @pytest.mark.asyncio
    async def test_runner_exception_handling(self, transport, valid_body):
        """Test the _runner method's exception handling."""
        handler = AsyncMock(side_effect=Exception("Test error"))

        async with transport:

            async def test_runner():
                async with anyio.create_task_group() as tg:
                    result = await tg.start(transport._runner, handler, valid_body)
                    return result

            result = await test_runner()

        # Should return an error response
        assert isinstance(result, str)
        assert "error" in result
        assert "Test error" in result

    @pytest.mark.asyncio
    async def test_stream_error_suppression(self):
        """Test that stream errors are properly suppressed."""
        # Test the suppress_stream_errors context manager
        with suppress_stream_errors:
            raise anyio.BrokenResourceError("Test broken resource")

        with suppress_stream_errors:
            raise anyio.ClosedResourceError("Test closed resource")

        # Should not raise any exceptions
        assert True

    @pytest.mark.asyncio
    async def test_send_function_behavior(self, transport, sse_headers, valid_body):
        """Test the send function behavior in streaming context."""

        async def capturing_handler(message, send):
            await send('{"id": 1, "result": "first"}')
            await send('{"id": 2, "result": "second"}')
            return NoMessage.NOTIFICATION

        async with transport:
            result = await transport.dispatch(capturing_handler, "POST", sse_headers, valid_body)

        # Should return a stream when send() is called
        assert isinstance(result.content, MemoryObjectReceiveStream)
        assert result.status_code == HTTPStatus.OK
        assert result.headers == SSE_HEADERS

    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, transport, valid_headers, valid_body):
        """Test that multiple requests can be handled concurrently."""
        call_count = 0

        async def slow_handler(message, send):
            nonlocal call_count
            call_count += 1
            await anyio.sleep(0.01)  # Small delay
            return f'{{"result": "response_{call_count}", "id": {call_count}}}'

        async with transport:
            # Start multiple concurrent requests
            results = []
            async with anyio.create_task_group() as tg:

                async def make_request():
                    result = await transport.dispatch(slow_handler, "POST", valid_headers, valid_body)
                    results.append(result)

                for i in range(3):
                    tg.start_soon(make_request)

        # All requests should have been processed
        assert call_count == 3

    def test_sse_headers_constant(self):
        """Test SSE headers constant."""
        assert SSE_HEADERS["Cache-Control"] == "no-cache, no-transform"
        assert SSE_HEADERS["Connection"] == "keep-alive"
        assert SSE_HEADERS["Content-Type"] == CONTENT_TYPE_SSE

    def test_content_type_sse_constant(self):
        """Test SSE content type constant."""
        assert CONTENT_TYPE_SSE == "text/event-stream"

    @pytest.mark.asyncio
    async def test_transport_reuse_after_close(self, transport):
        """Test that transport can be reused after closing."""
        # First use
        async with transport:
            assert transport._tg is not None

        assert transport._tg is None

        # Second use
        async with transport:
            assert transport._tg is not None

        assert transport._tg is None

    @pytest.mark.asyncio
    async def test_multiple_start_calls(self, transport):
        """Test behavior when start is called multiple times."""
        # First start
        t1 = await transport.start()
        assert t1 is transport
        assert transport._tg is not None

        # Should be able to close and restart
        await transport.aclose()
        assert transport._tg is None

        # Second start
        t2 = await transport.start()
        assert t2 is transport
        assert transport._tg is not None

        await transport.aclose()

    @pytest.mark.asyncio
    async def test_dispatch_with_malformed_json_body(self, transport, valid_headers):
        """Test dispatch with malformed JSON in request body."""
        malformed_body = '{"jsonrpc": "2.0", "method": "test", invalid json'

        async with transport:
            # Should still call the handler - JSON validation is done at handler level
            result = await transport.dispatch(None, "POST", valid_headers, malformed_body)

        assert result.status_code == HTTPStatus.BAD_REQUEST

    @pytest.mark.asyncio
    async def test_initialize_request_skips_version_check(self, transport):
        """Test that initialize requests skip protocol version validation."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            # No protocol version header
        }

        initialize_body = json.dumps(
            {"jsonrpc": "2.0", "method": "initialize", "params": {"protocolVersion": "2025-06-18"}, "id": 1}
        )

        handler = AsyncMock(return_value='{"result": "initialized"}')

        async with transport:
            result = await transport.dispatch(handler, "POST", headers, initialize_body)

        assert result.status_code == HTTPStatus.OK
        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_cleanup_on_handler_exception(self, transport, sse_headers, valid_body):
        """Test that streams are properly cleaned up when handler raises exception."""

        async def failing_handler(message, send):
            await send('{"result": "before_error"}')
            raise Exception("Handler failed")

        async with transport:
            result = await transport.dispatch(failing_handler, "POST", sse_headers, valid_body)

        # Since send() was called before the exception, it returns a stream
        assert result.status_code == HTTPStatus.OK
        assert isinstance(result.content, MemoryObjectReceiveStream)
        assert result.headers == SSE_HEADERS
