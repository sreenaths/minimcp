import json
import logging
from unittest.mock import AsyncMock, Mock, patch

import anyio
import mcp.types as types
import pytest

from minimcp.server.json_rpc import JSON_RPC_VERSION
from minimcp.server.responder import Responder
from minimcp.server.transports import TransportRequestHandler, stdio_transport
from minimcp.server.types import Message


class AsyncIteratorMock:
    """Mock async iterator for testing."""

    def __init__(self, items):
        self.items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.items)
        except StopIteration:
            raise StopAsyncIteration


class TestStdioTransport:
    """Test suite for stdio_transport function."""

    @pytest.fixture
    def mock_handler(self) -> AsyncMock:
        """Create a mock TransportRequestHandler for testing."""
        handler = AsyncMock(spec=TransportRequestHandler)
        handler.return_value = {"jsonrpc": JSON_RPC_VERSION, "id": "test", "result": {"status": "ok"}}
        return handler

    @pytest.fixture
    def sample_message(self) -> Message:
        """Create a sample JSON-RPC message for testing."""
        return {"jsonrpc": JSON_RPC_VERSION, "id": "test-123", "method": "tools/list", "params": {}}

    @pytest.fixture
    def sample_response(self) -> Message:
        """Create a sample response message for testing."""
        return {"jsonrpc": JSON_RPC_VERSION, "id": "test-123", "result": {"tools": []}}

    @pytest.mark.asyncio
    async def test_empty_stdin_completes_gracefully(self):
        """Test that stdio_transport completes when stdin is empty."""
        with patch("sys.stdout") as mock_stdout, patch("sys.stdin") as mock_stdin:
            mock_stdout.buffer = Mock()
            mock_stdin.buffer = Mock()

            mock_wrapped_stdout = AsyncMock()
            mock_wrapped_stdin = AsyncIteratorMock([])  # Empty iterator

            with patch("anyio.wrap_file") as mock_wrap:
                mock_wrap.side_effect = [mock_wrapped_stdin, mock_wrapped_stdout]

                mock_handler = AsyncMock()

                # Should complete without error
                await stdio_transport(mock_handler)

                # Handler should not be called with empty input
                mock_handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_valid_json_message(self, sample_message: Message, sample_response: Message):
        """Test handling a valid JSON message."""
        message_line = json.dumps(sample_message)

        with patch("sys.stdout") as mock_stdout, patch("sys.stdin") as mock_stdin:
            mock_stdout.buffer = Mock()
            mock_stdin.buffer = Mock()

            mock_wrapped_stdout = AsyncMock()
            mock_wrapped_stdin = AsyncIteratorMock([message_line])

            with patch("anyio.wrap_file") as mock_wrap:
                mock_wrap.side_effect = [mock_wrapped_stdin, mock_wrapped_stdout]

                mock_handler = AsyncMock()
                mock_handler.return_value = sample_response

                await stdio_transport(mock_handler)

                # Verify handler was called with correct arguments
                mock_handler.assert_called_once()
                call_args = mock_handler.call_args
                assert call_args[0][0] == sample_message  # First argument is the parsed message
                assert isinstance(call_args[0][1], Responder)  # Second argument is write_msg function

                # Verify response was written to stdout
                mock_wrapped_stdout.write.assert_called()
                written_data = mock_wrapped_stdout.write.call_args[0][0]
                assert json.loads(written_data.strip()) == sample_response

    @pytest.mark.asyncio
    async def test_handle_invalid_json(self):
        """Test handling invalid JSON input."""
        invalid_json = '{"invalid": json}'

        with patch("sys.stdout") as mock_stdout, patch("sys.stdin") as mock_stdin:
            mock_stdout.buffer = Mock()
            mock_stdin.buffer = Mock()

            mock_wrapped_stdout = AsyncMock()
            mock_wrapped_stdin = AsyncIteratorMock([invalid_json])

            with patch("anyio.wrap_file") as mock_wrap:
                mock_wrap.side_effect = [mock_wrapped_stdin, mock_wrapped_stdout]

                mock_handler = AsyncMock()

                await stdio_transport(mock_handler)

                # Verify handler was not called for invalid JSON
                mock_handler.assert_not_called()

                # Verify error response was written to stdout
                mock_wrapped_stdout.write.assert_called()
                written_data = mock_wrapped_stdout.write.call_args[0][0]
                error_response = json.loads(written_data.strip())

                assert error_response["jsonrpc"] == JSON_RPC_VERSION
                assert "error" in error_response
                assert error_response["error"]["code"] == types.PARSE_ERROR

    @pytest.mark.asyncio
    async def test_handler_returns_none(self, sample_message):
        """Test when handler returns None."""
        message_line = json.dumps(sample_message)

        with patch("sys.stdout") as mock_stdout, patch("sys.stdin") as mock_stdin:
            mock_stdout.buffer = Mock()
            mock_stdin.buffer = Mock()

            mock_wrapped_stdout = AsyncMock()
            mock_wrapped_stdin = AsyncIteratorMock([message_line])

            with patch("anyio.wrap_file") as mock_wrap:
                mock_wrap.side_effect = [mock_wrapped_stdin, mock_wrapped_stdout]

                mock_handler = AsyncMock()
                mock_handler.return_value = None

                await stdio_transport(mock_handler)

                # Verify handler was called
                mock_handler.assert_called_once()

                # When handler returns None, write_msg is called with None
                # and write_msg should not write anything for None
                mock_wrapped_stdout.write.assert_not_called()

    @pytest.mark.asyncio
    async def test_handler_exception(self, sample_message):
        """Test when handler raises an exception."""
        message_line = json.dumps(sample_message)

        with patch("sys.stdout") as mock_stdout, patch("sys.stdin") as mock_stdin:
            mock_stdout.buffer = Mock()
            mock_stdin.buffer = Mock()

            mock_wrapped_stdout = AsyncMock()
            mock_wrapped_stdin = AsyncIteratorMock([message_line])

            with patch("anyio.wrap_file") as mock_wrap:
                mock_wrap.side_effect = [mock_wrapped_stdin, mock_wrapped_stdout]

                mock_handler = AsyncMock()
                mock_handler.side_effect = Exception("Handler error")

                # The exception should propagate up through the task group
                with pytest.raises(Exception):
                    await stdio_transport(mock_handler)

                # Verify handler was called
                mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_messages(self, sample_message: Message, sample_response: Message):
        """Test handling multiple messages in sequence."""
        message1 = {**sample_message, "id": "msg1"}
        message2 = {**sample_message, "id": "msg2"}
        response1 = {**sample_response, "id": "msg1"}
        response2 = {**sample_response, "id": "msg2"}

        message_lines = [json.dumps(message1), json.dumps(message2)]

        with patch("sys.stdout") as mock_stdout, patch("sys.stdin") as mock_stdin:
            mock_stdout.buffer = Mock()
            mock_stdin.buffer = Mock()

            mock_wrapped_stdout = AsyncMock()
            mock_wrapped_stdin = AsyncIteratorMock(message_lines)

            with patch("anyio.wrap_file") as mock_wrap:
                mock_wrap.side_effect = [mock_wrapped_stdin, mock_wrapped_stdout]

                mock_handler = AsyncMock()
                mock_handler.side_effect = [response1, response2]

                await stdio_transport(mock_handler)

                # Verify handler was called twice
                assert mock_handler.call_count == 2

                # Verify both responses were written
                assert mock_wrapped_stdout.write.call_count == 2

    @pytest.mark.asyncio
    async def test_logging_behavior(self, sample_message: Message, caplog):
        """Test that appropriate log messages are generated."""
        message_line = json.dumps(sample_message)

        with caplog.at_level(logging.INFO):
            with patch("sys.stdout") as mock_stdout, patch("sys.stdin") as mock_stdin:
                mock_stdout.buffer = Mock()
                mock_stdin.buffer = Mock()

                mock_wrapped_stdout = AsyncMock()
                mock_wrapped_stdin = AsyncIteratorMock([message_line])

                with patch("anyio.wrap_file") as mock_wrap:
                    mock_wrap.side_effect = [mock_wrapped_stdin, mock_wrapped_stdout]

                    mock_handler = AsyncMock()
                    mock_handler.return_value = {"jsonrpc": JSON_RPC_VERSION, "id": "test", "result": {}}

                    await stdio_transport(mock_handler)

        # Check that expected log messages were generated
        log_messages = [record.message for record in caplog.records]
        assert any("Handling incoming message" in msg for msg in log_messages)
        assert any("Writing response message" in msg for msg in log_messages)

    @pytest.mark.asyncio
    async def test_concurrent_message_handling(self, sample_message):
        """Test that messages are handled concurrently using task group."""
        # Create multiple messages
        messages = [{**sample_message, "id": f"msg{i}"} for i in range(3)]
        message_lines = [json.dumps(msg) for msg in messages]

        with patch("sys.stdout") as mock_stdout, patch("sys.stdin") as mock_stdin:
            mock_stdout.buffer = Mock()
            mock_stdin.buffer = Mock()

            mock_wrapped_stdout = AsyncMock()
            mock_wrapped_stdin = AsyncIteratorMock(message_lines)

            with patch("anyio.wrap_file") as mock_wrap:
                mock_wrap.side_effect = [mock_wrapped_stdin, mock_wrapped_stdout]

                # Handler that takes some time to process
                mock_handler = AsyncMock()

                async def slow_handler(*args, **kwargs):
                    await anyio.sleep(0.001)  # Small delay
                    return {"jsonrpc": JSON_RPC_VERSION, "id": args[0]["id"], "result": {}}

                mock_handler.side_effect = slow_handler

                await stdio_transport(mock_handler)

                # All messages should have been processed
                assert mock_handler.call_count == len(messages)
                assert mock_handler.call_args_list[0][0][0] == messages[0]
                assert mock_handler.call_args_list[1][0][0] == messages[1]
                assert mock_handler.call_args_list[2][0][0] == messages[2]

    @pytest.mark.asyncio
    async def test_concurrent_message_handling_reverse_return_order(self, sample_message):
        """Test that messages are handled concurrently using task group."""
        # Create multiple messages
        messages = [{**sample_message, "id": f"msg{i}", "sleep": 0.04 - i * 0.01} for i in range(3)]
        message_lines = [json.dumps(msg) for msg in messages]

        with patch("sys.stdout") as mock_stdout, patch("sys.stdin") as mock_stdin:
            mock_stdout.buffer = Mock()
            mock_stdin.buffer = Mock()

            mock_wrapped_stdout = AsyncMock()
            mock_wrapped_stdin = AsyncIteratorMock(message_lines)

            with patch("anyio.wrap_file") as mock_wrap:
                mock_wrap.side_effect = [mock_wrapped_stdin, mock_wrapped_stdout]

                async def slow_handler(message, _):
                    await anyio.sleep(message["sleep"])  # Small delay in decreasing order
                    return {"jsonrpc": JSON_RPC_VERSION, "id": message["id"], "result": {"sleep": message["sleep"]}}

                await stdio_transport(slow_handler)

                # All messages should have been processed
                written_data = mock_wrapped_stdout.write.call_args_list

                assert len(written_data) == 3
                assert json.loads(written_data[0][0][0])["id"] == messages[2]["id"]
                assert json.loads(written_data[1][0][0])["id"] == messages[1]["id"]
                assert json.loads(written_data[2][0][0])["id"] == messages[0]["id"]

    @pytest.mark.asyncio
    async def test_write_msg_with_flush(self):
        """Test that write_msg properly flushes output."""
        test_message = {"test": "flush"}

        with patch("sys.stdout") as mock_stdout, patch("sys.stdin") as mock_stdin:
            mock_stdout.buffer = Mock()
            mock_stdin.buffer = Mock()

            mock_wrapped_stdout = AsyncMock()
            mock_wrapped_stdin = AsyncIteratorMock(['{"test": "input"}'])

            with patch("anyio.wrap_file") as mock_wrap:
                mock_wrap.side_effect = [mock_wrapped_stdin, mock_wrapped_stdout]

                captured_write_msg = None

                async def capture_handler(message, write_msg):
                    nonlocal captured_write_msg
                    captured_write_msg = write_msg
                    return test_message

                mock_handler = AsyncMock(side_effect=capture_handler)

                await stdio_transport(mock_handler)

                # Verify flush was called
                mock_wrapped_stdout.flush.assert_called()
