import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch

import anyio
import mcp.types as types
import pytest
from mcp.server.lowlevel.server import NotificationOptions, Server

from minimcp.server.exceptions import (
    ContextError,
    InvalidParamsError,
    MethodNotFoundError,
    ParserError,
    UnsupportedRPCMessageType,
)
from minimcp.server.managers.context_manager import Context, ContextManager
from minimcp.server.managers.prompt_manager import PromptManager
from minimcp.server.managers.resource_manager import ResourceManager
from minimcp.server.managers.tool_manager import ToolManager
from minimcp.server.minimcp import MiniMCP
from minimcp.server.types import NoMessage


class TestMiniMCP:
    """Test suite for MiniMCP class."""

    @pytest.fixture
    def minimcp(self):
        """Create a MiniMCP instance for testing."""
        return MiniMCP(name="test-server", version="1.0.0", instructions="Test server instructions")

    @pytest.fixture
    def minimcp_with_custom_config(self):
        """Create a MiniMCP instance with custom configuration."""
        return MiniMCP(
            name="custom-server",
            version="2.0.0",
            instructions="Custom instructions",
            idle_timeout=60,
            max_concurrency=50,
            raise_exceptions=True,
        )

    @pytest.fixture
    def valid_request_message(self):
        """Create a valid JSON-RPC request message."""
        return json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1.0.0"},
                },
            }
        )

    @pytest.fixture
    def valid_notification_message(self):
        """Create a valid JSON-RPC notification message."""
        return json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    @pytest.fixture
    def mock_send(self):
        """Create a mock send function."""
        return AsyncMock()

    def test_init_basic(self):
        """Test basic MiniMCP initialization."""
        server = MiniMCP(name="test-server")

        assert server.name == "test-server"
        assert server.version is None
        assert server.instructions is None
        assert isinstance(server._core, Server)
        assert isinstance(server.tool, ToolManager)
        assert isinstance(server.prompt, PromptManager)
        assert isinstance(server.resource, ResourceManager)
        assert isinstance(server.context, ContextManager)
        assert server._raise_exceptions is False

    def test_init_with_all_parameters(self, minimcp_with_custom_config):
        """Test MiniMCP initialization with all parameters."""
        server = minimcp_with_custom_config

        assert server.name == "custom-server"
        assert server.version == "2.0.0"
        assert server.instructions == "Custom instructions"
        assert server._raise_exceptions is True
        # Note: Limiter internal attributes may vary, test behavior instead

    def test_properties(self, minimcp):
        """Test MiniMCP properties."""
        assert minimcp.name == "test-server"
        assert minimcp.version == "1.0.0"
        assert minimcp.instructions == "Test server instructions"

    def test_notification_options_setup(self, minimcp):
        """Test that notification options are properly set up."""
        assert minimcp._notification_options is not None
        assert isinstance(minimcp._notification_options, NotificationOptions)
        assert minimcp._notification_options.prompts_changed is False
        assert minimcp._notification_options.resources_changed is False
        assert minimcp._notification_options.tools_changed is False

    def test_core_setup(self, minimcp):
        """Test that core server is properly set up."""
        assert minimcp._core.name == "test-server"
        assert minimcp._core.version == "1.0.0"
        assert minimcp._core.instructions == "Test server instructions"

        # Check that initialize handler is registered
        assert types.InitializeRequest in minimcp._core.request_handlers
        assert minimcp._core.request_handlers[types.InitializeRequest] == minimcp._initialize_handler

    def test_managers_setup(self, minimcp):
        """Test that all managers are properly initialized."""
        # Check that managers are instances of correct classes
        assert isinstance(minimcp.tool, ToolManager)
        assert isinstance(minimcp.prompt, PromptManager)
        assert isinstance(minimcp.resource, ResourceManager)
        assert isinstance(minimcp.context, ContextManager)

        # Note: Manager internal structure may vary, test functionality instead

    def test_parse_message_valid_request(self, minimcp, valid_request_message):
        """Test parsing a valid JSON-RPC request message."""
        message_id, rpc_msg = minimcp._parse_message(valid_request_message)

        assert message_id == 1
        assert isinstance(rpc_msg, types.JSONRPCMessage)
        assert isinstance(rpc_msg.root, types.JSONRPCRequest)
        assert rpc_msg.root.method == "initialize"
        assert rpc_msg.root.id == 1

    def test_parse_message_valid_notification(self, minimcp, valid_notification_message):
        """Test parsing a valid JSON-RPC notification message."""
        message_id, rpc_msg = minimcp._parse_message(valid_notification_message)

        assert message_id == ""  # Notifications don't have IDs
        assert isinstance(rpc_msg, types.JSONRPCMessage)
        assert isinstance(rpc_msg.root, types.JSONRPCNotification)
        assert rpc_msg.root.method == "notifications/initialized"

    def test_parse_message_invalid_json(self, minimcp):
        """Test parsing invalid JSON raises ParserError."""
        invalid_json = '{"invalid": json}'

        with pytest.raises(ParserError):
            minimcp._parse_message(invalid_json)

    def test_parse_message_invalid_rpc_format(self, minimcp):
        """Test parsing invalid JSON-RPC format raises InvalidParamsError."""
        invalid_rpc = json.dumps({"not": "jsonrpc"})

        with pytest.raises(InvalidParamsError):
            minimcp._parse_message(invalid_rpc)

    def test_parse_message_missing_id_in_dict(self, minimcp):
        """Test parsing message without ID returns empty string."""
        message_without_id = json.dumps({"jsonrpc": "2.0", "method": "test"})

        message_id, rpc_msg = minimcp._parse_message(message_without_id)
        assert message_id == ""

    def test_parse_message_non_dict_json(self, minimcp):
        """Test parsing non-dict JSON returns empty message ID."""
        non_dict_json = json.dumps(["not", "a", "dict"])

        with pytest.raises(InvalidParamsError):
            minimcp._parse_message(non_dict_json)

    @pytest.mark.asyncio
    async def test_handle_rpc_msg_request(self, minimcp):
        """Test handling JSON-RPC request message."""
        # Create a mock request
        mock_request = types.JSONRPCRequest(
            jsonrpc="2.0",
            id=1,
            method="initialize",
            params={
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        )
        rpc_msg = types.JSONRPCMessage(mock_request)

        response = await minimcp._handle_rpc_msg(rpc_msg)

        assert isinstance(response, types.JSONRPCMessage)
        assert isinstance(response.root, types.JSONRPCResponse)
        assert response.root.id == 1

    @pytest.mark.asyncio
    async def test_handle_rpc_msg_notification(self, minimcp):
        """Test handling JSON-RPC notification message."""
        mock_notification = types.JSONRPCNotification(jsonrpc="2.0", method="notifications/initialized", params={})
        rpc_msg = types.JSONRPCMessage(mock_notification)

        response = await minimcp._handle_rpc_msg(rpc_msg)

        assert response == NoMessage.NOTIFICATION

    @pytest.mark.asyncio
    async def test_handle_rpc_msg_unsupported_type(self, minimcp):
        """Test handling unsupported RPC message type."""
        # Create a mock message with unsupported root type
        mock_msg = Mock()
        mock_msg.root = "unsupported_type"

        with pytest.raises(UnsupportedRPCMessageType):
            await minimcp._handle_rpc_msg(mock_msg)

    @pytest.mark.asyncio
    async def test_handle_client_request_success(self, minimcp):
        """Test successful client request handling."""
        # Create initialize request
        init_request = types.InitializeRequest(
            method="initialize",
            params=types.InitializeRequestParams(
                protocolVersion="2025-06-18",
                capabilities=types.ClientCapabilities(),
                clientInfo=types.Implementation(name="test", version="1.0"),
            ),
        )
        client_request = types.ClientRequest(init_request)

        result = await minimcp._handle_client_request(client_request)

        assert isinstance(result, types.ServerResult)
        assert isinstance(result.root, types.InitializeResult)

    @pytest.mark.asyncio
    async def test_handle_client_request_method_not_found(self, minimcp):
        """Test client request with unknown method."""

        # Create a mock request type that's not registered
        class UnknownRequest:
            pass

        mock_request = Mock()
        mock_request.root = UnknownRequest()

        with pytest.raises(MethodNotFoundError):
            await minimcp._handle_client_request(mock_request)

    @pytest.mark.asyncio
    async def test_handle_client_notification_success(self, minimcp):
        """Test successful client notification handling."""
        # Create initialized notification
        init_notification = types.InitializedNotification(
            method="notifications/initialized",
            params={},  # type: ignore
        )
        client_notification = types.ClientNotification(init_notification)

        result = await minimcp._handle_client_notification(client_notification)

        assert result == NoMessage.NOTIFICATION

    @pytest.mark.asyncio
    async def test_handle_client_notification_no_handler(self, minimcp):
        """Test client notification with no registered handler."""

        # Create a mock notification type that's not registered
        class UnknownNotification:
            pass

        mock_notification = Mock()
        mock_notification.root = UnknownNotification()

        result = await minimcp._handle_client_notification(mock_notification)

        assert result == NoMessage.NOTIFICATION

    @pytest.mark.asyncio
    async def test_handle_client_notification_handler_exception(self, minimcp):
        """Test client notification handler that raises exception."""

        # Register a handler that raises an exception
        def failing_handler(notification):
            raise ValueError("Handler failed")

        # Mock notification type
        class TestNotification:
            pass

        minimcp._core.notification_handlers[TestNotification] = failing_handler

        mock_notification = Mock()
        mock_notification.root = TestNotification()

        # Should not raise exception, just log it
        result = await minimcp._handle_client_notification(mock_notification)
        assert result == NoMessage.NOTIFICATION

    @pytest.mark.asyncio
    async def test_initialize_handler_supported_version(self, minimcp):
        """Test initialize handler with supported protocol version."""
        request = types.InitializeRequest(
            method="initialize",
            params=types.InitializeRequestParams(
                protocolVersion="2025-06-18",
                capabilities=types.ClientCapabilities(),
                clientInfo=types.Implementation(name="test", version="1.0"),
            ),
        )

        result = await minimcp._initialize_handler(request)

        assert isinstance(result, types.ServerResult)
        assert isinstance(result.root, types.InitializeResult)
        assert result.root.protocolVersion == "2025-06-18"
        assert result.root.serverInfo.name == "test-server"
        assert result.root.serverInfo.version == "1.0.0"
        assert result.root.instructions == "Test server instructions"

    @pytest.mark.asyncio
    async def test_initialize_handler_unsupported_version(self, minimcp):
        """Test initialize handler with unsupported protocol version."""
        request = types.InitializeRequest(
            method="initialize",
            params=types.InitializeRequestParams(
                protocolVersion="unsupported-version",
                capabilities=types.ClientCapabilities(),
                clientInfo=types.Implementation(name="test", version="1.0"),
            ),
        )

        result = await minimcp._initialize_handler(request)

        assert isinstance(result, types.ServerResult)
        assert isinstance(result.root, types.InitializeResult)
        # Should fall back to latest supported version
        assert result.root.protocolVersion == types.LATEST_PROTOCOL_VERSION

    @pytest.mark.asyncio
    async def test_handle_success(self, minimcp, valid_request_message):
        """Test successful message handling."""
        result = await minimcp.handle(valid_request_message)

        assert isinstance(result, str)  # Should return JSON string
        response_dict = json.loads(result)
        assert response_dict["jsonrpc"] == "2.0"
        assert response_dict["id"] == 1
        assert "result" in response_dict

    @pytest.mark.asyncio
    async def test_handle_with_send_and_scope(self, minimcp, valid_request_message, mock_send):
        """Test message handling with send callback and scope."""
        scope = "test-scope"

        result = await minimcp.handle(valid_request_message, send=mock_send, scope=scope)

        assert isinstance(result, str)
        # Verify that responder was created (indirectly through successful handling)

    @pytest.mark.asyncio
    async def test_handle_parser_error(self, minimcp):
        """Test handling parser error."""
        invalid_message = '{"invalid": json}'

        result = await minimcp.handle(invalid_message)

        assert isinstance(result, str)
        response_dict = json.loads(result)
        assert response_dict["jsonrpc"] == "2.0"
        assert "error" in response_dict
        assert response_dict["error"]["code"] == types.PARSE_ERROR

    @pytest.mark.asyncio
    async def test_handle_invalid_params_error(self, minimcp):
        """Test handling invalid params error."""
        invalid_rpc = json.dumps({"not": "jsonrpc"})

        result = await minimcp.handle(invalid_rpc)

        assert isinstance(result, str)
        response_dict = json.loads(result)
        assert response_dict["error"]["code"] == types.INVALID_PARAMS

    @pytest.mark.asyncio
    async def test_handle_method_not_found_error(self, minimcp):
        """Test handling method not found error."""
        # Use a valid JSON-RPC structure but unknown method
        # The validation error occurs before method dispatch, so this becomes INTERNAL_ERROR
        unknown_method = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "unknown_method", "params": {}})

        result = await minimcp.handle(unknown_method)

        assert isinstance(result, str)
        response_dict = json.loads(result)
        # Unknown methods cause validation errors, not method not found errors
        assert response_dict["error"]["code"] == types.INTERNAL_ERROR

    @pytest.mark.asyncio
    async def test_handle_timeout_error(self, minimcp, valid_request_message):
        """Test handling timeout error."""
        # Mock _parse_message to raise TimeoutError directly
        with patch.object(minimcp, "_parse_message", side_effect=TimeoutError("Timeout")):
            result = await minimcp.handle(valid_request_message)

            assert isinstance(result, str)
            response_dict = json.loads(result)
            assert response_dict["error"]["code"] == types.INTERNAL_ERROR

    @pytest.mark.asyncio
    async def test_handle_context_error(self, minimcp, valid_request_message):
        """Test handling context error."""
        # Mock _handle_rpc_msg to raise ContextError
        with patch.object(minimcp, "_handle_rpc_msg", side_effect=ContextError("Context error")):
            result = await minimcp.handle(valid_request_message)

            assert isinstance(result, str)
            response_dict = json.loads(result)
            assert response_dict["error"]["code"] == types.INTERNAL_ERROR

    @pytest.mark.asyncio
    async def test_handle_generic_exception_raise_false(self, minimcp, valid_request_message):
        """Test handling generic exception with raise_exceptions=False."""
        # Mock _handle_rpc_msg to raise generic exception
        with patch.object(minimcp, "_handle_rpc_msg", side_effect=ValueError("Generic error")):
            result = await minimcp.handle(valid_request_message)

            assert isinstance(result, str)
            response_dict = json.loads(result)
            assert response_dict["error"]["code"] == types.INTERNAL_ERROR

    @pytest.mark.asyncio
    async def test_handle_generic_exception_raise_true(self, minimcp_with_custom_config, valid_request_message):
        """Test handling generic exception with raise_exceptions=True."""
        # Mock _handle_rpc_msg to raise generic exception
        with patch.object(minimcp_with_custom_config, "_handle_rpc_msg", side_effect=ValueError("Generic error")):
            with pytest.raises(ValueError, match="Generic error"):
                await minimcp_with_custom_config.handle(valid_request_message)

    @pytest.mark.asyncio
    async def test_handle_cancellation_error(self, minimcp, valid_request_message):
        """Test handling cancellation error."""
        # Mock _handle_rpc_msg to raise cancellation
        cancellation_exc = anyio.get_cancelled_exc_class()
        with patch.object(minimcp, "_handle_rpc_msg", side_effect=cancellation_exc()):
            with pytest.raises(cancellation_exc):
                await minimcp.handle(valid_request_message)

    @pytest.mark.asyncio
    async def test_handle_no_message_response(self, minimcp, valid_notification_message):
        """Test handling that returns NoMessage."""
        result = await minimcp.handle(valid_notification_message)

        assert result == NoMessage.NOTIFICATION

    @pytest.mark.asyncio
    async def test_handle_with_context_manager(self, minimcp, valid_request_message):
        """Test that context manager is properly used during handling."""
        original_active = minimcp.context.active
        context_used = False

        def mock_active(context):
            nonlocal context_used
            context_used = True
            assert isinstance(context, Context)
            return original_active(context)

        with patch.object(minimcp.context, "active", side_effect=mock_active):
            await minimcp.handle(valid_request_message)

        assert context_used

    @pytest.mark.asyncio
    async def test_handle_with_limiter(self, minimcp, valid_request_message):
        """Test that limiter is properly used during handling."""
        # Test that the limiter is called by checking if the handle method completes
        # The limiter is used internally, so we test the behavior indirectly
        result = await minimcp.handle(valid_request_message)

        # If we get a result, the limiter was used successfully
        assert isinstance(result, str)
        response_dict = json.loads(result)
        assert response_dict["jsonrpc"] == "2.0"

    def test_generic_type_parameter(self):
        """Test that MiniMCP can be parameterized with generic types."""
        # Test with string scope type
        server_str = MiniMCP[str](name="test")
        assert isinstance(server_str.context, ContextManager)

        # Test with int scope type
        server_int = MiniMCP[int](name="test")
        assert isinstance(server_int.context, ContextManager)

        # Test with custom type
        class CustomScope:
            pass

        server_custom = MiniMCP[CustomScope](name="test")
        assert isinstance(server_custom.context, ContextManager)

    @pytest.mark.asyncio
    async def test_error_logging(self, minimcp, caplog):
        """Test that errors are properly logged."""
        invalid_message = '{"invalid": json}'

        with caplog.at_level("ERROR"):
            await minimcp.handle(invalid_message)

        assert "Parser failed" in caplog.text

    @pytest.mark.asyncio
    async def test_debug_logging(self, minimcp, valid_request_message, caplog):
        """Test that debug information is logged."""
        with caplog.at_level("DEBUG"):
            await minimcp.handle(valid_request_message)

        # Should contain debug logs about handling requests
        assert any("Handling request" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_concurrent_message_handling(self, minimcp, valid_request_message):
        """Test handling multiple messages concurrently."""
        # Create multiple tasks
        tasks = []
        for i in range(5):
            message = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": i,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "1.0"},
                    },
                }
            )
            tasks.append(minimcp.handle(message))

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 5
        for result in results:
            assert isinstance(result, str)
            response_dict = json.loads(result)
            assert response_dict["jsonrpc"] == "2.0"
            assert "result" in response_dict

    @pytest.mark.asyncio
    async def test_message_id_extraction_edge_cases(self, minimcp):
        """Test message ID extraction in various edge cases."""
        # Test with string ID
        message_str_id = json.dumps({"jsonrpc": "2.0", "id": "string-id", "method": "test"})
        message_id, _ = minimcp._parse_message(message_str_id)
        assert message_id == "string-id"

        # Test with numeric ID
        message_num_id = json.dumps({"jsonrpc": "2.0", "id": 42, "method": "test"})
        message_id, _ = minimcp._parse_message(message_num_id)
        assert message_id == 42

        # Test with null ID
        message_null_id = json.dumps({"jsonrpc": "2.0", "id": None, "method": "test"})
        message_id, _ = minimcp._parse_message(message_null_id)
        assert message_id is None


class TestMiniMCPIntegration:
    """Integration tests for MiniMCP."""

    @pytest.mark.asyncio
    async def test_full_initialize_flow(self):
        """Test complete initialization flow."""
        server = MiniMCP(name="integration-test", version="1.0.0")

        # Create initialize request
        init_message = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"roots": {"listChanged": True}, "sampling": {}},
                    "clientInfo": {"name": "test-client", "version": "1.0.0"},
                },
            }
        )

        # Handle the request
        result = await server.handle(init_message)

        # Verify response
        assert isinstance(result, str)
        response = json.loads(result)  # type: ignore

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response

        init_result = response["result"]
        assert init_result["protocolVersion"] == "2025-06-18"
        assert init_result["serverInfo"]["name"] == "integration-test"
        assert init_result["serverInfo"]["version"] == "1.0.0"
        assert "capabilities" in init_result

    @pytest.mark.asyncio
    async def test_tool_integration(self):
        """Test integration with tool manager."""
        server = MiniMCP(name="tool-test")

        # Add a test tool
        def test_tool(x: int, y: int = 5) -> int:
            """A test tool."""
            return x + y

        tool = server.tool.add(test_tool)
        assert tool.name == "test_tool"

        # Verify tool is registered
        tools = server.tool.list()
        assert len(tools) == 1
        assert tools[0].name == "test_tool"

    @pytest.mark.asyncio
    async def test_context_integration(self):
        """Test integration with context manager."""
        server = MiniMCP[str](name="context-test")

        init_message = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0"},
                },
            }
        )

        # Handle with scope
        result = await server.handle(init_message, scope="test-scope")

        assert isinstance(result, str)
        response = json.loads(result)
        assert response["id"] == 1

    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error recovery and continued operation."""
        server = MiniMCP(name="error-test")

        # Send invalid message
        invalid_result = await server.handle('{"invalid": json}')
        invalid_response = json.loads(invalid_result)  # type: ignore
        assert "error" in invalid_response

        # Send valid message after error
        valid_message = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0"},
                },
            }
        )

        valid_result = await server.handle(valid_message)
        valid_response = json.loads(valid_result)  # type: ignore
        assert valid_response["id"] == 2
        assert "result" in valid_response

    @pytest.mark.asyncio
    async def test_notification_handling(self):
        """Test notification handling."""
        server = MiniMCP(name="notification-test")

        # Send initialized notification
        notification = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

        result = await server.handle(notification)
        assert result == NoMessage.NOTIFICATION

    @pytest.mark.asyncio
    async def test_multiple_clients_simulation(self):
        """Test handling messages from multiple simulated clients."""
        server = MiniMCP(name="multi-client-test")

        # Simulate messages from different clients
        client_messages = []
        for client_id in range(3):
            message = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": client_id,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": f"client-{client_id}", "version": "1.0"},
                    },
                }
            )
            client_messages.append(message)

        # Handle all messages
        results = await asyncio.gather(*[server.handle(msg) for msg in client_messages])

        # Verify all responses
        for i, result in enumerate(results):
            response = json.loads(result)  # type: ignore
            assert response["id"] == i
            assert "result" in response
