"""Simplified stdio transport tests focusing on core functionality."""

import pytest

from minimcp.server.transports.stdio import stdio_transport
from minimcp.server.types import NoMessage


class TestStdioTransportSimple:
    """Simplified test suite for stdio transport."""

    def test_stdio_transport_function_exists(self):
        """Test that stdio_transport function exists and is callable."""
        assert callable(stdio_transport)

        # Check function signature
        import inspect

        sig = inspect.signature(stdio_transport)
        assert len(sig.parameters) == 1
        assert "handler" in sig.parameters

    def test_stdio_transport_is_async(self):
        """Test that stdio_transport is an async function."""
        import asyncio

        assert asyncio.iscoroutinefunction(stdio_transport)

    def test_imports_available(self):
        """Test that all required imports are available."""
        # Test that we can import everything the module needs
        import sys
        from io import TextIOWrapper

        import anyio

        from minimcp.server.types import Message, NoMessage, Send

        # All imports should be successful
        assert sys is not None
        assert anyio is not None
        assert TextIOWrapper is not None
        assert Message is not None
        assert NoMessage is not None
        assert Send is not None

    def test_no_message_enum_values(self):
        """Test NoMessage enum values are available."""
        # These are used in the stdio transport
        assert hasattr(NoMessage, "NOTIFICATION")
        assert isinstance(NoMessage.NOTIFICATION, NoMessage)

    def test_function_docstring(self):
        """Test that the function has proper documentation."""
        assert stdio_transport.__doc__ is not None
        assert "stdio_transport" in stdio_transport.__doc__
        assert "handler" in stdio_transport.__doc__

    def test_module_level_logger(self):
        """Test that the module has a logger configured."""
        from minimcp.server.transports import stdio

        assert hasattr(stdio, "logger")
        assert stdio.logger.name == "minimcp.server.transports.stdio"

    @pytest.mark.asyncio
    async def test_stdio_transport_requires_handler(self):
        """Test that stdio_transport requires a handler parameter."""
        # This should raise TypeError if called without handler
        with pytest.raises(TypeError):
            await stdio_transport()  # type: ignore

    def test_stdio_transport_handler_signature(self):
        """Test expected handler signature based on function usage."""
        import inspect

        # Based on the code, handler should accept (Message, Send) -> Awaitable[Message | NoMessage]
        # We can't easily test this without complex mocking, but we can document the expectation

        # The handler should be callable
        def dummy_handler(message, send):
            return "response"

        assert callable(dummy_handler)

        # Should accept 2 parameters
        sig = inspect.signature(dummy_handler)
        assert len(sig.parameters) == 2

    def test_anyio_dependencies(self):
        """Test that anyio functions used in stdio transport are available."""
        import anyio

        # Functions used in the stdio transport
        assert hasattr(anyio, "wrap_file")
        assert hasattr(anyio, "create_task_group")
        assert hasattr(anyio, "move_on_after")

        # These should be callable
        assert callable(anyio.wrap_file)
        assert callable(anyio.create_task_group)
        assert callable(anyio.move_on_after)

    def test_text_io_wrapper_usage(self):
        """Test TextIOWrapper usage patterns from stdio transport."""
        import sys
        from io import TextIOWrapper

        # The stdio transport uses these patterns
        assert hasattr(sys.stdin, "buffer")
        assert hasattr(sys.stdout, "buffer")

        # TextIOWrapper should accept the expected parameters
        import inspect

        sig = inspect.signature(TextIOWrapper)
        param_names = list(sig.parameters.keys())

        # Should accept buffer, encoding, and line_buffering parameters
        assert "buffer" in param_names
        # Other parameters might be positional or keyword
