from unittest.mock import Mock

import mcp.types as types
import pytest

from minimcp.server.exceptions import ContextError
from minimcp.server.limiter import TimeLimiter
from minimcp.server.managers.context_manager import Context, ContextManager
from minimcp.server.responder import Responder


class TestContext:
    """Test suite for Context dataclass."""

    def test_context_creation_minimal(self):
        """Test creating a Context with minimal required fields."""
        message = types.JSONRPCMessage(types.JSONRPCRequest(method="test", id=1, jsonrpc="2.0"))
        time_limiter = Mock(spec=TimeLimiter)

        context = Context(message=message, time_limiter=time_limiter)

        assert context.message == message
        assert context.time_limiter == time_limiter
        assert context.scope is None
        assert context.responder is None

    def test_context_creation_with_all_fields(self):
        """Test creating a Context with all fields."""
        message = types.JSONRPCMessage(types.JSONRPCRequest(method="test", id=1, jsonrpc="2.0"))
        time_limiter = Mock(spec=TimeLimiter)
        scope = {"test": "scope"}
        responder = Mock(spec=Responder)

        context = Context(message=message, time_limiter=time_limiter, scope=scope, responder=responder)

        assert context.message == message
        assert context.time_limiter == time_limiter
        assert context.scope == scope
        assert context.responder == responder


class TestContextManager:
    """Test suite for ContextManager class."""

    @pytest.fixture
    def context_manager(self):
        """Create a ContextManager instance."""
        return ContextManager()

    @pytest.fixture
    def sample_context(self):
        """Create a sample Context for testing."""
        message = types.JSONRPCMessage(types.JSONRPCRequest(method="test", id=1, jsonrpc="2.0"))
        time_limiter = Mock(spec=TimeLimiter)
        scope = {"test": "scope"}
        responder = Mock(spec=Responder)

        return Context(message=message, time_limiter=time_limiter, scope=scope, responder=responder)

    def test_get_without_active_context_raises_error(self, context_manager: ContextManager):
        """Test that get() raises ContextError when no context is active."""
        with pytest.raises(ContextError, match="No Context: Called get outside of an active context"):
            context_manager.get()

    def test_get_message_without_active_context_raises_error(self, context_manager: ContextManager):
        """Test that get_message() raises ContextError when no context is active."""
        with pytest.raises(ContextError, match="No Context: Called get outside of an active context"):
            context_manager.get_message()

    def test_get_scope_without_active_context_raises_error(self, context_manager: ContextManager):
        """Test that get_scope() raises ContextError when no context is active."""
        with pytest.raises(ContextError, match="No Context: Called get outside of an active context"):
            context_manager.get_scope()

    def test_get_responder_without_active_context_raises_error(self, context_manager: ContextManager):
        """Test that get_responder() raises ContextError when no context is active."""
        with pytest.raises(ContextError, match="No Context: Called get outside of an active context"):
            context_manager.get_responder()

    def test_active_context_manager(self, context_manager: ContextManager, sample_context: Context):
        """Test the active context manager sets and clears context properly."""
        # Verify no context initially
        with pytest.raises(ContextError):
            context_manager.get()

        # Use context manager
        with context_manager.active(sample_context):
            # Context should be available
            retrieved_context = context_manager.get()
            assert retrieved_context == sample_context

        # Context should be cleared after exiting
        with pytest.raises(ContextError):
            context_manager.get()

    def test_get_within_active_context(self, context_manager: ContextManager, sample_context: Context):
        """Test that get() returns the active context."""
        with context_manager.active(sample_context):
            retrieved_context = context_manager.get()
            assert retrieved_context == sample_context
            assert retrieved_context.message == sample_context.message
            assert retrieved_context.time_limiter == sample_context.time_limiter
            assert retrieved_context.scope == sample_context.scope
            assert retrieved_context.responder == sample_context.responder

    def test_get_message_within_active_context(self, context_manager: ContextManager, sample_context: Context):
        """Test that get_message() returns the message from active context."""
        with context_manager.active(sample_context):
            message = context_manager.get_message()
            assert message == sample_context.message

    def test_get_scope_within_active_context(self, context_manager: ContextManager, sample_context: Context):
        """Test that get_scope() returns the scope from active context."""
        with context_manager.active(sample_context):
            scope = context_manager.get_scope()
            assert scope == sample_context.scope

    def test_get_scope_when_scope_is_none_raises_error(self, context_manager: ContextManager):
        """Test that get_scope() raises ContextError when scope is None."""
        message = types.JSONRPCMessage(types.JSONRPCRequest(method="test", id=1, jsonrpc="2.0"))
        time_limiter = Mock(spec=TimeLimiter)
        context_without_scope = Context(message=message, time_limiter=time_limiter, scope=None)

        with context_manager.active(context_without_scope):
            with pytest.raises(ContextError, match="Scope is not available in current context"):
                context_manager.get_scope()

    def test_get_responder_within_active_context(self, context_manager: ContextManager, sample_context: Context):
        """Test that get_responder() returns the responder from active context."""
        with context_manager.active(sample_context):
            responder = context_manager.get_responder()
            assert responder == sample_context.responder

    def test_get_responder_when_responder_is_none_raises_error(self, context_manager: ContextManager):
        """Test that get_responder() raises ContextError when responder is None."""
        message = types.JSONRPCMessage(types.JSONRPCRequest(method="test", id=1, jsonrpc="2.0"))
        time_limiter = Mock(spec=TimeLimiter)
        context_without_responder = Context(message=message, time_limiter=time_limiter, responder=None)

        with context_manager.active(context_without_responder):
            with pytest.raises(ContextError, match="Responder is not available in current context"):
                context_manager.get_responder()

    def test_nested_context_managers(self, context_manager: ContextManager):
        """Test nested context managers work correctly."""
        message1 = types.JSONRPCMessage(types.JSONRPCRequest(method="test1", id=1, jsonrpc="2.0"))
        message2 = types.JSONRPCMessage(types.JSONRPCRequest(method="test2", id=2, jsonrpc="2.0"))
        time_limiter = Mock(spec=TimeLimiter)

        context1 = Context(message=message1, time_limiter=time_limiter, scope="scope1")
        context2 = Context(message=message2, time_limiter=time_limiter, scope="scope2")

        with context_manager.active(context1):
            assert context_manager.get_scope() == "scope1"

            with context_manager.active(context2):
                assert context_manager.get_scope() == "scope2"

            # Should return to outer context
            assert context_manager.get_scope() == "scope1"

    def test_context_manager_exception_handling(self, context_manager: ContextManager, sample_context: Context):
        """Test that context is properly cleared even when exception occurs."""
        with pytest.raises(ValueError):
            with context_manager.active(sample_context):
                # Context should be available
                assert context_manager.get() == sample_context
                # Raise exception
                raise ValueError("Test exception")

        # Context should be cleared even after exception
        with pytest.raises(ContextError):
            context_manager.get()

    def test_multiple_context_managers_share_context_var(self):
        """Test that multiple ContextManager instances share the same ContextVar (by design)."""
        manager1 = ContextManager()
        manager2 = ContextManager()

        message1 = types.JSONRPCMessage(types.JSONRPCRequest(method="test1", id=1, jsonrpc="2.0"))
        message2 = types.JSONRPCMessage(types.JSONRPCRequest(method="test2", id=2, jsonrpc="2.0"))
        time_limiter = Mock(spec=TimeLimiter)

        context1 = Context(message=message1, time_limiter=time_limiter, scope="scope1")
        context2 = Context(message=message2, time_limiter=time_limiter, scope="scope2")

        # ContextVar is shared across instances, so the last set context wins
        with manager1.active(context1):
            with manager2.active(context2):
                # Both managers see the same context (context2) since they share the ContextVar
                assert manager1.get_scope() == "scope2"
                assert manager2.get_scope() == "scope2"

    def test_context_var_isolation(self, context_manager: ContextManager):
        """Test that ContextVar properly isolates contexts across different execution contexts."""
        import asyncio

        message1 = types.JSONRPCMessage(types.JSONRPCRequest(method="test1", id=1, jsonrpc="2.0"))
        message2 = types.JSONRPCMessage(types.JSONRPCRequest(method="test2", id=2, jsonrpc="2.0"))
        time_limiter = Mock(spec=TimeLimiter)

        context1 = Context(message=message1, time_limiter=time_limiter, scope="scope1")
        context2 = Context(message=message2, time_limiter=time_limiter, scope="scope2")

        async def task1():
            with context_manager.active(context1):
                await asyncio.sleep(0.01)
                return context_manager.get_scope()

        async def task2():
            with context_manager.active(context2):
                await asyncio.sleep(0.01)
                return context_manager.get_scope()

        async def run_test():
            results = await asyncio.gather(task1(), task2())
            assert results[0] == "scope1"
            assert results[1] == "scope2"

        asyncio.run(run_test())
