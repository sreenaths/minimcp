import asyncio
from typing import Any
from unittest.mock import Mock

import mcp.types as types
import pytest
from mcp.server.lowlevel.server import Server
from pydantic import ValidationError

from minimcp.server.managers.tool_manager import ToolDetails, ToolManager


class TestToolManager:
    """Test suite for ToolManager class."""

    @pytest.fixture
    def mock_core(self) -> Mock:
        """Create a mock Server for testing."""
        core = Mock(spec=Server)
        core.list_tools = Mock(return_value=Mock())
        core.call_tool = Mock(return_value=Mock())
        return core

    @pytest.fixture
    def tool_manager(self, mock_core: Mock) -> ToolManager:
        """Create a ToolManager instance with mocked core."""
        return ToolManager(mock_core)

    def test_init_hooks_core_methods(self, mock_core: Mock):
        """Test that ToolManager properly hooks into Server methods."""
        tool_manager = ToolManager(mock_core)

        # Verify that core methods were called to register handlers
        mock_core.list_tools.assert_called_once()
        mock_core.call_tool.assert_called_once_with(validate_input=False)

        # Verify internal state
        assert tool_manager._tools == {}

    def test_add_tool_basic_function(self, tool_manager: ToolManager):
        """Test adding a basic function as a tool."""

        def sample_add_tool(a: int, b: int) -> int:
            """A sample tool for testing."""
            return a + b

        result = tool_manager.add(sample_add_tool)

        # Verify the returned tool
        assert isinstance(result, types.Tool)
        assert result.name == "sample_add_tool"
        assert result.description == "A sample tool for testing."
        assert result.inputSchema is not None
        assert result.annotations is None
        assert result.meta is None

        # Verify internal state
        assert "sample_add_tool" in tool_manager._tools
        tool, func, func_meta = tool_manager._tools["sample_add_tool"]
        assert tool == result
        assert func == sample_add_tool
        assert func_meta is not None

    def test_add_tool_with_custom_details(self, tool_manager: ToolManager):
        """Test adding a tool with custom name, description, and metadata."""

        def basic_func(value: int) -> int:
            return value * 2

        custom_annotations = types.ToolAnnotations(title="math")
        custom_meta = {"version": "1.0"}

        result = tool_manager.add(
            basic_func,
            name="custom_name",
            description="Custom description",
            annotations=custom_annotations,
            meta=custom_meta,
        )

        assert result.name == "custom_name"
        assert result.description == "Custom description"
        assert result.annotations == custom_annotations
        assert result.meta == custom_meta

        # Verify it's stored with custom name
        assert "custom_name" in tool_manager._tools
        assert "basic_func" not in tool_manager._tools

    def test_add_tool_without_docstring(self, tool_manager: ToolManager):
        """Test adding a tool without docstring uses None as description."""

        def no_doc_tool(x: int) -> int:
            return x

        result = tool_manager.add(no_doc_tool)
        assert result.description is None

    def test_add_async_tool(self, tool_manager: ToolManager):
        """Test adding an async function as a tool."""

        async def async_tool(delay: float) -> str:
            """An async tool."""
            await asyncio.sleep(delay)
            return "done"

        result = tool_manager.add(async_tool)

        assert result.name == "async_tool"
        assert result.description == "An async tool."
        assert "async_tool" in tool_manager._tools

    def test_add_duplicate_tool_raises_error(self, tool_manager: ToolManager):
        """Test that adding a tool with duplicate name raises ValueError."""

        def tool1(x: int) -> int:
            return x

        def tool2(y: str) -> str:
            return y

        # Add first tool
        tool_manager.add(tool1, name="duplicate_name")

        # Adding second tool with same name should raise error
        with pytest.raises(ValueError, match="Tool duplicate_name already registered"):
            tool_manager.add(tool2, name="duplicate_name")

    def test_add_again_tool_raises_error(self, tool_manager: ToolManager):
        """Test that adding a tool with duplicate name raises ValueError."""

        def tool1(x: int) -> int:
            return x

        # Add tool
        tool_manager.add(tool1)

        # Adding tool again should raise error
        with pytest.raises(ValueError, match="Tool tool1 already registered"):
            tool_manager.add(tool1)

    def test_add_duplicate_function_name_raises_error(self, tool_manager: ToolManager):
        """Test that adding functions with same name raises ValueError."""

        def same_name(x: int) -> int:
            return x

        def same_name_different_func(y: str) -> str:  # Different function, same name
            return y

        # Add first function
        tool_manager.add(same_name)

        # This should work since we can give it a different name
        tool_manager.add(same_name_different_func, name="different_name")

        def different_scope_same_name():
            # But this should fail since it uses the function name
            with pytest.raises(ValueError, match="Tool same_name already registered"):
                # Create another function with same name
                def same_name(z: float) -> float:
                    return z

                tool_manager.add(same_name)

        different_scope_same_name()

    def test_remove_existing_tool(self, tool_manager: ToolManager):
        """Test removing an existing tool."""

        def test_tool(x: int) -> int:
            return x

        # Add tool first
        added_tool = tool_manager.add(test_tool)
        assert "test_tool" in tool_manager._tools

        # Remove the tool
        removed_tool = tool_manager.remove("test_tool")

        assert removed_tool == added_tool
        assert "test_tool" not in tool_manager._tools

    def test_remove_nonexistent_tool_raises_error(self, tool_manager: ToolManager):
        """Test that removing a non-existent tool raises ValueError."""
        with pytest.raises(ValueError, match="Tool nonexistent not found"):
            tool_manager.remove("nonexistent")

    @pytest.mark.asyncio
    async def test_list_tools_empty(self, tool_manager: ToolManager):
        """Test listing tools when no tools are registered."""
        result = tool_manager.list()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_tools_with_multiple_tools(self, tool_manager: ToolManager):
        """Test listing tools when multiple tools are registered."""

        def tool1(x: int) -> int:
            return x

        def tool2(y: str) -> str:
            return y

        added_tool1 = tool_manager.add(tool1)
        added_tool2 = tool_manager.add(tool2)

        result = tool_manager.list()

        assert len(result) == 2
        assert added_tool1 in result
        assert added_tool2 in result

    @pytest.mark.asyncio
    async def test_call_tool_sync_function(self, tool_manager: ToolManager):
        """Test calling a synchronous tool."""

        def multiply(x: int, y: int) -> int:
            """Multiply two numbers."""
            return x * y

        tool_manager.add(multiply)

        result = await tool_manager.call("multiply", {"x": 3, "y": 4})
        assert isinstance(result[0][0], types.TextContent)
        assert result[1]["result"] == 12

    @pytest.mark.asyncio
    async def test_call_tool_async_function(self, tool_manager: ToolManager):
        """Test calling an asynchronous tool."""

        async def async_multiply(x: int, y: int) -> int:
            """Async multiply two numbers."""
            await asyncio.sleep(0.01)  # Small delay to make it actually async
            return x * y

        tool_manager.add(async_multiply)

        result = await tool_manager.call("async_multiply", {"x": 5, "y": 6})
        assert isinstance(result[0][0], types.TextContent)
        assert result[1]["result"] == 30

    @pytest.mark.asyncio
    async def test_call_tool_with_default_arguments(self, tool_manager: ToolManager):
        """Test calling a tool with default arguments."""

        def greet(name: str, greeting: str = "Hello") -> str:
            """Greet someone."""
            return f"{greeting}, {name}!"

        tool_manager.add(greet)

        # Call with just required argument
        result = await tool_manager.call("greet", {"name": "Alice"})
        assert result[1]["result"] == "Hello, Alice!"

        # Call with both arguments
        result = await tool_manager.call("greet", {"name": "Bob", "greeting": "Hi"})
        assert result[1]["result"] == "Hi, Bob!"

    @pytest.mark.asyncio
    async def test_call_nonexistent_tool_raises_error(self, tool_manager: ToolManager):
        """Test that calling a non-existent tool raises ValueError."""
        with pytest.raises(ValueError, match="Tool nonexistent not found"):
            await tool_manager.call("nonexistent", {})

    @pytest.mark.asyncio
    async def test_call_tool_with_complex_return_type(self, tool_manager: ToolManager):
        """Test calling a tool that returns complex data structures."""

        def get_user_info(user_id: int) -> dict[str, Any]:
            """Get user information."""
            return {
                "id": user_id,
                "name": f"User {user_id}",
                "active": True,
                "metadata": {"created": "2024-01-01", "role": "user"},
            }

        tool_manager.add(get_user_info)

        result = await tool_manager.call("get_user_info", {"user_id": 123})
        expected = {
            "id": 123,
            "name": "User 123",
            "active": True,
            "metadata": {"created": "2024-01-01", "role": "user"},
        }
        assert isinstance(result[0][0], types.TextContent)
        assert result[1] == expected

    @pytest.mark.asyncio
    async def test_call_tool_argument_validation(self, tool_manager: ToolManager):
        """Test that tool arguments are properly validated by func_metadata."""

        def strict_tool(required_int: int, optional_str: str = "default") -> str:
            """A tool with strict typing."""
            return f"{required_int}-{optional_str}"

        tool_manager.add(strict_tool)

        # Valid call should work
        result = await tool_manager.call("strict_tool", {"required_int": 42})
        assert result[1]["result"] == "42-default"

        result = await tool_manager.call("strict_tool", {"required_int": "42"})
        assert result[1]["result"] == "42-default"

        # The actual validation happens in func_metadata, so we test that it's called
        # by ensuring the tool works with valid arguments and would fail with invalid ones
        # through the func_metadata validation layer

        with pytest.raises(ValidationError, match="required_int"):
            await tool_manager.call("strict_tool", {"invalid_int": 42})

    def test_tool_details_typed_dict(self):
        """Test ToolDetails TypedDict structure."""
        # This tests the type structure - mainly for documentation
        details: ToolDetails = {
            "name": "test_name",
            "description": "test_description",
            "annotations": types.ToolAnnotations(title="test"),
            "meta": {"version": "1.0"},
        }

        assert details["name"] == "test_name"
        assert details["description"] == "test_description"
        assert details["annotations"] == types.ToolAnnotations(title="test")
        assert details["meta"] == {"version": "1.0"}

    def test_integration_with_func_metadata(self, tool_manager: ToolManager):
        """Test integration with func_metadata for schema generation."""

        def typed_tool(count: int, message: str, active: bool = True) -> dict:
            """A well-typed tool for testing schema generation."""
            return {"count": count, "message": message, "active": active}

        result = tool_manager.add(typed_tool)

        # Verify that inputSchema was generated
        assert result.inputSchema is not None
        schema = result.inputSchema

        # Should have properties for the parameters
        assert "properties" in schema
        properties = schema["properties"]
        assert "count" in properties
        assert "message" in properties
        assert "active" in properties

        # Required should include non-default parameters
        assert "required" in schema
        required = schema["required"]
        assert "count" in required
        assert "message" in required
        assert "active" not in required  # Has default value

    @pytest.mark.asyncio
    async def test_full_workflow(self, tool_manager: ToolManager):
        """Test a complete workflow: add, list, call, remove."""

        def calculator(operation: str, a: float, b: float) -> float:
            """Perform basic calculations."""
            if operation == "add":
                return a + b
            elif operation == "multiply":
                return a * b
            else:
                raise ValueError(f"Unknown operation: {operation}")

        # Add tool
        added_tool = tool_manager.add(calculator, description="Basic calculator")
        assert added_tool.name == "calculator"
        assert added_tool.description == "Basic calculator"

        # List tools
        tools = tool_manager.list()
        assert len(tools) == 1
        assert tools[0] == added_tool

        # Call tool
        result = await tool_manager.call("calculator", {"operation": "add", "a": 10.5, "b": 5.2})
        assert result[1]["result"] == 15.7

        result = await tool_manager.call("calculator", {"operation": "multiply", "a": 3.0, "b": 4.0})
        assert result[1]["result"] == 12.0

        # Remove tool
        removed_tool = tool_manager.remove("calculator")
        assert removed_tool == added_tool

        # Verify it's gone
        tools = tool_manager.list()
        assert len(tools) == 0

        # Calling removed tool should fail
        with pytest.raises(ValueError, match="Tool calculator not found"):
            await tool_manager.call("calculator", {"operation": "add", "a": 1, "b": 2})
