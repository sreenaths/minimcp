"""
Integration tests for MCP server using fastmcp stdio client.
"""

import os
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio
from fastmcp.client import Client, PythonStdioTransport
from mcp.types import TextContent, TextResourceContents


@pytest_asyncio.fixture
async def mcp_client() -> AsyncGenerator[Client[PythonStdioTransport], None]:
    """Create and manage an MCP client connected to our test server."""
    # Get the path to our test server script
    test_server_path = Path(__file__).parent / "math_server_stdio.py"

    # Create the transport
    transport = PythonStdioTransport(
        script_path=str(test_server_path),
        env={**os.environ, "MCP_SERVER_LOG_FILE": "test_integration.log"},
        cwd=str(Path(__file__).parent.parent.parent),  # Project root
        keep_alive=False,
    )

    # Create the client
    client = Client(transport)

    try:
        # Use the client as an async context manager
        async with client:
            yield client
    except Exception as e:
        # If there's an error, make sure to clean up
        raise e


@pytest.mark.asyncio
async def test_server_initialization(mcp_client: Client[PythonStdioTransport]):
    """Test that the MCP server initializes correctly."""
    # The client should be initialized by the fixture
    assert mcp_client is not None

    # Test that we can get initialization result
    init_result = mcp_client.initialize_result
    assert init_result is not None
    assert init_result.serverInfo.name == "TestMathServer"
    assert init_result.serverInfo.version == "0.1.0"


@pytest.mark.asyncio
async def test_list_tools(mcp_client: Client[PythonStdioTransport]):
    """Test listing available tools."""
    tools = await mcp_client.list_tools()

    # Check that we have the expected tools
    tool_names = [tool.name for tool in tools]
    expected_tools = ["add", "subtract", "multiply", "divide"]

    for expected_tool in expected_tools:
        assert expected_tool in tool_names, f"Expected tool '{expected_tool}' not found in {tool_names}"

    # Check tool details for the add function
    add_tool = next(tool for tool in tools if tool.name == "add")
    assert add_tool.description == "Add two numbers"
    assert "a" in add_tool.inputSchema["properties"]
    assert "b" in add_tool.inputSchema["properties"]


@pytest.mark.asyncio
async def test_call_add_tool(mcp_client: Client[PythonStdioTransport]):
    """Test calling the add tool."""
    result = await mcp_client.call_tool("add", {"a": 5.0, "b": 3.0})

    assert result.is_error is False
    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert float(result.content[0].text) == 8.0


@pytest.mark.asyncio
async def test_call_subtract_tool(mcp_client: Client[PythonStdioTransport]):
    """Test calling the subtract tool."""
    result = await mcp_client.call_tool("subtract", {"a": 10.0, "b": 4.0})

    assert result.is_error is False
    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert float(result.content[0].text) == 6.0


@pytest.mark.asyncio
async def test_call_multiply_tool(mcp_client: Client[PythonStdioTransport]):
    """Test calling the multiply tool."""
    result = await mcp_client.call_tool("multiply", {"a": 6.0, "b": 7.0})

    assert result.is_error is False
    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert float(result.content[0].text) == 42.0


@pytest.mark.asyncio
async def test_call_divide_tool(mcp_client: Client[PythonStdioTransport]):
    """Test calling the divide tool."""
    result = await mcp_client.call_tool("divide", {"a": 15.0, "b": 3.0})

    assert result.is_error is False
    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert float(result.content[0].text) == 5.0


@pytest.mark.asyncio
async def test_call_divide_by_zero(mcp_client: Client[PythonStdioTransport]):
    """Test calling the divide tool with zero divisor."""
    from fastmcp.exceptions import ToolError

    with pytest.raises(ToolError) as exc_info:
        await mcp_client.call_tool("divide", {"a": 10.0, "b": 0.0})

    assert "Cannot divide by zero" in str(exc_info.value)


@pytest.mark.asyncio
async def test_list_prompts(mcp_client: Client[PythonStdioTransport]):
    """Test listing available prompts."""
    prompts = await mcp_client.list_prompts()

    # Check that we have the expected prompt
    prompt_names = [prompt.name for prompt in prompts]
    assert "math_help" in prompt_names

    # Check prompt details
    math_help_prompt = next(prompt for prompt in prompts if prompt.name == "math_help")
    assert math_help_prompt.description == "Get help with mathematical operations"


@pytest.mark.asyncio
async def test_get_prompt(mcp_client: Client[PythonStdioTransport]):
    """Test getting a prompt."""
    result = await mcp_client.get_prompt("math_help", {"operation": "addition"})

    assert len(result.messages) == 1
    assert result.messages[0].role == "user"

    # Ensure content is TextContent before accessing .text
    content = result.messages[0].content
    assert isinstance(content, TextContent)
    assert "addition" in content.text
    assert "math assistant" in content.text.lower()


@pytest.mark.asyncio
async def test_list_resources(mcp_client: Client[PythonStdioTransport]):
    """Test listing available resources."""
    resources = await mcp_client.list_resources()

    # Check that we have the expected resources
    resource_uris = [str(resource.uri) for resource in resources]
    expected_resources = ["math://constants"]

    for expected_resource in expected_resources:
        assert expected_resource in resource_uris, (
            f"Expected resource '{expected_resource}' not found in {resource_uris}"
        )


@pytest.mark.asyncio
async def test_read_resource(mcp_client: Client[PythonStdioTransport]):
    """Test reading a resource."""
    result = await mcp_client.read_resource("math://constants")

    assert len(result) == 1
    assert str(result[0].uri) == "math://constants"

    # The content should contain our math constants
    # Ensure content is TextResourceContents before accessing .text
    assert isinstance(result[0], TextResourceContents)
    content_text = result[0].text
    assert "pi" in content_text
    assert "3.14159" in content_text


@pytest.mark.asyncio
async def test_read_resource_template(mcp_client: Client[PythonStdioTransport]):
    """Test reading a resource template."""
    result = await mcp_client.read_resource("math://constants/pi")

    assert len(result) == 1
    assert str(result[0].uri) == "math://constants/pi"

    # The content should be just the pi value
    # Ensure content is TextResourceContents before accessing .text
    assert isinstance(result[0], TextResourceContents)
    content_text = result[0].text
    assert "3.14159" in content_text


@pytest.mark.asyncio
async def test_invalid_tool_call(mcp_client: Client[PythonStdioTransport]):
    """Test calling a non-existent tool."""
    from fastmcp.exceptions import ToolError

    with pytest.raises(ToolError) as exc_info:
        await mcp_client.call_tool("nonexistent_tool", {})

    assert "nonexistent_tool not found" in str(exc_info.value)


@pytest.mark.asyncio
async def test_invalid_resource_read(mcp_client: Client[PythonStdioTransport]):
    """Test reading a non-existent resource."""
    with pytest.raises(Exception):  # fastmcp should raise an exception for invalid resources
        await mcp_client.read_resource("math://nonexistent")


@pytest.mark.asyncio
async def test_tool_call_with_invalid_parameters(mcp_client: Client[PythonStdioTransport]):
    """Test calling a tool with invalid parameters."""
    from fastmcp.exceptions import ToolError

    # Try to call add with missing parameter
    with pytest.raises(ToolError) as exc_info:
        await mcp_client.call_tool("add", {"a": 5.0})  # Missing 'b' parameter

    assert "Field required" in str(exc_info.value)
