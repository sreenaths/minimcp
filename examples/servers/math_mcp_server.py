"""
A simple MCP server for mathematical operations.
"""

from pydantic import Field

from minimcp.server import MiniMCP

mcp = MiniMCP(name="MathServer", version="0.1.0")


@mcp.tool(name="add", description="Add two numbers")
def add(
    a: float = Field(description="The first float number"), b: float = Field(description="The second float number")
) -> float:
    return a + b


@mcp.tool(name="subtract", description="Subtract two numbers")
def subtract(
    a: float = Field(description="The first float number"), b: float = Field(description="The second float number")
) -> float:
    return a - b


@mcp.tool(name="multiply", description="Multiply two numbers")
def multiply(
    a: float = Field(description="The first float number"), b: float = Field(description="The second float number")
) -> float:
    return a * b


@mcp.tool(name="divide", description="Divide two numbers")
def divide(
    a: float = Field(description="The first float number"), b: float = Field(description="The second float number")
) -> float:
    return a / b
