"""
MiniMCP math server for integration tests.
"""

import anyio
from pydantic import Field

from minimcp import MiniMCP

# Create a simple math server for testing directly in this file
math_mcp = MiniMCP(
    name="TestMathServer",
    version="0.1.0",
    instructions="A simple MCP server for mathematical operations used in integration tests.",
)


# -- Tools --
@math_mcp.tool()
def add(a: float = Field(description="The first number"), b: float = Field(description="The second number")) -> float:
    """Add two numbers"""
    return a + b


@math_mcp.tool()
def subtract(
    a: float = Field(description="The first number"), b: float = Field(description="The second number")
) -> float:
    """Subtract two numbers"""
    return a - b


@math_mcp.tool()
def multiply(
    a: float = Field(description="The first number"), b: float = Field(description="The second number")
) -> float:
    """Multiply two numbers"""
    return a * b


@math_mcp.tool()
def divide(
    a: float = Field(description="The first number"), b: float = Field(description="The second number")
) -> float:
    """Divide two numbers"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


@math_mcp.tool(description="Add two numbers")
async def add_with_progress(
    a: float = Field(description="The first float number"), b: float = Field(description="The second float number")
) -> float:
    responder = math_mcp.context.get_responder()
    await responder.report_progress(0.1, message="Adding numbers")
    await anyio.sleep(0.1)
    await responder.report_progress(0.4, message="Adding numbers")
    await anyio.sleep(0.1)
    await responder.report_progress(0.7, message="Adding numbers")
    await anyio.sleep(0.1)
    return a + b


# -- Prompts --
@math_mcp.prompt()
def math_help(operation: str = Field(description="The mathematical operation to get help with")) -> str:
    """Get help with mathematical operations"""
    return f"""You are a helpful math assistant.
Provide guidance on how to perform the following mathematical operation: {operation}

Include:
1. Step-by-step instructions
2. Example calculations
3. Common pitfalls to avoid
"""


# -- Resources --
MATH_CONSTANTS = {"pi": 3.14159265359, "e": 2.71828182846, "golden_ratio": 1.61803398875, "sqrt_2": 1.41421356237}


@math_mcp.resource("math://constants")
def get_math_constants() -> dict:
    """Mathematical constants reference"""
    return MATH_CONSTANTS


@math_mcp.resource("math://constants/{constant_name}")
def get_math_constant(constant_name: str) -> float:
    """Get a specific mathematical constant"""
    if constant_name not in MATH_CONSTANTS:
        raise ValueError(f"Unknown constant: {constant_name}")
    return MATH_CONSTANTS[constant_name]
