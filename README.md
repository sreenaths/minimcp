<div align="center">

<!-- omit in toc -->
# ✨ MiniMCP

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![PyPI version](https://img.shields.io/pypi/v/minimcp.svg)](https://pypi.org/project/minimcp/)

A _**minimal, stateless, and lightweight**_ framework for building remote MCP servers.
</div>

MiniMCP is designed with simplicity and flexibility in mind and enforces no transport mechanism or session architecture—instead, it provides a simple asynchronous function to handle JSON-RPC 2.0 messages, letting you choose the rest. By default, it doesn’t use streams; concurrent messages are handled asynchronously, with concurrency support provided by the transport layer. It is built on the [official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk), enabling standardized context and resource sharing.

## Why MiniMCP?

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io) is a powerful, standardized way to provide context and tools to your LLMs. The official MCP Python SDK offers a low-level implementation of the protocol, while [FastMCP](https://github.com/jlowin/fastmcp) simplifies adoption with a high-level, Pythonic interface. _However, both require a transport layer that supports bidirectional communication and include additional complexities for managing message streams and sessions._

What if you just need a simple remote MCP server that responds to requests? What if you don’t want the server to dictate the transport mechanism, or you’d like to use your preferred protocol? What if the server cannot be mounted onto the framework your application is built on? What if you don’t need bidirectional messaging or any of the additional features? — ⭕ _The best part is no part._

MiniMCP provides an asynchronous handle function that accepts a JSON-RPC message (as a dict or JSON string) and returns a JSON-RPC message (as a dict). This allows you to integrate it into your application in whatever way you choose.

### Key Features

- 🔗 Easy to embed into existing servers, CLI tools, or background workers
- 🛠 Passing metadata and managing context per request
- ⚡ Asynchronous, stateless message processing (stateless between requests)
- 📝 Easy handler registration for different MCP message types
- 🧩 Separation of concerns: transport layer is completely separate from message handling
- 📦 Minimal dependencies—just the official SDK

### Non-Features

- 🚫 Session management — _Easily build your own with metadata and context_
- 🚫 Authentication — _Use your existing authentication system_
- 🚫 No server-initiated messaging

## Using MiniMCP

### Installation

```bash
uv add minimcp
```

or

```bash
pip install minimcp
```

### Integration

Minimal code snippet showing basic tool usage using FastAPI.

```python
from fastapi import FastAPI, Request
from minimcp.server import MiniMCP

app = FastAPI()
mcp = MiniMCP(name="MathServer")

@mcp.tool(description="Add two numbers")
def add(a:int, b:int) -> int:
    return a + b

@app.post("/mcp")
async def handle_mcp_request(request: Request):
    msg = await request.json()
    return await mcp.handle(msg)
```

## Examples

The example demonstrates a [Math MCP server](https://github.com/sreenaths/minimcp/blob/main/examples/math_mcp_server/math_mcp.py) with four tools (add, subtract, multiply, and divide) capable of working across different transport mechanisms. To run the examples, you’ll need a MiniMCP development setup. After cloning this repository, execute the following command from the project root to set up the development environment.

```bash
uv sync --frozen --all-extras --dev
```

### FastAPI

[This example](https://github.com/sreenaths/minimcp/blob/main/examples/math_mcp_server/fastapi.py) demos embedding MiniMCP server into a FastAPI application.

```bash
uv run uvicorn examples.math_mcp_server.fastapi:app --reload
```

### Stdio

[This example](https://github.com/sreenaths/minimcp/blob/main/examples/math_mcp_server/stdio.py) demos using MiniMCP with stdio.

It can be run in Claude desktop app with the following configuration.

```json
{
    "mcpServers":
    {
        "math-server":
        {
            "command": "uv",
            "args":
            [
                "--directory",
                "/path/to/minimcp",
                "run",
                "-m",
                "examples.math_mcp_server.stdio"
            ]
        }
    }
}
```

## License

[Apache License, Version 2.0](https://github.com/sreenaths/minimcp/blob/main/LICENSE)
