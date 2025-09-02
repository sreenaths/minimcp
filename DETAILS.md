<div align="center">

<!-- omit in toc -->
# ✨ MiniMCP (Generated)

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

A minimal example demonstrating how to expose an MCP tool over HTTP with FastAPI.

```python
from fastapi import FastAPI, Request
from minimcp import MiniMCP

from minimcp.server.transports.http import starlette_http_transport

# This can be an existing FastAPI/Starlette app (with authentication, middleware, etc.)
app = FastAPI()

# Create an MCP instance
mcp = MiniMCP(name="MathServer")

# Register a simple tool
@mcp.tool(description="Add two numbers")
def add(a:int, b:int) -> int:
    return a + b

# Define the MCP endpoint
@app.post("/mcp")
async def handle_mcp_request(request: Request):
    return await starlette_http_transport(request, mcp.handle)
```

## Transports

The official MCP specification currently defines two standard transport mechanisms for client-server communication - [stdio](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#stdio) and [Streamable HTTP](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#streamable-http). It allows some flexibility in implementation and also supports custom transports. However, implementers must ensure the following:

- All messages MUST use JSON-RPC 2.0 format and be UTF-8 encoded.
- Lifecycle requirements during both initialization and operation phases are met.
- Each message represents an individual request, notification, or response.

Based on the official specification, MiniMCP implements four transport mechanisms, detailed below:

### 1. Stdio

Consistent with the standard, stdio enables bidirectional communication and is commonly employed for developing local MCP servers.

- The server reads messages from its standard input (stdin) and sends messages to its standard output (stdout).
- Messages are delimited by newlines.
- Only valid MCP messages should be written into stdin and stdout.

### 2. HTTP

HTTP is a subset of Streamable HTTP and doesn't provide bidirectional communication. But on the hind side, just like in the above integration example, it can be technically added as a restful API end point in any Python application for developing remote MCP servers.

- Every message sent from the client MUST be a new HTTP POST request to the MCP endpoint.
- The body of the POST request MUST be a single JSON-RPC request or notification.
- If the input is a request - The server MUST return Content-Type: application/json, to return one response JSON object.
- If the input is a notification - If the server accepts, the server MUST return HTTP status code 202 Accepted with no body. If the server cannot accept, it MUST return an HTTP error status code (e.g., 400 Bad Request). The HTTP response body MAY comprise a JSON-RPC error response that has no id.
- Multiple POST requests must be served concurrently by the server.

### 3. Streamable HTTP

All of the above + the following.

- The SSE stream SHOULD eventually include JSON-RPC response for the JSON-RPC request sent in the POST body.

### 4. Websocket

## Examples

The examples demonstrates a [Math MCP server](https://github.com/sreenaths/minimcp/blob/main/examples/math_mcp_server/math_mcp.py) with four tools (add, subtract, multiply, and divide) working with different transport mechanisms and frameworks. To run the examples, you’ll need a MiniMCP development setup. After cloning this repository, execute the following command from the project root to set up the development environment.

```bash
uv sync --frozen --all-extras --dev
```

Following table shows different examples and the command to run them.

| # | Transport | Command |
|---|---|---|
| 1 | Stdio | `uv run -m examples.math_mcp_server.stdio` |
| 2 | HTTP with FastAPI | `uv run uvicorn examples.math_mcp_server.fastapi_http:app --reload` |

Claude desktop can be configured to run the stdio example as follows.

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
