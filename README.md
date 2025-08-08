<div align="center">

<!-- omit in toc -->
# ✨ MiniMCP

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![PyPI version](https://img.shields.io/pypi/v/minimcp.svg)](https://pypi.org/project/minimcp/)
</div>

A **minimal, stateless, and lightweight** MCP server designed for easy integration into any Python application. MiniMCP enforces no transport mechanism or session architecture—instead, it provides a simple asynchronous function to handle JSON-RPC messages, letting you choose the rest. By default, it doesn’t use streams; concurrent messages are handled asynchronously, with concurrency support provided by the transport layer. It is built on the [official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk), enabling standardized context and resource sharing.

### Why MiniMCP?

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) is very powerful, and the official MCP Python SDK provides a low-level implementation of the protocol. [FastMCP](https://github.com/jlowin/fastmcp) makes MCP adoption simpler by offering a high-level, Pythonic interface and its own extended capabilities.

But what if you just need a simple MCP server—local or remote—that responds to requests? What if you don’t want the server to dictate the transport mechanism, or you’d like to use your preferred protocol? What if the server cannot be mounted onto the framework your application is built on? What if you don’t need bidirectional messaging or any of the additional features? — ⭕ _The best part is no part._

MiniMCP provides an asynchronous handle function that accepts a JSON-RPC 2.0 message (as a dict or JSON string) and returns a JSON-RPC 2.0 message (as a dict). This allows you to integrate it into your application in whatever way you choose.

#### Key Features
- 🔗 Easy to embed into existing servers, CLI tools, or background workers
- 🛠 Passing and managing context per request
- ⚡ Asynchronous, stateless message processing (stateless between requests)
- 📝 Easy handler registration for different MCP message types
- 🧩 Separation of concerns: transport layer is completely separate from message handling
- 📦 Minimal dependencies—just the official SDK

#### Non-Features
- 🚫 No built-in session management
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
Minimal code snippet showing basic tool usage.
```python
mcp = MiniMCP(name="ServerName", version="0.1.0")

@mcp.tool(Tool(...))
def tool_handler():
    ...

response = await mcp.handle(client_message)
```

## Examples
MiniMCP dev setup is required for running the examples. Once cloned, run the following in MiniMCP project root for a dev setup.
```bash
uv sync --frozen --all-extras --dev
```

### FastAPI
The example demos embedding MiniMCP server into a FastAPI application.
```bash
uv run uvicorn examples.servers.fastapi:app --reload
```

## License
[Apache License, Version 2.0](./LICENSE)
