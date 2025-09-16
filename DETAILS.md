<div align="center">

<!-- omit in toc -->
# ✨ MiniMCP

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![PyPI version](https://img.shields.io/pypi/v/minimcp.svg)](https://pypi.org/project/minimcp/)

A **minimal, stateless, and lightweight** framework for building remote MCP servers.
</div>

_MiniMCP is designed with simplicity in mind. It provides a single asynchronous function to handle MCP messages—just pass in the request message, and you’ll get the response back_ ⭐ _While MiniMCP supports bidirectional messaging, it’s not a mandatory requirement—So you can use plain HTTP for communication_ ⭐ _MiniMCP is primarily built for remote MCP servers but works just as well for local servers_ ⭐ _MiniMCP ships with built-in transport mechanisms (stdio, HTTP via Starlette, and Streamable HTTP via Starlette). These wrap your handler, but you’re free to use them directly or implement your own_ ⭐ _MiniMCP is built on the [official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk), ensuring standardized context and resource sharing._

## What is MCP?

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io) is a powerful, standardized way for AI applications to connect with external data sources and tools. It follows a client–server architecture, where communication happens through well-defined MCP messages in the JSON-RPC 2.0 format. The key advantage of MCP is interoperability: once a server supports MCP, any MCP-compatible AI client can connect to it without custom integration code. The official MCP Python SDK provides a low-level implementation of the protocol, while [FastMCP](https://github.com/jlowin/fastmcp) offers a higher-level, Pythonic interface.

## Why MiniMCP?

MiniMCP rethinks the MCP server from the ground up, keeping the core functionality lightweight and independent of the transport layer, bidirectional communication, session management, and auth mechanisms. Additionally, instead of a stream-based interface, MiniMCP exposes a simple asynchronous handle function that takes a JSON-RPC 2.0 message string as input and returns a JSON-RPC 2.0 message string as output.

- **Stateless:** Scalability, simplicity, and reliability are crucial for remote MCP servers. MiniMCP is stateless at its core, making it robust, easy to scale, and straightforward to maintain.
- **Bidirectional is optional:** Many use cases work perfectly with a simple request–response channel without needing bidirectional communication. MiniMCP was built with this in mind and provides a simple HTTP transport while adhering to the specification.
- **Embeddable:** Already have an application built with FastAPI (or another framework)? You can embed a MiniMCP server under a single endpoint—or multiple servers under multiple endpoints—_Unlike with mounting, you can use your existing dependency injection system._
- **Scope and Context:** MiniMCP provides a type-checked scope object that travels with each message. This allows you to pass extra details such as authentication, user info, session data, or database handles. Inside the handler, the scope is available in the context—_so you’re free to use your preferred session or user management mechanisms._
- **Security:** MiniMCP encourages you to use your existing battle-tested security mechanism instead of enforcing one - _A MiniMCP server built with FastAPI can be as secure as any FastAPI application!_
- **Stream on Demand:** MiniMCP comes with a smart Streamable HTTP implementation. If the handler just returns a response, the server replies with a normal JSON HTTP response. An event stream is only opened when the server actually needs to push notifications to the client.
- **Separation of Concerns:** The transport layer is fully decoupled from message handling. This makes it easy to adapt MiniMCP to different environments and protocols without rewriting your core business logic.
- **Minimal Dependencies:** MiniMCP keeps its footprint small, depending only on the official MCP SDK. This makes it lightweight, easy to maintain, and less prone to dependency conflicts.

### Currently Supported Features

The following features are already available in MiniMCP.

- 🧩 Server primitives - Tools, Prompts and Resources
- 🔗 Transports - stdio, HTTP, Streamable HTTP
- 🔄 Server to client messages - Progress notification
- 🛠 Typed scope and handler context
- ⚡ Asynchronous - stateless message processing
- 📝 Easy handler registration for different MCP message types
- ⏱️ Enforces idle time and concurrency limits
- 📦 Web frameworks - In-built support for Starlette/FastAPI

### Planned (if needed)

These features may be added in the future if the need arises.

- ⚠️ Server-initiated messaging
- ⚠️ Client primitives - Sampling, Elicitation, Logging
- ⚠️ Pagination
- ⚠️ Resumable Streamable HTTP with GET method support
- ⚠️ MCP Client (_As shown in the [integration tests](https://github.com/sreenaths/minimcp/tree/main/tests/integration), MiniMCP works seamlessly with existing MCP clients, and there’s currently no need for a custom client_)

### Unlikely

These features are not expected to be built into MiniMCP in the foreseeable future.

- 🚫 Session management
- 🚫 Authentication

## Using MiniMCP

The snippets below provide a quick overview of how to use MiniMCP. Checkout the [examples](https://github.com/sreenaths/minimcp/tree/main/examples) for more.

### Installation

```bash
pip install minimcp
```

### Basic MiniMCP

The following example demonstrates simple registration and basic message processing using the handle function.

```python
mcp = MiniMCP(name="MathServer")

# Tool
@mcp.tool()
def add(a:int, b:int) -> int:
    "Add two numbers"
    return a + b

# Prompt
@math_mcp.prompt()
def problem_solving(problem_description: str) -> str:
    "Prompt to systematically solve math problems."
    return f"""You are a math problem solver. Solve the following problem step by step.
Problem: {problem_description}
"""

# Resource
@mcp.resource("math://constants/pi")
def pi_value() -> float:
    """Value of π (pi) to be used"""
    return 3.14

request_msg = '{"jsonrpc": "2.0", "id": "1", "method": "ping"}'
response_msp = await mcp.handle(request_msg, scope={...})
# response_msp = '{"jsonrpc": "2.0", "id": "1", "result": {}}'
```

### FastAPI Integration

This minimal example shows how to expose an MCP tool over HTTP using FastAPI.

```python
from fastapi import FastAPI, Request
from minimcp import MiniMCP, starlette

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
    return await starlette.http_transport(math_mcp.handle, request)
```

## Transports

The official MCP specification currently defines two standard transport mechanisms: [stdio](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#stdio) and [Streamable HTTP](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#streamable-http). It also provides flexibility in implementations and also permits custom transports. MiniMCP uses this flexibility to introduce a third option: [HTTP transport](https://github.com/sreenaths/minimcp/blob/main/docs/transport-specification-compliance.md#2-http-transport).

HTTP is a subset of Streamable HTTP and doesn't support bidirectional communication. However, as shown in the integration example, it can be added as a RESTful API endpoint in any Python application to host remote MCP servers. Importantly, it remains compatible with Streamable HTTP MCP clients.

MiniMCP also provides a smart Streamable HTTP implementation. It adapts to usage patterns: if the handler simply returns a response, the server replies with a normal JSON HTTP response. An event stream is opened only when the server needs to push notifications to the client. To keep things simple and stateless, this is currently implemented using polling to keep the stream alive, with the option to support fully resumable Streamable HTTP in the future.

For more details on supported transports, please check the [specification compliance](https://github.com/sreenaths/minimcp/blob/main/docs/transport-specification-compliance.md) document.

You can use the transports as shown below, or wrap `math_mcp.handle` in a custom function to pass a scope or manage lifecycle:

```python
# Stdio
anyio.run(stdio_transport, math_mcp.handle)

# HTTP
await starlette.http_transport(math_mcp.handle, request)

# Streamable HTTP
await starlette.streamable_http_transport(math_mcp.handle, request)
```

## API Reference

This section provides an overview of the key classes, their functions, and the arguments they accept.

### MiniMCP

As the name suggests, MiniMCP is the core class for creating a server. It requires a server name as its only mandatory argument; all other arguments are optional. You can also specify the type of the scope object, which is passed through the system.

MiniMCP provides:

- Tool, Prompt, and Resource managers — used to register handlers.
- A Context manager — accessible inside handlers.

The handle function processes incoming messages. It accepts a JSON-RPC 2.0 message string and two optional parameters: a send function and a scope object.

MiniMCP controls how many handlers can run at the same time and how long each handler can remain idle. By default, idle_timeout is set to 30 seconds and max_concurrency to 100.

```python
# Instantiation
mcp = MiniMCP[ScopeT](name, [version, instructions, idle_timeout, max_concurrency, raise_exceptions])

# Managers
mcp.tool
mcp.prompt
mcp.resource
mcp.context

# Message handling
response = await mcp.handle(message, [send, scope])
```

### Primitive Managers/Decorators

MiniMCP supports three server primitives, each managed by its own primitive manager. These managers are implemented as callable classes that can be used as decorators for registering handler functions.

When called, a manager accepts a primitive definition. If none is provided, the definition is automatically inferred from the handler function.

In addition to decorator usage, all three primitive managers also expose methods to add, list, remove, and invoke handlers programmatically.

#### Tool Manager

```python
# A a decorator
@mcp.tool([name, title, description, annotations, meta])
def handler_fun(...):...

# Methods for programmatic access
mcp.tool.add(func, [name, title, description, annotations, meta])
mcp.tool.remove(name)
mcp.tool.list()
mcp.tool.call(name, args)
```

#### Prompt Manager

```python
# A a decorator
@mcp.prompt([name, title, description, meta])
def handler_fun(...):...

# Methods for programmatic access
mcp.prompt.add(func, [name, title, description, meta])
mcp.prompt.remove(name)
mcp.prompt.list()
mcp.prompt.get(name, args)
```

#### Resource Manager

```python
# A a decorator
@mcp.resource(url, [name, title, description, mime_type, annotations, meta])
def handler_fun(...):...

# Methods for programmatic access
mcp.resource.add(func, url, [name, title, description, annotations, meta])
mcp.resource.remove(name)
mcp.resource.list()
mcp.resource.list_templates()
mcp.resource.read(uri)
mcp.resource.read_by_name(name, args)
```

### Context Manager

The context manager keeps track of the current handler context. From inside a handler, you can call `mcp_instance.context.get()` to access the associated context. If called outside of a handler, it will raise a NoContext error.

```python
# Context structure
Context(Generic[ScopeT]):
    message: JSONRPCMessage      # Parsed request message
    time_limiter: TimeLimiter    # To reset handler idle timeout
    scope: ScopeT | None         # Scope object passed while calling handle()
    responder: Responder | None  # To send notifications back to the client

# Accessing context
mcp.context.get() -> Context[ScopeT]

# Following syntactic sugar is also provided for use without null check
mcp.context.get_scope() -> ScopeT
mcp.context.get_responder() -> Responder
```

## Examples

The [examples](https://github.com/sreenaths/minimcp/blob/main/examples) include a [Math MCP server](https://github.com/sreenaths/minimcp/blob/main/examples/math_mcp_server/math_mcp.py) with four tools—add, subtract, multiply, and divide—demonstrating how MiniMCP works with different transport mechanisms and frameworks.

To run the examples, you’ll need a MiniMCP development setup. After cloning this repository, run the following command from the project root to set up the environment:

```bash
uv sync --frozen --all-extras --dev
```

The table below lists the available examples along with the commands to run them.

| # | Transport | Command |
|---|---|---|
| 1 | Stdio | `uv run -m examples.math_mcp_server.stdio` |
| 2 | HTTP with FastAPI | `uv run uvicorn examples.math_mcp_server.fastapi_http:app --reload` |
| 3 | Streamable HTTP with FastAPI | `uv run uvicorn examples.math_mcp_server.fastapi_streamable_http:app --reload` |

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
