<div align="center">

<!-- omit in toc -->
# ✨ MiniMCP

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![PyPI version](https://img.shields.io/pypi/v/minimcp.svg)](https://pypi.org/project/minimcp/)

A **minimal, stateless, and lightweight** framework for building remote MCP servers.
</div>

_MiniMCP is designed with simplicity and flexibility in mind and provides a simple asynchronous function to handle MCP messages - You just have to call the function with the request message and it returns the response message, its as simple as that_ ⭐ _Bidirectional communication is not mandatory in MiniMCP but is supported - So you can use plain HTTP for communication_ ⭐ _It is primarily build for remote MCP servers - But supports local MCP servers too_ ⭐ _MiniMCP does provide a set of transport mechanisms (stdio, HTTP with Starlette, Streamable HTTP with Starlette) that wraps around the handler - The user is free to use them or are encouraged to build their own_ ⭐ _It is built on the [official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk), enabling standardized context and resource sharing._

## What is MCP?

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io) is a powerful, standardized way for AI applications to connect with external data sources and tools. It uses a client-server architecture where the client and server talk to each other using standardized messages defined by MCP, and uses JSON-RPC 2.0 format. Handler functions can be registered to process the messages.The key benefit is that once a server supports MCP, any MCP-compatible AI client can connect to it without needing custom integration code. The official MCP Python SDK offers a low-level implementation of the protocol, while [FastMCP](https://github.com/jlowin/fastmcp) provides a high-level, Pythonic interface.

## Why MiniMCP?

MiniMCP rethinks MCP server from the ground up, keeping the core functionality lightweight and independent of the transport layer, bidirectional communication, session management, and auth mechanisms. Additionally, instead of a stream-based interface, MiniMCP uses an asynchronous handle function that accepts a string JSON-RPC 2.0 message as argument and returns a JSON-RPC message string as response.

- **Stateless:** Scalability, simplicity and reliability are crucial while building remote MCP servers. Being stateless at the core, MiniMCP is able to ensure these and provides a robust and easy to maintain server.
- **Bidirectional is optional:** It seems like a large number of use-cases can be served by a simple request–response channel without needing bidirectional communication. MiniMCP was built with this in mind and provides a simple HTTP transport while adhering to the specification.
- **Embeddable:** You might already have an application built using FastAPI (or some other framework) and you just want to host a remote MCP server under a specific endpoint in it. Using MiniMCP you can easily embed an MCP server under an end point or multiple servers under multiple end points - _Unlike mounting you can use your existing dependency injection system._
- **Scope and Context:** MiniMCP facilitates a type-checked _scope object_ that can be passed with each message. This enabled you to pass extra details including auth, user, session, database etc along with the message. Inside the handled, scope is available as part of the context - _You are free to use your favourite user or session management mechanisms!_
- **Security:** MiniMCP encourages you to use your existing battle-tested security mechanism instead of enforcing one - _A MiniMCP server built with FastAPI can be as secure as any FastAPI application! In a way, an MCP server is a standardized way to access a set of APIs!_
- **Stream on Demand:** MiniMCP comes with a smart Streamable HTTP implementation. It keeps track of the usage pattern and sends back a normal JSON HTTP response if the handler just returns a response. In other words, the event stream is used only if a notification needs to be sent from the server to the client.
- **Separation of Concerns:** The transport layer is completely decoupled from message handling. This allows you to adapt MiniMCP to different environments and protocols without rewriting your core business logic.
- **Minimal Dependencies:** MiniMCP keeps its footprint small, relying only on the official MCP SDK. This makes it lightweight, easy to maintain, and less prone to dependency conflicts.

### Supported Features

The following are currently supported by MiniMCP.

- 🧩 Server primitives - Tools, Prompts and Resources
- 🔗 Transports - stdio, HTTP, Streamable HTTP
- 🔄 Server to client messages - Progress notification
- 🛠 Typed scope and handler context
- ⚡ Asynchronous - stateless message processing
- 📝 Easy handler registration for different MCP message types
- ⏱️ Enforces idle time and concurrency limits
- 📦 Web frameworks - In-built support for Starlette/FastAPI

### Future Support

If need arises following features will be supported in the future.

- ⚠️ Server-initiated messaging
- ⚠️ Client primitives - Sampling, Elicitation, Logging
- ⚠️ Pagination
- ⚠️ Resumable Streamable HTTP with GET method support
- ⚠️ MCP Client (_As demonstrated in these [integration tests](https://github.com/sreenaths/minimcp/tree/main/tests/integration), MiniMCP can be used with existing MCP clients and it doesn't seem like we would need a custom client for the time being_)

### Non-Features

The following features are very unlikely to be built into MiniMCP in the foreseeable future.

- 🚫 Session management
- 🚫 Authentication

## Using MiniMCP

The following gives you a brief idea on how to use MiniMCP. Checkout the [examples](https://github.com/sreenaths/minimcp/tree/main/examples) for more.

### Installation

```bash
pip install minimcp
```

### Basic MiniMCP

A basic example demonstrating primitive registration and basic message processing using handle function.

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
    return f"""You are a math problem solver.
......
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

A minimal example demonstrating how to expose an MCP tool over HTTP with FastAPI.

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

The official MCP specification currently defines two standard transport mechanisms - [stdio](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#stdio) and [Streamable HTTP](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#streamable-http). It also provides flexibility for different implementations and also permits custom transports. MiniMCP makes use of this flexibility to provide a third [HTTP transport](https://github.com/sreenaths/minimcp/blob/main/docs/transport-specification-compliance.md#2-http-transport).

HTTP is a subset of Streamable HTTP and doesn't provide bidirectional communication. But on the hind side, just like in the above integration example, it can be technically added as a restful API end point in any Python application for developing remote MCP servers. It also works with Streamable HTTP clients.

As explained earlier, MiniMCP comes with a smart Streamable HTTP implementation. It keeps track of the usage pattern and sends back a normal JSON HTTP response if the handler just returns a response. In other words, the event stream is used only if a notification needs to be sent from the server to the client. For simplicity and statelessness, it uses polling to keep the stream open, and fully resumable Streamable HTTP can be supported in the future if needed.

Please check the [specification compliance](https://github.com/sreenaths/minimcp/blob/main/docs/transport-specification-compliance.md) for more details on the transports supported by MiniMCP.

## API Reference

This section gives an ide a about the key classes, functions in them, and their respective arguments.

### MiniMCP

As the name would suggest, MiniMCP is the central entity to cerate a server. It expects the server name as a mandatory argument, and other arguments are optional. It even takes the type of the scope object that can be passed through the system. It provides tool, prompt and resource manager instance that can be used to register handlers, and a context manager instance that can be used from inside the handlers. The handle function that process message accepts a string message, and two optional params - send function and scope object.

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

Limiter in MiniMCP enforces the maximum number of handlers that can run at an instance, and the time each of them can be idle for. By default idle_timeout is 30 seconds, and max_concurrency is 100. Both are passed as an optional argument of MiniMCP.

### Primitive Managers/Decorators

MiniMCP supports three server primitives through three primitive managers as explained below. They are implemented as callable classes that return decorators to register handler functions. On call, they also accept primitive definitions. If not specified the definition is inferred from the handled function. All the 3 primitive managers provide additional functions to `add`, `list`, `remove` and use the handlers programmatically.

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

Context manager keep track of the handler context and you can call `mcp_instance.context.get()` from inside a handler to get its respective context. Calling from outside a handle would throw a `No Context` error.

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

The [examples](https://github.com/sreenaths/minimcp/blob/main/examples) demonstrates a [Math MCP server](https://github.com/sreenaths/minimcp/blob/main/examples/math_mcp_server/math_mcp.py) with four tools (add, subtract, multiply, and divide) working with different transport mechanisms and frameworks. To run the examples, you’ll need a MiniMCP development setup. After cloning this repository, execute the following command from the project root to set up the development environment.

```bash
uv sync --frozen --all-extras --dev
```

The following table shows different examples and the command to run them.

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
