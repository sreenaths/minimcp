# MiniMCP

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![PyPI version](https://img.shields.io/pypi/v/minimcp.svg)](https://pypi.org/project/minimcp/)

A minimal, stateless, and lightweight MCP server designed for easy integration into any Python application. MiniMCP enforces no transport mechanism or session architecture—instead, it provides a simple asynchronous function to handle JSON-RPC messages, letting you choose the rest. It doesn’t use streams; instead, concurrent messages are handled primarily through its asynchronous nature. It is built on the [official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk), enabling standardized context and resource sharing.

### Why MiniMCP?

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) is very powerful, and the official MCP Python SDK provides a low-level implementation of the protocol. FastMCP makes MCP adoption simpler by offering a high-level, Pythonic interface and its own extended capabilities.

_But what if you just need a simple MCP server (local or remote) that responds to your requests? What if you don’t want the server to dictate the transport mechanism, or you’d like to use your preferred protocol? What if the server cannot be mounted onto the framework your application is built on? What if you don’t need bidirectional messaging or any of the additional features?_

_— The best part is no part._ ✨

MiniMCP provides an asynchronous handle function that accepts a JSON-RPC message (as a dict or JSON) and returns a JSON-RPC message (as a dict). This allows you to integrate it into your application in whatever way you choose.

#### What does MiniMCP provide?
- Easy handler registration for different MCP message types
- Passing and managing request context between handlers
- Asynchronous, stateless message processing
- Separation of concerns: transport layer is completely separate from message handling
- Minimal dependencies—just the official SDK

## Using MiniMCP

### Installation

```
pip install minimcp
```

## License
[Apache 2.0](./LICENSE)
