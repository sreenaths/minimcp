# MiniMCP

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![PyPI version](https://img.shields.io/pypi/v/minimcp.svg)](https://pypi.org/project/minimcp/)

A minimal, stateless, and lightweight MCP server designed for easy integration into any Python application. MiniMCP enforces no transport mechanism — instead, it provides a function to handle JSON-RPC messages and lets you define the rest. It is built on the [official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk), enabling standardized, remote context and resource sharing.


### Why MiniMCP?

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) is powerful, and the official MCP Python SDK provides a low-level implementation of the protocol. FastMCP makes MCP adoption simpler by offering a high-level, Pythonic interface and comes with its own extended capabilities.

But what if you don’t want the server to dictate the transport mechanism, or you’d like to use your preferred protocol? What if FastMCP cannot be mounted onto the framework your application is built on? What if you don’t need all the additional features?

MiniMCP provides a simple async handle function that accepts JSON and returns JSON, letting you integrate it into your application however you choose.

## License
[Apache 2.0](./LICENSE)
