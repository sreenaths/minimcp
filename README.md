# MiniMCP

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![PyPI version](https://img.shields.io/pypi/v/minimcp.svg)](https://pypi.org/project/minimcp/)

A minimal, stateless, and lightweight MCP server designed for easy integration into any Python application. MiniMCP enforces no transport mechanism — instead, it provides a simple async handle function that accepts JSON and returns JSON, letting you drop it into your application however you like. MiniMCP is built on the [official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk), enabling standardized, remote context and resource sharing.

### Why MiniMCP?

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) protocol is powerful and the official MCP Python SDK provides a low-level implimentation of the same. FastMCP made MCP adoption simpler by providing a high-level, Pythonic interface.

## License
[Apache 2.0](./LICENSE)
