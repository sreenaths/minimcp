# MCP Messages

Sample JSON-RPC 2.0 messages - request and expected response.

## Ping

```json
{
  "jsonrpc": "2.0",
  "id": "123",
  "method": "ping"
}

{
    "jsonrpc": "2.0",
    "id": "123",
    "result": {}
}
```

## Initialize

```json
{
    "jsonrpc": "2.0",
    "id": "init-1",
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "roots": {"listChanged": true}
        },
        "clientInfo": {
            "name": "mcp-client",
            "version": "1.0.0"
        }
    }
}

{
    "jsonrpc": "2.0",
    "id": "init-1",
    "result": {
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "experimental": {},
            "tools": {
                "listChanged": false
            }
        },
        "serverInfo": {
            "name": "math-server",
            "version": "1.0.0"
        }
    }
}
```

## List Tools

```json
{
    "jsonrpc": "2.0",
    "id": "tools-1",
    "method": "tools/list",
    "params": {}
}

{
    "jsonrpc": "2.0",
    "id": "tools-1",
    "result": {
        "tools": [
            {
                "name": "add",
                "description": "Add two numbers",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "number"
                        },
                        "b": {
                            "type": "number"
                        }
                    },
                    "required": [
                        "a",
                        "b"
                    ],
                    "additionalProperties": false
                }
            }
        ]
    }
}
```

## Tool Calling

```json
{
    "jsonrpc": "2.0",
    "id": "call-1",
    "method": "tools/call",
    "params": {
        "name": "add",
        "arguments": {
            "a": 5,
            "b": 3
        }
    }
}

{
    "jsonrpc": "2.0",
    "id": "call-1",
    "result": {
        "content": [
            {
                "type": "text",
                "text": "8"
            }
        ],
        "isError": false
    }
}
```
