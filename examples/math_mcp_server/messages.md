# MCP Messages

Sample JSON-RPC 2.0 messages - request and expected response.

## Ping

```json
// Request
{
  "jsonrpc": "2.0",
  "id": "123",
  "method": "ping"
}

// Response
{
    "jsonrpc": "2.0",
    "id": "123",
    "result": {}
}
```

## Initialize

```json
// Request
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

// Response
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

## Tools

### List Tools

```json
// Request
{
    "jsonrpc": "2.0",
    "id": "tools-1",
    "method": "tools/list",
    "params": {}
}

// Response
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

### Tool Calling

```json
// Request
{
    "jsonrpc": "2.0",
    "id": "tool-call-1",
    "method": "tools/call",
    "params": {
        "name": "add",
        "arguments": {
            "a": 5,
            "b": 3
        }
    }
}

// Response
{
    "jsonrpc": "2.0",
    "id": "tool-call-1",
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

### Tool Calling With Progress

```json
// Request
{
    "jsonrpc": "2.0",
    "id": "tool-call-2",
    "method": "tools/call",
    "params": {
        "name": "add",
        "arguments": {
            "a": 5,
            "b": 3
        },
        "_meta": {
            "progressToken": "pg_10"
        }
    }
}

// Responses
{
    "method": "notifications/progress",
    "params": {
        "progressToken": "pg_10",
        "progress": 0.3,
        "message": "Adding numbers"
    },
    "jsonrpc": "2.0"
}

{
    "method": "notifications/progress",
    "params": {
        "progressToken": "pg_10",
        "progress": 0.7,
        "message": "Adding numbers"
    },
    "jsonrpc": "2.0"
}

{
    "jsonrpc": "2.0",
    "id": "tool-call-2",
    "result": {
        "content": [
            {
                "type": "text",
                "text": "8.0"
            }
        ],
        "structuredContent": {
            "result": 8.0
        },
        "isError": false
    }
}
```

## Prompt

### List Prompts

```json
// Request
{
    "jsonrpc": "2.0",
    "id": "prompt-1",
    "method": "prompts/list",
    "params": {}
}

// Responses
{
    "jsonrpc": "2.0",
    "id": "prompt-1",
    "result": {
        "prompts": [
            {
                "name": "problem_solving",
                "description": "General Math Problem Solving Prompt",
                "arguments": [
                    {
                        "name": "problem",
                        "description": "The problem to solve",
                        "required": true
                    }
                ]
            }
        ]
    }
}
```

### Get Prompt

```json
{
    "jsonrpc": "2.0",
    "id": "prompt-get-1",
    "method": "prompts/get",
    "params": {
        "name": "problem_solving",
        "arguments": {
            "problem": "What is the value of x, when 2x + 5 = 17",
        }
    }
}

{
    "jsonrpc": "2.0",
    "id": "prompt-get-1",
    "result": {
        "description": "General Math Problem Solving Prompt",
        "messages": [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": "You are a math problem solver.\nSolve the following problem step by step and provide the final simplified answer.\n\nProblem: What is the value of x, when 2x + 5 = 17\n\nOutput:\n1. Step-by-step reasoning\n2. Final answer in simplest form\n"
                }
            }
        ]
    }
}
```
