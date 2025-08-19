from typing import Any

from mcp.types import ErrorData, JSONRPCError, JSONRPCMessage, JSONRPCResponse, ServerResult

from minimcp.server.utils import to_dict

JSON_RPC_VERSION = "2.0"


def build_response_message(message_id: str | int, response: ServerResult) -> JSONRPCMessage:
    return JSONRPCMessage(JSONRPCResponse(jsonrpc=JSON_RPC_VERSION, id=message_id, result=to_dict(response)))


def build_error_message(
    error_code: int, rpc_message: dict[str, Any] | None, error: BaseException, error_data: ErrorData | None = None
) -> JSONRPCMessage:
    message_id = rpc_message.get("id", "") if isinstance(rpc_message, dict) else ""
    error_message = f"{error.__class__.__name__}: {error}"
    error_data = error_data or ErrorData(code=error_code, message=error_message, data=None)

    return JSONRPCMessage(JSONRPCError(jsonrpc=JSON_RPC_VERSION, id=message_id, error=error_data))
