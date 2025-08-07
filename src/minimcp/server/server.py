from .server_core import ServerCore

import mcp.types as types
import logging
import asyncio

from .exceptions import InvalidMessage

logger = logging.getLogger(__name__)


class MiniMCP:
    _core: ServerCore
    _timeout: int

    def __init__(self,
        name: str, version: str | None = None,
        instructions: str | None = None,
        timeout: int = 30
    ) -> None:
        self._core = ServerCore(name=name, version=version, instructions=instructions)
        self._timeout = timeout

    async def handle(self, message: str | dict) -> dict | None:

        if isinstance(message, str):
            rpc_msg = types.JSONRPCMessage.model_validate_json(message)
        elif isinstance(message, dict):
            rpc_msg = types.JSONRPCMessage.model_validate(message)
        else:
            err = InvalidMessage("Invalid message type: Must be a string or dict")
            logger.error(err)
            raise err

        try:
            response = await asyncio.wait_for(self._handle_rpc_msg(rpc_msg), timeout=self._timeout)
        except asyncio.TimeoutError:
            logger.error(f"Handler timed out: {rpc_msg}")
            raise

        if response is None:
            logger.info(f"No response returned for message: {rpc_msg}")
            return None

        return response.model_dump(by_alias=True, mode="json", exclude_none=True)

    async def _handle_rpc_msg(self, rpc_msg: types.JSONRPCMessage) -> types.JSONRPCMessage | None:

        # --- Handle request ---
        if isinstance(rpc_msg, types.JSONRPCRequest):
            request_id = rpc_msg.id

            client_request = types.ClientRequest.model_validate(
                rpc_msg.model_dump(by_alias=True, mode="json", exclude_none=True)
            )

            logger.info(f"Handling request: {client_request}")
            response = await self._core.handle_request(client_request)
            logger.info(f"Handled request: {client_request}")

            if response is None:
                return None
            elif isinstance(response, types.ErrorData):
                return_msg = types.JSONRPCError(jsonrpc="2.0", id=request_id, error=response)
            else:
                return_msg = types.JSONRPCResponse(
                    jsonrpc="2.0",
                    id=request_id,
                    result=response.model_dump(by_alias=True, mode="json", exclude_none=True),
                )

            logger.info(f"Returning response: {return_msg}")
            return types.JSONRPCMessage(return_msg)

        # --- Handle notification ---
        elif isinstance(rpc_msg, types.JSONRPCNotification):
            client_notification = types.ClientNotification.model_validate(
                rpc_msg.model_dump(by_alias=True, mode="json", exclude_none=True)
            )

            logger.info(f"Handling notification: {client_notification}")
            await self._core.handle_notification(client_notification)
            logger.info(f"Handled notification: {client_notification}")
            return None
        else:
            err = InvalidMessage("Invalid message: Message to MCP server must be a request or notification")
            logger.error(err)
            raise err
