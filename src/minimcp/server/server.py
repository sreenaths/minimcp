import asyncio
import logging
from collections.abc import Callable
from functools import partial

import mcp.shared.version as version
import mcp.types as types
from typing_extensions import Unpack

from minimcp.server.server_core import NotificationOptions, ServerCore

from .exceptions import InvalidMessage
from .managers.tool_manager import ToolDetails, ToolManager

logger = logging.getLogger(__name__)

JSON_RPC_VERSION = "2.0"


class MiniMCP:
    _core: ServerCore
    _timeout: int
    _notification_options: NotificationOptions | None = None

    def __init__(
        self, name: str, version: str | None = None, instructions: str | None = None, timeout: int = 30
    ) -> None:
        self._timeout = timeout

        # TODO: Add support for server-to-client notifications
        self._notification_options = NotificationOptions(
            prompts_changed=False,
            resources_changed=False,
            tools_changed=False,
        )

        # Setup core
        self._core = ServerCore(name=name, version=version, instructions=instructions)
        self._core.request_handlers[types.InitializeRequest] = self._initialize_handler

        # Setup managers
        self.tool_manager = ToolManager(self._core)  # TODO: Make validate_input configurable

    # --- Properties ---
    @property
    def name(self) -> str:
        return self._core.name

    @property
    def instructions(self) -> str | None:
        return self._core.instructions

    @property
    def version(self) -> str | None:
        return self._core.version

    # --- Decorators ---
    def tool(self, **kwargs: Unpack[ToolDetails]) -> Callable[[Callable], types.Tool]:
        return partial(self.tool_manager.add, **kwargs)

    # --- Handlers ---
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
        msg_root = rpc_msg.root

        # --- Handle request ---
        if isinstance(msg_root, types.JSONRPCRequest):
            request_id = msg_root.id

            client_request = types.ClientRequest.model_validate(
                msg_root.model_dump(by_alias=True, mode="json", exclude_none=True)
            )

            logger.info(f"Handling request: {client_request}")
            response = await self._core.handle_request(client_request)
            logger.info(f"Handled request: {client_request}")

            if response is None:
                return None
            elif isinstance(response, types.ErrorData):
                return_msg = types.JSONRPCError(jsonrpc=JSON_RPC_VERSION, id=request_id, error=response)
            else:
                return_msg = types.JSONRPCResponse(
                    jsonrpc=JSON_RPC_VERSION,
                    id=request_id,
                    result=response.model_dump(by_alias=True, mode="json", exclude_none=True),
                )

            logger.info(f"Returning response: {return_msg}")
            return types.JSONRPCMessage(return_msg)

        # --- Handle notification ---
        elif isinstance(msg_root, types.JSONRPCNotification):
            # TODO: Add full support for client notification - This just implements the handler,
            # decorators for usage needs to be added.
            client_notification = types.ClientNotification.model_validate(
                msg_root.model_dump(by_alias=True, mode="json", exclude_none=True)
            )

            logger.info(f"Handling notification: {client_notification}")
            await self._core.handle_notification(client_notification)
            logger.info(f"Handled notification: {client_notification}")
            return None
        else:
            err = InvalidMessage("Invalid message: Message to MCP server must be a request or notification")
            logger.error(err)
            raise err

    async def _initialize_handler(self, req: types.InitializeRequest) -> types.ServerResult:
        client_protocol_version = req.params.protocolVersion
        server_protocol_version = (
            client_protocol_version
            if client_protocol_version in version.SUPPORTED_PROTOCOL_VERSIONS
            else types.LATEST_PROTOCOL_VERSION
        )

        init_options = self._core.create_initialization_options(
            notification_options=self._notification_options,
        )

        init_result = types.InitializeResult(
            protocolVersion=server_protocol_version,
            capabilities=init_options.capabilities,
            serverInfo=types.Implementation(
                name=init_options.server_name,
                version=init_options.server_version,
            ),
            instructions=init_options.instructions,
        )

        return types.ServerResult(init_result)
