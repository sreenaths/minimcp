import asyncio
import logging
from collections.abc import Callable
from functools import partial
from typing import Any

import mcp.shared.version as version
import mcp.types as types
from pydantic import ValidationError
from typing_extensions import Unpack

from minimcp.server.server_core import NotificationOptions, ServerCore
from minimcp.server.utils import to_dict

from .exceptions import ErrorWithData, UnsupportedRPCMessageType
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
    async def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        try:
            rpc_msg = types.JSONRPCMessage.model_validate(message)
            response = await asyncio.wait_for(self._handle_rpc_msg(rpc_msg), timeout=self._timeout)

        # --- Centralized error handling - All expected exceptions must be handled here ---
        except ValidationError as e:
            logger.error("Invalid Message: %s", e)
            response = self._build_error_msg(types.INVALID_REQUEST, message, e)
        except UnsupportedRPCMessageType as e:
            logger.error("Unsupported Message Type: %s", e)
            response = self._build_error_msg(types.INVALID_REQUEST, message, e)
        except ErrorWithData as e:
            logger.error("Error While Handling Message: %s", e)
            response = self._build_error_msg(types.INTERNAL_ERROR, message, e, e.data)
        except asyncio.TimeoutError as e:
            logger.error("Message Handler Timed Out: %s", message)
            response = self._build_error_msg(types.INTERNAL_ERROR, message, e)

        if response is None:
            logger.info(f"No response returned for message: {message}")
            return None

        return to_dict(response)

    def _build_error_msg(
        self, error_code: int, message: dict, error: Exception, error_data: types.ErrorData | None = None
    ) -> types.JSONRPCMessage:
        message_id = message.get("id", "") if isinstance(message, dict) else ""

        error_data = error_data or types.ErrorData(code=error_code, message=str(error), data=None)

        return types.JSONRPCMessage(types.JSONRPCError(jsonrpc=JSON_RPC_VERSION, id=message_id, error=error_data))

    async def _handle_rpc_msg(self, rpc_msg: types.JSONRPCMessage) -> types.JSONRPCMessage | None:
        msg_root = rpc_msg.root

        # --- Handle request ---
        if isinstance(msg_root, types.JSONRPCRequest):
            client_request = types.ClientRequest.model_validate(to_dict(msg_root))

            logger.info(f"Handling request: {client_request}")
            response = await self._core.handle_request(client_request)
            logger.info(f"Handled request: {client_request}")

            if response is None:
                # Request was cancelled - Do nothing
                return None
            elif isinstance(response, types.ErrorData):
                raise ErrorWithData(response)

            logger.info(f"Returning response: {response}")
            return types.JSONRPCMessage(
                types.JSONRPCResponse(
                    jsonrpc=JSON_RPC_VERSION,
                    id=msg_root.id,
                    result=to_dict(response),
                )
            )

        # --- Handle notification ---
        elif isinstance(msg_root, types.JSONRPCNotification):
            # TODO: Add full support for client notification - This just implements the handler.
            client_notification = types.ClientNotification.model_validate(to_dict(msg_root))

            logger.info(f"Handling notification: {client_notification}")
            await self._core.handle_notification(client_notification)
            logger.info(f"Handled notification: {client_notification}")
            return None
        else:
            raise UnsupportedRPCMessageType("Message to MCP server must be a request or notification")

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
