import json
import logging
from typing import Any, Generic

import anyio
import mcp.shared.version as version
import mcp.types as types
from mcp.server.lowlevel.server import NotificationOptions, Server
from mcp.shared.exceptions import McpError
from pydantic import ValidationError

import minimcp.server.json_rpc as json_rpc
from minimcp.server.managers.context_manager import ContextManager, ScopeT
from minimcp.server.utils import to_dict

from .exceptions import UnsupportedRPCMessageType
from .managers.tool_manager import ToolManager

logger = logging.getLogger(__name__)


class MiniMCP(Generic[ScopeT]):
    _core: Server
    _notification_options: NotificationOptions | None = None

    _timeout: int
    _max_concurrency: int
    _raise_exceptions: bool

    tool: ToolManager
    context: ContextManager[ScopeT]

    def __init__(
        self,
        name: str,
        version: str | None = None,
        instructions: str | None = None,
        timeout: int = 30,
        max_concurrency: int = 100,
        raise_exceptions: bool = False,
    ) -> None:
        """
        Initialize the MCP server.

        Args:
            name: The name of the MCP server.
            version: The version of the MCP server.
            instructions: The instructions for the MCP server.

            timeout: Time in seconds after which a message handler will timeout.
            max_concurrency: The maximum number of message handlers that could be run at
                the same time, beyond which the handle() calls will be blocked.
            raise_exceptions: Whether to raise uncaught exceptions while handling messages.
        """
        self._timeout = timeout
        self._max_concurrency = max_concurrency

        self._raise_exceptions = raise_exceptions
        self._limiter = anyio.CapacityLimiter(self._max_concurrency)

        # TODO: Add support for automatic server-to-client notifications
        self._notification_options = NotificationOptions(
            prompts_changed=False,
            resources_changed=False,
            tools_changed=False,
        )

        # Setup core
        self._core = Server(name=name, version=version, instructions=instructions)
        self._core.request_handlers[types.InitializeRequest] = self._initialize_handler

        # Setup managers
        self.tool = ToolManager(self._core)
        self.context = ContextManager()

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

    # --- Handlers ---
    async def handle(self, message: dict[str, Any], scope: ScopeT | None = None) -> dict[str, Any] | None:
        try:
            rpc_msg = types.JSONRPCMessage.model_validate(message)

            async with self._limiter:
                with anyio.fail_after(self._timeout):
                    with self.context.active(rpc_msg, scope):
                        response = await self._handle_rpc_msg(rpc_msg)

        # --- Centralized MCP error handling - All expected exceptions must be handled here ---
        except ValidationError as e:
            logger.error("Invalid Message: %s", e)
            response = json_rpc.build_error_message(types.INVALID_REQUEST, message, e)
        except UnsupportedRPCMessageType as e:
            logger.error("Unsupported Message Type: %s", e)
            response = json_rpc.build_error_message(types.INVALID_REQUEST, message, e)
        except McpError as e:
            logger.error("Error While Handling Message: %s", e)
            response = json_rpc.build_error_message(types.INTERNAL_ERROR, message, e, e.error)
        except TimeoutError as e:
            logger.error("Message Handler Timed Out: %s", message)
            response = json_rpc.build_error_message(types.INTERNAL_ERROR, message, e)
        except anyio.get_cancelled_exc_class():
            logger.info("Message Cancelled: %s", message)
            response = None
        except Exception as e:
            logger.error("Uncaught Exception: %s", e)
            if self._raise_exceptions:
                raise
            response = json_rpc.build_error_message(types.INTERNAL_ERROR, message, e)

        if response is None:
            logger.info(f"No response returned for message: {message}")
            return None

        return to_dict(response)

    async def handles(
        self, message: str, scope: ScopeT | None = None, raise_json_decode_error: bool = False
    ) -> str | None:
        try:
            message_dict = json.loads(message)
        except json.JSONDecodeError as e:
            if raise_json_decode_error:
                raise
            response = to_dict(json_rpc.build_error_message(types.PARSE_ERROR, {}, e))
        else:
            response = await self.handle(message_dict, scope)

        return json.dumps(response) if response is not None else None

    async def _handle_rpc_msg(self, rpc_msg: types.JSONRPCMessage) -> types.JSONRPCMessage | None:
        msg_root = rpc_msg.root

        # --- Handle request ---
        if isinstance(msg_root, types.JSONRPCRequest):
            client_request = types.ClientRequest.model_validate(to_dict(msg_root))

            logger.info(f"Handling request: {client_request}")
            response = await self._handle_client_request(client_request)
            logger.info(f"Handled request: {client_request}")

            if response is None:
                # Request was cancelled - Do nothing
                return None

            logger.info(f"Returning response: {response}")
            return json_rpc.build_response_message(msg_root.id, response)

        # --- Handle notification ---
        elif isinstance(msg_root, types.JSONRPCNotification):
            # TODO: Add full support for client notification - This just implements the handler.
            client_notification = types.ClientNotification.model_validate(to_dict(msg_root))

            logger.info(f"Handling notification: {client_notification}")
            await self._handle_client_notification(client_notification)
            logger.info(f"Handled notification: {client_notification}")
            return None
        else:
            raise UnsupportedRPCMessageType("Message to MCP server must be a request or notification")

    async def _handle_client_request(self, request: types.ClientRequest) -> types.ServerResult:
        if handler := self._core.request_handlers.get(type(request.root)):  # type: ignore
            logger.debug("Dispatching request of type %s", type(request.root).__name__)
            return await handler(request.root)
        else:
            raise McpError(types.ErrorData(code=types.METHOD_NOT_FOUND, message="Method not found"))

    async def _handle_client_notification(self, notification: types.ClientNotification):
        notify = notification.root
        if handler := self._core.notification_handlers.get(type(notify)):  # type: ignore
            logger.debug("Dispatching notification of type %s", type(notify).__name__)

            try:
                await handler(notify)
            except Exception:
                logger.exception("Uncaught exception in notification handler")

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
