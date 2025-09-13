import json
import logging
import uuid
from typing import Generic

import anyio
import mcp.shared.version as version
import mcp.types as types
from mcp.server.lowlevel.server import NotificationOptions, Server
from pydantic import ValidationError

import minimcp.server.json_rpc as json_rpc
from minimcp.server.exceptions import (
    ContextError,
    InvalidParamsError,
    MethodNotFoundError,
    ParserError,
    UnsupportedRPCMessageType,
)
from minimcp.server.limiter import Limiter
from minimcp.server.managers.context_manager import Context, ContextManager, ScopeT
from minimcp.server.managers.prompt_manager import PromptManager
from minimcp.server.managers.resource_manager import ResourceManager
from minimcp.server.managers.tool_manager import ToolManager
from minimcp.server.responder import Responder
from minimcp.server.types import Message, NoMessage, Send
from minimcp.utils.model import to_dict, to_json

logger = logging.getLogger(__name__)


class MiniMCP(Generic[ScopeT]):
    _core: Server
    _notification_options: NotificationOptions | None = None

    _raise_exceptions: bool
    _limiter: Limiter

    tool: ToolManager
    prompt: PromptManager
    resource: ResourceManager
    context: ContextManager[ScopeT]

    def __init__(
        self,
        name: str,
        version: str | None = None,
        instructions: str | None = None,
        idle_timeout: int = 30,
        max_concurrency: int = 100,
        raise_exceptions: bool = False,
    ) -> None:
        """
        Initialize the MCP server.

        Args:
            name: The name of the MCP server.
            version: The version of the MCP server.
            instructions: The instructions for the MCP server.

            idle_timeout: Time in seconds after which a message handler will be considered idle and timed out.
            max_concurrency: The maximum number of message handlers that could be run at
                the same time, beyond which the handle() calls will be blocked.
            raise_exceptions: Whether to raise uncaught exceptions while handling
                messages. Useful for development and testing.
        """
        self._raise_exceptions = raise_exceptions
        self._limiter = Limiter(idle_timeout, max_concurrency)

        # TODO: Add support for server-to-client notifications
        self._notification_options = NotificationOptions(
            prompts_changed=False,
            resources_changed=False,
            tools_changed=False,
        )

        # Setup core
        self._core = Server(name=name, version=version, instructions=instructions)
        self._core.request_handlers[types.InitializeRequest] = self._initialize_handler
        # MiniMCP handles InitializeRequest but not InitializedNotification as it is stateless

        # Setup managers
        self.tool = ToolManager(self._core)
        self.prompt = PromptManager(self._core)
        self.resource = ResourceManager(self._core)

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
    async def handle(
        self, message: Message, send: Send | None = None, scope: ScopeT | None = None
    ) -> Message | NoMessage:
        message_id = ""
        try:
            message_id, rpc_msg = self._parse_message(message)

            async with self._limiter() as time_limiter:
                responder = Responder(message, send, time_limiter) if send else None
                context = Context(message=rpc_msg, time_limiter=time_limiter, scope=scope, responder=responder)
                with self.context.active(context):
                    response = await self._handle_rpc_msg(rpc_msg)

        # --- Centralized MCP error handling - Handle all MCP/JSON-RPC exceptions here ---
        # - Each transport may implement additional handling if needed.
        # - Error inside each tool call will be handled by the core and returned as part of CallToolResult.
        except ParserError as e:
            logger.error("Parser failed: %s", e)
            response = json_rpc.build_error_message(types.PARSE_ERROR, message_id, e)
        except InvalidParamsError as e:
            logger.error("Invalid params: %s", e)
            response = json_rpc.build_error_message(types.INVALID_PARAMS, message_id, e)
        except UnsupportedRPCMessageType as e:
            logger.error("Unsupported message type: %s", e)
            response = json_rpc.build_error_message(types.INVALID_REQUEST, message_id, e)
        except MethodNotFoundError as e:
            logger.error("Method not found: %s", e)
            response = json_rpc.build_error_message(types.METHOD_NOT_FOUND, message_id, e)
        except TimeoutError as e:
            logger.error("Message handler timed out: %s", e)
            response = json_rpc.build_error_message(types.INTERNAL_ERROR, message_id, e)
        except (Exception, ContextError) as e:
            logger.exception("Unhandled exception")
            if self._raise_exceptions:
                raise
            response = json_rpc.build_error_message(types.INTERNAL_ERROR, message_id, e)
        except anyio.get_cancelled_exc_class() as e:
            logger.debug("Task cancelled: %s. Message: %s", e, message)
            raise  # Cancel must be re-raised

        if isinstance(response, NoMessage):
            return response

        return to_json(response)

    def _parse_message(self, message: Message) -> tuple[str | int, types.JSONRPCMessage]:
        try:
            dict_msg = json.loads(message)
            message_id = dict_msg.get("id", "") if isinstance(dict_msg, dict) else ""
            rpc_msg = types.JSONRPCMessage.model_validate(dict_msg)
            return message_id, rpc_msg
        except json.JSONDecodeError as e:
            raise ParserError(str(e)) from e
        except ValidationError as e:
            raise InvalidParamsError(str(e)) from e

    async def _handle_rpc_msg(self, rpc_msg: types.JSONRPCMessage) -> types.JSONRPCMessage | NoMessage:
        msg_root = rpc_msg.root

        # --- Handle request ---
        if isinstance(msg_root, types.JSONRPCRequest):
            client_request = types.ClientRequest.model_validate(to_dict(msg_root))

            logger.debug("Handling request %s - %s", msg_root.id, client_request)
            response = await self._handle_client_request(client_request)
            logger.info("Successfully handled request %s - Response: %s", msg_root.id, response)

            return json_rpc.build_response_message(msg_root.id, response)

        # --- Handle notification ---
        elif isinstance(msg_root, types.JSONRPCNotification):
            # TODO: Add full support for client notification - This just implements the handler.
            client_notification = types.ClientNotification.model_validate(to_dict(msg_root))
            notification_id = uuid.uuid4()  # Creating an id for debugging

            logger.debug("Handling notification %s - %s", notification_id, client_notification)
            response = await self._handle_client_notification(client_notification)
            logger.info("Successfully handled notification %s", notification_id)

            return response
        else:
            raise UnsupportedRPCMessageType("Message to MCP server must be a request or notification")

    async def _handle_client_request(self, request: types.ClientRequest) -> types.ServerResult:
        request_type = type(request.root)
        if handler := self._core.request_handlers.get(request_type):
            logger.debug("Dispatching request of type %s", request_type.__name__)
            return await handler(request.root)
        else:
            raise MethodNotFoundError(f"Method not found for request type {request_type.__name__}")

    async def _handle_client_notification(self, notification: types.ClientNotification) -> NoMessage:
        notification_type = type(notification.root)
        if handler := self._core.notification_handlers.get(notification_type):
            logger.debug("Dispatching notification of type %s", notification_type.__name__)

            try:
                # Avoiding the "fire-and-forget" pattern for notifications at the server layer.
                # This behavior should be handled at the transport layer.
                # This ensures all handlers are explicitly controlled and have a defined time to live.
                await handler(notification.root)
            except Exception:
                logger.exception("Uncaught exception in notification handler")

        else:
            logger.debug("No handler found for notification type %s", notification_type.__name__)

        return NoMessage.NOTIFICATION

    async def _initialize_handler(self, req: types.InitializeRequest) -> types.ServerResult:
        client_protocol_version = req.params.protocolVersion
        server_protocol_version = (
            client_protocol_version
            if client_protocol_version in version.SUPPORTED_PROTOCOL_VERSIONS
            else types.LATEST_PROTOCOL_VERSION
        )
        # TODO: Error handling on protocol version mismatch. Handled in HTTP transport.
        # https://modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle#error-handling

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
