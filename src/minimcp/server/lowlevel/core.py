# TODO: Delete this and import from mcp.server.lowlevel.core
# if and when the ServerCore module becomes part of the official MCP SDK

"""
MCP ServerCore Module

This module provides a framework for building MCP (Model Context Protocol) servers.
ServerCore is the basic building block - It sets the standard and allows you to easily
handle various types of MCP messages in an asynchronous manner.

ServerCore is stateless, and enforces no transport mechanism, session architecture or context management.
Additional features can be implemented by inheriting or compositing the ServerCore class
depending on your requirements.

Usage:
1. Create a ServerCore instance:
   mcp = ServerCore()

2. Define request handlers using decorators:
   @mcp.list_prompts()
   async def handle_list_prompts() -> list[types.Prompt]:
       # Implementation

   @mcp.get_prompt()
   async def handle_get_prompt(
       name: str, arguments: dict[str, str] | None
   ) -> types.GetPromptResult:
       # Implementation

   @mcp.list_tools()
   async def handle_list_tools() -> list[types.Tool]:
       # Implementation

   @mcp.call_tool()
   async def handle_call_tool(
       name: str, arguments: dict | None
   ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
       # Implementation

   @mcp.list_resource_templates()
   async def handle_list_resource_templates() -> list[types.ResourceTemplate]:
       # Implementation

3. Define notification handlers if needed:
   @mcp.progress_notification()
   async def handle_progress(
       progress_token: str | int, progress: float, total: float | None,
       message: str | None
   ) -> None:
       # Implementation

4. Handle request:
    req: types.ClientRequest = ...
    response: types.ServerResult = await mcp.handle_request(req)
"""

import json
import logging
import warnings
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, cast

import anyio
import jsonschema
import mcp.types as types
from mcp.server.lowlevel.helper_types import ReadResourceContents  # TODO: Could this be in mcp.types?
from mcp.server.models import InitializationOptions
from mcp.shared.exceptions import McpError
from pydantic import AnyUrl

from minimcp.server.lowlevel.types import (
    CombinationContent,
    NotificationOptions,
    StructuredContent,
    UnstructuredContent,
)

logger = logging.getLogger(__name__)


class ServerCore:
    name: str
    version: str | None = None
    instructions: str | None = None

    request_handlers: dict[type, Callable[..., Awaitable[types.ServerResult]]]
    notification_handlers: dict[type, Callable[..., Awaitable[None]]]
    _tool_cache: dict[str, types.Tool]

    def __init__(
        self,
        name: str,
        version: str | None = None,
        instructions: str | None = None,
    ):
        self.name = name
        self.version = version
        self.instructions = instructions

        self.request_handlers = {
            types.PingRequest: _ping_handler,
        }
        self.notification_handlers = {}
        self._tool_cache = {}

    def create_initialization_options(
        self,
        notification_options: NotificationOptions | None = None,
        experimental_capabilities: dict[str, dict[str, Any]] | None = None,
    ) -> InitializationOptions:
        """Create initialization options from this server instance."""

        def pkg_version(package: str) -> str:
            try:
                from importlib.metadata import version

                return version(package)
            except Exception:
                pass

            return "unknown"

        return InitializationOptions(
            server_name=self.name,
            server_version=self.version if self.version else pkg_version("mcp"),
            capabilities=self.get_capabilities(
                notification_options or NotificationOptions(),
                experimental_capabilities or {},
            ),
            instructions=self.instructions,
        )

    def get_capabilities(
        self,
        notification_options: NotificationOptions,
        experimental_capabilities: dict[str, dict[str, Any]],
    ) -> types.ServerCapabilities:
        """Convert existing handlers to a ServerCapabilities object."""
        prompts_capability = None
        resources_capability = None
        tools_capability = None
        logging_capability = None
        completions_capability = None

        # Set prompt capabilities if handler exists
        if types.ListPromptsRequest in self.request_handlers:
            prompts_capability = types.PromptsCapability(listChanged=notification_options.prompts_changed)

        # Set resource capabilities if handler exists
        if types.ListResourcesRequest in self.request_handlers:
            resources_capability = types.ResourcesCapability(
                subscribe=False, listChanged=notification_options.resources_changed
            )

        # Set tool capabilities if handler exists
        if types.ListToolsRequest in self.request_handlers:
            tools_capability = types.ToolsCapability(listChanged=notification_options.tools_changed)

        # Set logging capabilities if handler exists
        if types.SetLevelRequest in self.request_handlers:
            logging_capability = types.LoggingCapability()

        # Set completions capabilities if handler exists
        if types.CompleteRequest in self.request_handlers:
            completions_capability = types.CompletionsCapability()

        return types.ServerCapabilities(
            prompts=prompts_capability,
            resources=resources_capability,
            tools=tools_capability,
            logging=logging_capability,
            experimental=experimental_capabilities,
            completions=completions_capability,
        )

    # -- Prompt handlers --
    def list_prompts(self):
        def decorator(func: Callable[[], Awaitable[list[types.Prompt]]]):
            logger.debug("Registering handler for PromptListRequest")

            async def handler(_: Any):
                prompts = await func()
                return types.ServerResult(types.ListPromptsResult(prompts=prompts))

            self.request_handlers[types.ListPromptsRequest] = handler
            return func

        return decorator

    def get_prompt(self):
        def decorator(
            func: Callable[[str, dict[str, str] | None], Awaitable[types.GetPromptResult]],
        ):
            logger.debug("Registering handler for GetPromptRequest")

            async def handler(req: types.GetPromptRequest):
                prompt_get = await func(req.params.name, req.params.arguments)
                return types.ServerResult(prompt_get)

            self.request_handlers[types.GetPromptRequest] = handler
            return func

        return decorator

    # -- Resource handlers --
    def list_resources(self):
        def decorator(func: Callable[[], Awaitable[list[types.Resource]]]):
            logger.debug("Registering handler for ListResourcesRequest")

            async def handler(_: Any):
                resources = await func()
                return types.ServerResult(types.ListResourcesResult(resources=resources))

            self.request_handlers[types.ListResourcesRequest] = handler
            return func

        return decorator

    def list_resource_templates(self):
        def decorator(func: Callable[[], Awaitable[list[types.ResourceTemplate]]]):
            logger.debug("Registering handler for ListResourceTemplatesRequest")

            async def handler(_: Any):
                templates = await func()
                return types.ServerResult(types.ListResourceTemplatesResult(resourceTemplates=templates))

            self.request_handlers[types.ListResourceTemplatesRequest] = handler
            return func

        return decorator

    def read_resource(self):
        def decorator(
            func: Callable[[AnyUrl], Awaitable[str | bytes | Iterable[ReadResourceContents]]],
        ):
            logger.debug("Registering handler for ReadResourceRequest")

            async def handler(req: types.ReadResourceRequest):
                result = await func(req.params.uri)

                def create_content(data: str | bytes, mime_type: str | None):
                    match data:
                        case str() as data:
                            return types.TextResourceContents(
                                uri=req.params.uri,
                                text=data,
                                mimeType=mime_type or "text/plain",
                            )
                        case bytes() as data:
                            import base64

                            return types.BlobResourceContents(
                                uri=req.params.uri,
                                blob=base64.b64encode(data).decode(),
                                mimeType=mime_type or "application/octet-stream",
                            )

                match result:
                    case str() | bytes() as data:
                        warnings.warn(
                            "Returning str or bytes from read_resource is deprecated. "
                            "Use Iterable[ReadResourceContents] instead.",
                            DeprecationWarning,
                            stacklevel=2,
                        )
                        content = create_content(data, None)
                    case Iterable() as contents:
                        contents_list = [
                            create_content(content_item.content, content_item.mime_type) for content_item in contents
                        ]
                        return types.ServerResult(
                            types.ReadResourceResult(
                                contents=contents_list,
                            )
                        )
                    case _:  # pyright: ignore[reportUnnecessaryComparison]
                        raise ValueError(f"Unexpected return type from read_resource: {type(result)}")

                return types.ServerResult(
                    types.ReadResourceResult(
                        contents=[content],
                    )
                )

            self.request_handlers[types.ReadResourceRequest] = handler
            return func

        return decorator

    def subscribe_resource(self):
        def decorator(func: Callable[[AnyUrl], Awaitable[None]]):
            logger.debug("Registering handler for SubscribeRequest")

            async def handler(req: types.SubscribeRequest):
                await func(req.params.uri)
                return types.ServerResult(types.EmptyResult())

            self.request_handlers[types.SubscribeRequest] = handler
            return func

        return decorator

    def unsubscribe_resource(self):
        def decorator(func: Callable[[AnyUrl], Awaitable[None]]):
            logger.debug("Registering handler for UnsubscribeRequest")

            async def handler(req: types.UnsubscribeRequest):
                await func(req.params.uri)
                return types.ServerResult(types.EmptyResult())

            self.request_handlers[types.UnsubscribeRequest] = handler
            return func

        return decorator

    # -- Logging handlers --
    def set_logging_level(self):
        def decorator(func: Callable[[types.LoggingLevel], Awaitable[None]]):
            logger.debug("Registering handler for SetLevelRequest")

            async def handler(req: types.SetLevelRequest):
                await func(req.params.level)
                return types.ServerResult(types.EmptyResult())

            self.request_handlers[types.SetLevelRequest] = handler
            return func

        return decorator

    # -- Tool handlers --
    def list_tools(self):
        def decorator(func: Callable[[], Awaitable[list[types.Tool]]]):
            logger.debug("Registering handler for ListToolsRequest")

            async def handler(_: Any):
                tools = await func()
                # Refresh the tool cache
                self._tool_cache.clear()
                for tool in tools:
                    self._tool_cache[tool.name] = tool
                return types.ServerResult(types.ListToolsResult(tools=tools))

            self.request_handlers[types.ListToolsRequest] = handler
            return func

        return decorator

    def _make_error_result(self, error_message: str) -> types.ServerResult:
        """Create a ServerResult with an error CallToolResult."""
        return types.ServerResult(
            types.CallToolResult(
                content=[types.TextContent(type="text", text=error_message)],
                isError=True,
            )
        )

    async def _get_cached_tool_definition(self, tool_name: str) -> types.Tool | None:
        """Get tool definition from cache, refreshing if necessary.

        Returns the Tool object if found, None otherwise.
        """
        if tool_name not in self._tool_cache:
            if types.ListToolsRequest in self.request_handlers:
                logger.debug("Tool cache miss for %s, refreshing cache", tool_name)
                await self.request_handlers[types.ListToolsRequest](None)

        tool = self._tool_cache.get(tool_name)
        if tool is None:
            logger.warning("Tool '%s' not listed, no validation will be performed", tool_name)

        return tool

    def call_tool(self, *, validate_input: bool = True):
        """Register a tool call handler.

        Args:
            validate_input: If True, validates input against inputSchema. Default is True.

        The handler validates input against inputSchema (if validate_input=True), calls the tool function,
        and builds a CallToolResult with the results:
        - Unstructured content (iterable of ContentBlock): returned in content
        - Structured content (dict): returned in structuredContent, serialized JSON text returned in content
        - Both: returned in content and structuredContent

        If outputSchema is defined, validates structuredContent or errors if missing.
        """

        def decorator(
            func: Callable[
                ...,
                Awaitable[UnstructuredContent | StructuredContent | CombinationContent],
            ],
        ):
            logger.debug("Registering handler for CallToolRequest")

            async def handler(req: types.CallToolRequest):
                try:
                    tool_name = req.params.name
                    arguments = req.params.arguments or {}
                    tool = await self._get_cached_tool_definition(tool_name)

                    # input validation
                    if validate_input and tool:
                        try:
                            jsonschema.validate(instance=arguments, schema=tool.inputSchema)
                        except jsonschema.ValidationError as e:
                            return self._make_error_result(f"Input validation error: {e.message}")

                    # tool call
                    results = await func(tool_name, arguments)

                    # output normalization
                    unstructured_content: UnstructuredContent
                    maybe_structured_content: StructuredContent | None
                    if isinstance(results, tuple) and len(results) == 2:
                        # tool returned both structured and unstructured content
                        unstructured_content, maybe_structured_content = cast(CombinationContent, results)
                    elif isinstance(results, dict):
                        # tool returned structured content only
                        maybe_structured_content = cast(StructuredContent, results)
                        unstructured_content = [types.TextContent(type="text", text=json.dumps(results, indent=2))]
                    elif hasattr(results, "__iter__"):
                        # tool returned unstructured content only
                        unstructured_content = cast(UnstructuredContent, results)
                        maybe_structured_content = None
                    else:
                        return self._make_error_result(f"Unexpected return type from tool: {type(results).__name__}")

                    # output validation
                    if tool and tool.outputSchema is not None:
                        if maybe_structured_content is None:
                            return self._make_error_result(
                                "Output validation error: outputSchema defined but no structured output returned"
                            )
                        else:
                            try:
                                jsonschema.validate(instance=maybe_structured_content, schema=tool.outputSchema)
                            except jsonschema.ValidationError as e:
                                return self._make_error_result(f"Output validation error: {e.message}")

                    # result
                    return types.ServerResult(
                        types.CallToolResult(
                            content=list(unstructured_content),
                            structuredContent=maybe_structured_content,
                            isError=False,
                        )
                    )
                except Exception as e:
                    return self._make_error_result(str(e))

            self.request_handlers[types.CallToolRequest] = handler
            return func

        return decorator

    # -- Other handlers --
    def progress_notification(self):
        def decorator(
            func: Callable[[str | int, float, float | None, str | None], Awaitable[None]],
        ):
            logger.debug("Registering handler for ProgressNotification")

            async def handler(req: types.ProgressNotification):
                await func(
                    req.params.progressToken,
                    req.params.progress,
                    req.params.total,
                    req.params.message,
                )

            self.notification_handlers[types.ProgressNotification] = handler
            return func

        return decorator

    def completion(self):
        """Provides completions for prompts and resource templates"""

        def decorator(
            func: Callable[
                [
                    types.PromptReference | types.ResourceTemplateReference,
                    types.CompletionArgument,
                    types.CompletionContext | None,
                ],
                Awaitable[types.Completion | None],
            ],
        ):
            logger.debug("Registering handler for CompleteRequest")

            async def handler(req: types.CompleteRequest):
                completion = await func(req.params.ref, req.params.argument, req.params.context)
                return types.ServerResult(
                    types.CompleteResult(
                        completion=completion
                        if completion is not None
                        else types.Completion(values=[], total=None, hasMore=None),
                    )
                )

            self.request_handlers[types.CompleteRequest] = handler
            return func

        return decorator

    async def handle_request(self, request: types.ClientRequest) -> types.ServerResult | types.ErrorData | None:
        if handler := self.request_handlers.get(type(request.root)):  # type: ignore
            logger.debug("Dispatching request of type %s", type(request.root).__name__)

            try:
                response = await handler(request.root)
            except McpError as err:
                response = err.error
            except anyio.get_cancelled_exc_class():
                logger.info(
                    "Request %s cancelled - duplicate response suppressed",
                )
                return None
            except Exception as err:
                response = types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=str(err),
                    data=None,
                )
                response._error = err  # pyright: ignore

        else:
            response = types.ErrorData(
                code=types.METHOD_NOT_FOUND,
                message="Method not found",
            )

        return response

    async def handle_notification(self, notification: types.ClientNotification):
        notify = notification.root
        if handler := self.notification_handlers.get(type(notify)):  # type: ignore
            logger.debug("Dispatching notification of type %s", type(notify).__name__)

            try:
                await handler(notify)
            except Exception:
                logger.exception("Uncaught exception in notification handler")


async def _ping_handler(request: types.PingRequest) -> types.ServerResult:
    return types.ServerResult(types.EmptyResult())
