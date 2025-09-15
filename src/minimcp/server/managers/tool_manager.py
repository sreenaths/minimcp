import builtins
import inspect
import logging
from collections.abc import Callable
from functools import partial
from typing import Any

import mcp.types as types
from mcp.server.lowlevel.server import CombinationContent, Server, StructuredContent, UnstructuredContent
from typing_extensions import TypedDict, Unpack

from minimcp.utils.func import FuncDetails, extract_func_details, validate_func_name

logger = logging.getLogger(__name__)


class ToolDefinition(TypedDict, total=False):
    name: str | None
    title: str | None
    description: str | None
    annotations: types.ToolAnnotations | None
    meta: dict[str, Any] | None


class ToolManager:
    """
    Manages tool definitions and handlers.
    """

    _tools: dict[str, tuple[types.Tool, types.AnyFunction, FuncDetails]]

    def __init__(self, core: Server):
        self._tools = {}
        self._hook_core(core)

    def _hook_core(self, core: Server) -> None:
        core.list_tools()(self._async_list)

        # Validation done by func_meta in call. Hence passing validate_input=False
        # TODO: Ensure only one validation is required
        core.call_tool(validate_input=False)(self.call)

    def __call__(self, **kwargs: Unpack[ToolDefinition]) -> Callable[[Callable], types.Tool]:
        """
        Decorator to add a tool to the MCP tool manager.
        """
        return partial(self.add, **kwargs)

    def add(self, func: types.AnyFunction, **kwargs: Unpack[ToolDefinition]) -> types.Tool:
        """
        Add a tool to the MCP tool manager.
        """

        details = extract_func_details(func)

        tool_name = validate_func_name(kwargs.get("name", details.name))
        if tool_name in self._tools:
            raise ValueError(f"Tool {tool_name} already registered")

        parameters = details.meta.arg_model.model_json_schema(by_alias=True)

        tool = types.Tool(
            name=tool_name,
            title=kwargs.get("title", None),
            description=kwargs.get("description", details.doc),
            inputSchema=parameters,
            outputSchema=details.meta.output_schema,
            annotations=kwargs.get("annotations", None),
            _meta=kwargs.get("meta", None),
        )

        self._tools[tool_name] = (tool, func, details)
        logger.debug("Tool %s added", tool_name)

        return tool

    def remove(self, name: str) -> types.Tool:
        """
        Remove a tool from the MCP tool manager.
        """
        if name not in self._tools:
            raise ValueError(f"Tool {name} not found")

        logger.debug("Removing tool %s", name)
        return self._tools.pop(name)[0]

    async def _async_list(self) -> builtins.list[types.Tool]:
        return self.list()

    def list(self) -> builtins.list[types.Tool]:
        return [tool[0] for tool in self._tools.values()]

    async def call(
        self, name: str, args: dict[str, Any]
    ) -> UnstructuredContent | StructuredContent | CombinationContent:
        """
        Call a tool - Can be called from anywhere.
        """

        if name not in self._tools:
            raise ValueError(f"Tool {name} not found")

        _, handler, details = self._tools[name]

        parsed_args = details.meta.pre_parse_json(args)
        validated_args = details.meta.arg_model.model_validate(parsed_args)

        result = handler(**validated_args.model_dump_one_level())

        if inspect.iscoroutine(result):
            result = await result

        logger.debug("Tool %s handled with args %s", name, args)
        return details.meta.convert_result(result)
