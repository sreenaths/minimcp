import builtins
import inspect
from collections.abc import Callable
from functools import partial
from typing import Any

import mcp.types as types
from mcp.server.fastmcp.utilities.func_metadata import FuncMetadata, func_metadata
from mcp.server.lowlevel.server import Server
from typing_extensions import TypedDict, Unpack


class ToolDetails(TypedDict, total=False):
    name: str | None
    description: str | None
    annotations: types.ToolAnnotations | None
    meta: dict[str, Any] | None


class ToolManager:
    """
    Manages tool definitions and handlers.
    """

    _tools: dict[str, tuple[types.Tool, types.AnyFunction, FuncMetadata]]

    def __init__(self, core: Server):
        self._tools = {}
        self._hook_core(core)

    def __call__(self, **kwargs: Unpack[ToolDetails]) -> Callable[[Callable], types.Tool]:
        """
        Decorator to add a tool to the MCP tool manager.
        """
        return partial(self.add, **kwargs)

    def add(self, func: types.AnyFunction, **kwargs: Unpack[ToolDetails]) -> types.Tool:
        """
        Add a tool to the MCP tool manager.
        """

        tool_name = kwargs.get("name", func.__name__)

        if not tool_name:
            raise ValueError("Tool name is required")

        if tool_name in self._tools:
            raise ValueError(f"Tool {tool_name} already registered")

        func_meta = func_metadata(func)
        parameters = func_meta.arg_model.model_json_schema(by_alias=True)

        tool = types.Tool(
            name=tool_name,
            description=kwargs.get("description", func.__doc__),
            inputSchema=parameters,
            outputSchema=func_meta.output_schema,
            annotations=kwargs.get("annotations", None),
            _meta=kwargs.get("meta", None),
        )

        self._tools[tool_name] = (tool, func, func_meta)

        return tool

    def remove(self, name: str) -> types.Tool:
        """
        Remove a tool from the MCP tool manager.
        """
        if name not in self._tools:
            raise ValueError(f"Tool {name} not found")

        return self._tools.pop(name)[0]

    def _hook_core(self, core: Server):
        core.list_tools()(self._async_list)
        # Validation done by func_meta in call. Hence passing validate_input=False
        # TODO: Ensure both the validations are similar
        core.call_tool(validate_input=False)(self.call)

    async def _async_list(self) -> builtins.list[types.Tool]:
        return self.list()

    def list(self) -> builtins.list[types.Tool]:
        return [tool[0] for tool in self._tools.values()]

    async def call(self, name: str, args: dict[str, Any]) -> Any:
        """
        Call a tool - Can be called from anywhere.
        """

        if name not in self._tools:
            raise ValueError(f"Tool {name} not found")

        _, handler, func_meta = self._tools[name]

        parsed_args = func_meta.pre_parse_json(args)
        validated_args = func_meta.arg_model.model_validate(parsed_args)
        result = handler(**validated_args.model_dump_one_level())

        if inspect.isawaitable(result):
            result = await result

        return func_meta.convert_result(result)
