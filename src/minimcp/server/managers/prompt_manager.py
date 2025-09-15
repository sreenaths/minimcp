import builtins
import inspect
import logging
from collections.abc import Callable
from functools import partial
from typing import Any

import pydantic_core
from mcp.server.lowlevel.server import Server
from mcp.types import AnyFunction, GetPromptResult, Prompt, PromptArgument, PromptMessage, TextContent
from pydantic import TypeAdapter, validate_call
from typing_extensions import TypedDict, Unpack

from minimcp.utils.func import extract_func_details, validate_func_name

logger = logging.getLogger(__name__)


class PromptDefinition(TypedDict, total=False):
    name: str | None
    title: str | None
    description: str | None
    meta: dict[str, Any] | None


class PromptManager:
    _prompts: dict[str, tuple[Prompt, AnyFunction]]

    def __init__(self, core: Server):
        self._prompts = {}
        self._hook_core(core)

    def _hook_core(self, core: Server) -> None:
        core.list_prompts()(self._async_list)
        core.get_prompt()(self.get)
        # core.complete()(self._async_complete) # TODO: Implement completion for prompts

    def __call__(self, **kwargs: Unpack[PromptDefinition]) -> Callable[[Callable], Prompt]:
        """
        Decorator to add a prompt to the MCP prompt manager.
        """
        return partial(self.add, **kwargs)

    def add(self, func: AnyFunction, **kwargs: Unpack[PromptDefinition]) -> Prompt:
        """
        Add a prompt to the MCP prompt manager.
        """

        details = extract_func_details(func)

        prompt_name = validate_func_name(kwargs.get("name", details.name))
        if prompt_name in self._prompts:
            raise ValueError(f"Prompt {prompt_name} already registered")

        # Get schema from TypeAdapter - will fail if function isn't properly typed
        parameters = TypeAdapter(func).json_schema()

        # Convert parameters to PromptArguments
        arguments: list[PromptArgument] = []
        if "properties" in parameters:
            for param_name, param in parameters["properties"].items():
                required = param_name in parameters.get("required", [])
                arguments.append(
                    PromptArgument(
                        name=param_name,
                        description=param.get("description"),
                        required=required,
                    )
                )

        # ensure the arguments are properly cast
        func = validate_call(func)

        prompt = Prompt(
            name=prompt_name,
            title=kwargs.get("title", None),
            description=kwargs.get("description", details.doc),
            arguments=arguments,
            _meta=kwargs.get("meta", None),
        )

        self._prompts[prompt_name] = (prompt, func)

        return prompt

    def remove(self, name: str) -> Prompt:
        """
        Remove a prompt from the MCP prompt manager.
        """
        if name not in self._prompts:
            raise ValueError(f"Prompt {name} not found")

        return self._prompts.pop(name)[0]

    def list(self) -> builtins.list[Prompt]:
        return [prompt[0] for prompt in self._prompts.values()]

    async def _async_list(self) -> builtins.list[Prompt]:
        return self.list()

    async def get(self, name: str, args: dict[str, str] | None) -> GetPromptResult:
        if name not in self._prompts:
            raise ValueError(f"Prompt {name} not found")

        try:
            prompt = self._prompts[name][0]
            prompt_func = self._prompts[name][1]

            if prompt.arguments:
                self._validate_arguments(prompt.arguments, args)

            messages = await self._format_messages(prompt_func, args)

            return GetPromptResult(
                description=prompt.description,
                messages=messages,
                _meta=prompt.meta,
            )
        except Exception as e:
            msg = f"Error getting prompt {name}: {e}"
            logger.exception(msg)
            raise ValueError(msg)

    def _validate_arguments(
        self, prompt_arguments: builtins.list[PromptArgument], available_args: dict[str, Any] | None
    ) -> None:
        """Validate the arguments."""
        required = {arg.name for arg in prompt_arguments if arg.required}
        provided = set(available_args or {})

        missing = required - provided
        if missing:
            raise ValueError(f"Missing required arguments: {missing}")

    async def _format_messages(
        self, prompt_func: AnyFunction, arguments: dict[str, Any] | None = None
    ) -> builtins.list[PromptMessage]:
        """Format the prompt with arguments."""

        # Call function and check if result is a coroutine
        result = prompt_func(**(arguments or {}))

        if inspect.iscoroutine(result):
            result = await result

        return self._convert_result(result)

    def _convert_result(self, result: Any) -> builtins.list[PromptMessage]:
        """Convert the result to messages."""

        if not isinstance(result, list | tuple):
            result = [result]

        try:
            messages: list[PromptMessage] = []

            for msg in result:
                if isinstance(msg, PromptMessage):
                    messages.append(msg)
                elif isinstance(msg, dict):
                    # Try to validate as PromptMessage
                    messages.append(PromptMessage.model_validate(msg))
                elif isinstance(msg, str):
                    # Create a user message with text content
                    content = TextContent(type="text", text=msg)
                    messages.append(PromptMessage(role="user", content=content))
                else:
                    # Convert to JSON string and create user message
                    content_text = pydantic_core.to_json(msg, fallback=str, indent=2).decode()
                    content = TextContent(type="text", text=content_text)
                    messages.append(PromptMessage(role="user", content=content))

            return messages
        except Exception:
            raise ValueError("Could not convert prompt result to message")
