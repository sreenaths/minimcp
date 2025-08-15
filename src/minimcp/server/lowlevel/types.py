from collections.abc import Iterable
from typing import Any, TypeAlias

from mcp.types import ContentBlock

# TODO: Could this be in mcp.types?


class NotificationOptions:
    def __init__(
        self,
        prompts_changed: bool = False,
        resources_changed: bool = False,
        tools_changed: bool = False,
    ):
        self.prompts_changed = prompts_changed
        self.resources_changed = resources_changed
        self.tools_changed = tools_changed


# Type aliases for tool call results
StructuredContent: TypeAlias = dict[str, Any]
UnstructuredContent: TypeAlias = Iterable[ContentBlock]
CombinationContent: TypeAlias = tuple[UnstructuredContent, StructuredContent]
