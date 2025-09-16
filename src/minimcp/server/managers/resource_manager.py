import builtins
import inspect
import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import pydantic_core
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.lowlevel.server import Server
from mcp.types import Annotations, AnyFunction, Resource, ResourceTemplate
from pydantic import AnyUrl
from typing_extensions import TypedDict, Unpack

from minimcp.utils.func import extract_func_details, validate_func_name

logger = logging.getLogger(__name__)


class ResourceDefinition(TypedDict, total=False):
    name: str | None
    title: str | None
    description: str | None
    mime_type: str | None
    annotations: Annotations | None
    meta: dict[str, Any] | None


@dataclass
class _ResourceDetails:
    resource: Resource | ResourceTemplate
    func: AnyFunction
    normalized_uri: str
    uri_pattern: re.Pattern | None = None  # Pattern calculated only for resource templates


TEMPLATE_PARAM_REGEX = re.compile(r"{(\w+)}")


class ResourceManager:
    _resources: dict[str, _ResourceDetails]

    def __init__(self, core: Server):
        self._resources = {}
        self._hook_core(core)

    def _hook_core(self, core: Server) -> None:
        core.list_resources()(self._async_list)
        core.list_resource_templates()(self._async_list_templates)
        core.read_resource()(self.read)
        # core.subscribe_resource()(self.get) # TODO: Implement
        # core.unsubscribe_resource()(self.get) # TODO: Implement

    def __call__(
        self, uri: str, **kwargs: Unpack[ResourceDefinition]
    ) -> Callable[[AnyFunction], Resource | ResourceTemplate]:
        """
        Decorator to add a resource to the MCP resource manager.
        """

        def decorator(func: AnyFunction) -> Resource | ResourceTemplate:
            return self.add(func, uri, **kwargs)

        return decorator

    def _get_normalized_uri(self, uri: str) -> str:
        """
        Normalize URI - replace parameters with | to deduplicate
        """
        return TEMPLATE_PARAM_REGEX.sub("|", uri)

    def add(self, func: AnyFunction, uri: str, **kwargs: Unpack[ResourceDefinition]) -> Resource | ResourceTemplate:
        """
        Add a resource or resource template to the MCP resource manager.
        """

        if not uri:
            raise ValueError("URI is required, pass it as part of the options")

        details = extract_func_details(func)

        resource_name = validate_func_name(kwargs.get("name", details.name))
        if resource_name in self._resources:
            raise ValueError(f"Resource {resource_name} already registered")

        normalized_uri = self._get_normalized_uri(uri)

        # Check if similar resource/resource template has already been registered based on normalized URI
        for r in self._resources.values():
            if r.normalized_uri == normalized_uri:
                raise ValueError(f"Resource {uri} already registered under the name {r.resource.name}")

        uri_params = set(TEMPLATE_PARAM_REGEX.findall(uri))
        func_params = set(inspect.signature(func).parameters.keys())

        if uri_params or func_params:
            # Resource Template
            if uri_params != func_params:
                raise ValueError(f"Mismatch between URI parameters {uri_params} and function parameters {func_params}")

            resource = ResourceTemplate(
                name=resource_name,
                title=kwargs.get("title", None),
                uriTemplate=uri,
                description=kwargs.get("description", details.doc),
                mimeType=kwargs.get("mime_type", None),
                annotations=kwargs.get("annotations", None),
                _meta=kwargs.get("meta", None),
            )

            resource_details = _ResourceDetails(
                resource=resource,
                func=func,
                normalized_uri=normalized_uri,
                uri_pattern=_uri_to_pattern(uri),
            )
        else:
            # Resource
            resource = Resource(
                name=resource_name,
                title=kwargs.get("title", None),
                uri=uri,  # type: ignore[arg-type]
                description=kwargs.get("description", details.doc),
                mimeType=kwargs.get("mime_type", None),
                annotations=kwargs.get("annotations", None),
                _meta=kwargs.get("meta", None),
            )

            resource_details = _ResourceDetails(
                resource=resource,
                func=func,
                normalized_uri=uri,
            )

        self._resources[resource_name] = resource_details

        return resource

    def remove(self, name: str) -> Resource | ResourceTemplate:
        """
        Remove a resource from the MCP resource manager.
        """
        if name not in self._resources:
            raise ValueError(f"Resource {name} not found")

        return self._resources.pop(name).resource

    def list(self) -> builtins.list[Resource]:
        return [r.resource for r in self._resources.values() if isinstance(r.resource, Resource)]

    async def _async_list(self) -> builtins.list[Resource]:
        return self.list()

    def list_templates(self) -> builtins.list[ResourceTemplate]:
        return [r.resource for r in self._resources.values() if isinstance(r.resource, ResourceTemplate)]

    async def _async_list_templates(self) -> builtins.list[ResourceTemplate]:
        return self.list_templates()

    async def read(self, uri: AnyUrl | str) -> Iterable[ReadResourceContents]:
        uri = str(uri)

        details, args = self._find_matching_details(uri)
        if details is None:
            raise ValueError(f"Resource {uri} not found")

        return await self._read_resource(details, args)

    async def read_by_name(self, name: str, args: dict[str, str] | None = None) -> Iterable[ReadResourceContents]:
        if name not in self._resources:
            raise ValueError(f"Resource {name} not found")

        details = self._resources[name]
        return await self._read_resource(details, args)

    async def _read_resource(
        self, details: _ResourceDetails, args: dict[str, str] | None
    ) -> Iterable[ReadResourceContents]:
        try:
            # TODO: Validate arguments - Something like this
            # if details.is_template and args:
            #     details.func.meta.arg_model.model_validate(args)

            result = details.func(**(args or {}))
            if inspect.iscoroutine(result):
                result = await result

            if isinstance(result, bytes) or isinstance(result, str):
                content = result
            else:
                content = pydantic_core.to_json(result, fallback=str, indent=2).decode()

            return [ReadResourceContents(content=content, mime_type=details.resource.mimeType)]
        except Exception as e:
            msg = f"Error reading resource {details.resource.name}: {e}"
            logger.exception(msg)
            raise ValueError(msg)

    def _find_matching_details(self, uri: str) -> tuple[_ResourceDetails | None, dict[str, str] | None]:
        # Find exact match
        for r in self._resources.values():
            if r.normalized_uri == uri and isinstance(r.resource, Resource):
                return r, None

        # Find with pattern matching
        for r in self._resources.values():
            if r.uri_pattern and (match := r.uri_pattern.match(uri)):
                return r, match.groupdict()

        return None, None


def _uri_to_pattern(uri: str) -> re.Pattern:
    """
    Convert a URI to a regular expression pattern.
    """
    # Replace {...} placeholders with a sentinel, escape the rest, then restore named groups
    SENT = "\x00"  # unlikely to appear in templates
    # Protect placeholders so re.escape doesn't touch the braces
    protected = re.sub(r"\{(\w+)\}", lambda m: f"{SENT}{m.group(1)}{SENT}", uri)
    escaped = re.escape(protected)
    # Turn sentinels back into named groups that match a single path segment
    pattern_str = re.sub(
        rf"{re.escape(SENT)}(\w+){re.escape(SENT)}",
        r"(?P<\1>[^/]+)",
        escaped,
    )
    return re.compile(pattern_str + r"$")  # equivalent to fullmatch
