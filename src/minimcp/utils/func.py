import inspect
from dataclasses import dataclass

from mcp.server.fastmcp.utilities.func_metadata import FuncMetadata, func_metadata
from mcp.types import AnyFunction


@dataclass
class FuncDetails:
    name: str | None
    doc: str | None
    meta: FuncMetadata


def extract_func_details(func: AnyFunction) -> FuncDetails:
    """
    Extracts the details of a function.
    """

    validate_func(func)

    func_name = getattr(func, "__name__", None)
    func_doc = inspect.getdoc(func)

    # -- Extract metadata ---
    func_meta = func_metadata(func)

    return FuncDetails(name=func_name, doc=func_doc, meta=func_meta)


def validate_func(func: AnyFunction) -> None:
    """
    Validates a function's usability with MCP.
    """

    if isinstance(func, classmethod):
        raise ValueError("Function cannot be a classmethod")

    if isinstance(func, staticmethod):
        raise ValueError("Function cannot be a staticmethod")

    if getattr(func, "__isabstractmethod__", False):
        raise ValueError("Function cannot be an abstract method")

    if not inspect.isroutine(func):
        raise ValueError("Value passed is not a function or method")

    sig = inspect.signature(func)
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            raise ValueError("Functions with *args are not supported")
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            raise ValueError("Functions with **kwargs are not supported")


def validate_func_name(name: str | None) -> str:
    if not name:
        raise ValueError("Name is not available")

    if name == "<lambda>":
        raise ValueError("Name must be provided for lambda functions")

    return name
