import logging
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Generic, TypeVar

from mcp.types import JSONRPCMessage

from minimcp.server.exceptions import ContextError
from minimcp.server.responder import Responder

logger = logging.getLogger(__name__)


ScopeT = TypeVar("ScopeT", bound=object)


@dataclass
class Context(Generic[ScopeT]):
    message: JSONRPCMessage
    scope: ScopeT | None = None
    responder: Responder | None = None


class ContextManager(Generic[ScopeT]):
    _ctx: ContextVar[Context[ScopeT]] = ContextVar("ctx")

    @contextmanager
    def active(self, message: JSONRPCMessage, scope: ScopeT | None = None, responder: Responder | None = None):
        # Build context
        context = Context(message=message, scope=scope, responder=responder)

        # Set context
        token: Token = self._ctx.set(context)
        try:
            yield
        finally:
            # Clear context
            self._ctx.reset(token)

    def _get(self) -> Context[ScopeT]:
        try:
            return self._ctx.get()
        except LookupError:
            err = ContextError("No Context: Called get outside of an active context")
            logger.error(err)
            raise err

    def get_message(self) -> JSONRPCMessage:
        return self._get().message

    def get_scope(self) -> ScopeT:
        scope = self._get().scope
        if scope is None:
            raise ContextError("Scope is not available in current context")
        return scope

    def get_responder(self) -> Responder:
        responder = self._get().responder
        if responder is None:
            raise ContextError("Responder is not available in current context")
        return responder
