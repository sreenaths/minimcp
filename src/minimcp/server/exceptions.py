class UnsupportedRPCMessageType(Exception):
    """Unsupported message type received by MiniMCP server"""

    pass


class ContextError(Exception):
    """Context error"""

    pass


class ParserError(Exception):
    """Parser error"""

    pass


class InvalidParamsError(Exception):
    """Invalid params error"""

    pass


class MethodNotFoundError(Exception):
    """Method not found error"""

    pass
