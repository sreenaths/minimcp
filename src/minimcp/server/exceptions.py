from mcp.types import ErrorData


class UnsupportedRPCMessageType(Exception):
    """Unsupported message type received by MiniMCP server"""

    pass


class ErrorWithData(Exception):
    """Error with data received by MiniMCP server"""

    data: ErrorData

    def __init__(self, data: ErrorData):
        self.data = data


class ContextError(Exception):
    """Context error"""

    pass
