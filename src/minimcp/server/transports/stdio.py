import json
import logging
import sys
from io import TextIOWrapper

import anyio
import mcp.types as types

from minimcp.server import json_rpc
from minimcp.server.responder import Responder
from minimcp.server.transports.types import TransportRequestHandler
from minimcp.server.types import Message
from minimcp.server.utils import to_dict

logger = logging.getLogger(__name__)


async def stdio_transport(handler: TransportRequestHandler):
    """
    stdio_transport makes it easy to use MiniMCP over stdio.
    - The anyio.wrap_file implementation naturally apply backpressure.
    - Concurrency management is expected to be enforced by MiniMCP.

    Args:
        handler: A function that will be called for each incoming message. It will be called
            with the message and a send function to write responses. Message returned by the function
            will be send back to the client. If the function returns None, nothing would be sent back.

    Returns:
        None
    """

    stdin = anyio.wrap_file(TextIOWrapper(sys.stdin.buffer, encoding="utf-8"))
    stdout = anyio.wrap_file(TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True))

    async def write_msg(response: Message | None):
        if response is not None:
            logger.info("Writing response message to stdio: %s", response)
            await stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            await stdout.flush()

    async def handle_message(line: str):
        line = line.rstrip("\n")
        logger.info("Handling incoming message: %s", line)

        try:
            message_dict = json.loads(line)
        except json.JSONDecodeError as e:
            response = to_dict(json_rpc.build_error_message(types.PARSE_ERROR, {}, e))
        else:
            responder = Responder(message_dict, write_msg)
            response = await handler(message_dict, responder)

        await write_msg(response)

    async with anyio.create_task_group() as tg:
        async for line in stdin:
            tg.start_soon(handle_message, line)
