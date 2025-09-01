import logging
import sys
from collections.abc import Awaitable, Callable
from io import TextIOWrapper

import anyio

from minimcp.server.types import Message, NoMessage, Send

logger = logging.getLogger(__name__)


async def stdio_transport(handler: Callable[[Message, Send], Awaitable[Message | NoMessage]]):
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

    async def write_msg(response: Message | NoMessage):
        if not isinstance(response, NoMessage):
            logger.info("Writing response message to stdio: %s", response)
            await stdout.write(response + "\n")
            await stdout.flush()

    async def handle_message(line: str):
        line = line.rstrip("\n").strip()

        if line:
            logger.info("Handling incoming message: %s", line)

            response = await handler(line, write_msg)

            await write_msg(response)

    async with anyio.create_task_group() as tg:
        async for line in stdin:
            tg.start_soon(handle_message, line)
