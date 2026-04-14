# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client abstractions for controller services.

Example::

    from common.buffers import Buffer
    from common.clients import Client
    from common.service import GRPCInferenceServer


    # Upstream code populates the request buffer; the client will set done=True when finished
    request_buffer: Buffer[int] = Buffer()
    output_buffer: Buffer[str] = Buffer()
    server = GRPCInferenceServer(...)  # concrete server instance

    client = Client(server)
    client(
        request_iterator=request_buffer,
        output_buffer=output_buffer,
        context=...,
        request_id="req-1",
    )
"""

import traceback
from abc import ABC
from abc import abstractmethod
from collections.abc import Iterator
from typing import Any
from typing import Generic
from typing import TypeVar

import grpc

from base_utils import logger
from common.buffers import Buffer
from common.service import GRPCInferenceServer

ReqT = TypeVar("ReqT")
RespT = TypeVar("RespT")


class Client(ABC, Generic[ReqT, RespT]):
    """Abstract client that manages requests and an output buffer.

    Each client maintains its own thread-safe output buffer that was created by
    the caller. The request generation strategy is also provided by a callable
    so that one client can consume from another client's output buffer or from
    other iterators without coupling that logic to the client itself. The
    request iterator must be thread-safe. If the request iterator is an output
    buffer of another client, it will automatically be thread-safe.
    """

    def __init__(
        self,
        server: GRPCInferenceServer,
    ) -> None:
        """Initialize a client that streams server responses into an output buffer.

        Args:
            server: GRPCInferenceServer used to create the response iterator.
        """
        self.server = server
        logger.debug(f"Client initialized with server: {server}")

    def is_healthy(self) -> bool:
        """Check if the inference server is healthy.

        Returns:
            bool: True if the server is healthy, False otherwise.

        Raises:
            ConnectionError: If the server is not healthy.
        """
        logger.debug(f"Checking health of server: {self.server}")
        try:
            self.server.is_healthy()
        except ConnectionError as e:
            logger.error(f"Server at {self.server} is not healthy: {e}\n" + traceback.format_exc())
            raise e
        logger.debug(f"Server {self.server} is healthy")
        return True

    def __call__(
        self,
        request_iterator: Iterator[ReqT],
        output_buffer: Buffer[RespT],
        context: grpc.ServicerContext,
        request_id: str,
        *args: tuple[object, ...],
        **kwargs: Any,
    ) -> None:
        """Run the client by streaming responses into the output buffer.

        Args:
            request_iterator: Inbound requests (buffer, generator, or gRPC iterator).
            output_buffer: Buffer receiving server responses. Client doesn't own buffer.
            context: gRPC servicer context.
            request_id: Correlation identifier.
            *args: Additional positional arguments forwarded to ``_impl``.
            **kwargs: Additional keyword arguments forwarded to ``_impl``.

        Returns:
            None. ``output_buffer`` is populated in place.
        """
        logger.debug(f"Client __call__ invoked: request_id={request_id}")
        if not self.is_healthy():
            logger.error(f"Server at {self.server} is not healthy")
            context.abort(grpc.StatusCode.INTERNAL, f"Server at {self.server} is not healthy")
        try:
            logger.debug(f"Starting _impl for request_id={request_id}")
            self._impl(
                request_iterator=request_iterator,
                output_buffer=output_buffer,
                context=context,
                request_id=request_id,
                *args,
                **kwargs,
            )
            logger.debug(f"Completed _impl for request_id={request_id}")
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Error running client: {e}\n{tb}")
            context.abort(grpc.StatusCode.INTERNAL, f"{type(e).__name__}: {e}\n{tb}")
        finally:
            output_buffer.done = True
            logger.debug(f"Client __call__ finished: request_id={request_id}")

    @abstractmethod
    def _impl(
        self,
        request_iterator: Iterator[ReqT],
        output_buffer: Buffer[RespT],
        context: grpc.ServicerContext,
        request_id: str,
        *args: tuple[object, ...],
        **kwargs: Any,
    ) -> None:
        """Implement client-specific logic to produce responses.

        Args:
            request_iterator: Inbound requests to consume.
            output_buffer: Destination buffer for responses.
            context: gRPC servicer context.
            request_id: Correlation identifier.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            None. Subclasses must place results into ``output_buffer`` and sets buffer ``done``.
        """
        raise NotImplementedError
