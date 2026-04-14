# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the common client abstractions."""

import unittest
from unittest.mock import MagicMock

import grpc

from common.buffers import Buffer
from common.buffers import RequestIteratorFromBuffer
from common.clients import Client


class _DummyClient(Client[int, int]):
    """Concrete client for testing that echoes requests into the output buffer."""

    def _impl(
        self,
        request_iterator,
        output_buffer,
        context,
        request_id,
        *args,
        **kwargs,
    ) -> None:
        for item in request_iterator:
            output_buffer.put(item)


class _ErrorClient(Client[int, int]):
    """Concrete client for testing error handling."""

    def _impl(
        self,
        request_iterator,
        output_buffer,
        context,
        request_id,
        *args,
        **kwargs,
    ) -> None:  # pragma: no cover - exercised indirectly
        raise RuntimeError("boom")


class TestClient(unittest.TestCase):
    """Unit tests for the Client base class."""

    def test_call_streams_responses_into_buffer_and_marks_done(self) -> None:
        """__call__ drains requests through _impl into output buffer and sets done."""
        request_iter = iter([1, 2, 3])
        output_buffer: Buffer[int] = Buffer()
        server = MagicMock()
        server.is_healthy.return_value = True
        context = MagicMock(spec=grpc.ServicerContext)

        client = _DummyClient(server)
        client(
            request_iterator=request_iter,
            output_buffer=output_buffer,
            context=context,
            request_id="req-1",
        )

        self.assertTrue(output_buffer.done)
        self.assertEqual(
            list(RequestIteratorFromBuffer(output_buffer, poll_timeout=0.01)),
            [1, 2, 3],
        )
        server.is_healthy.assert_called_once()
        context.abort.assert_not_called()

    def test_call_handles_impl_exception_and_marks_done(self) -> None:
        """Exceptions propagate through context.abort and still mark buffer done."""
        request_iter = iter([1])
        output_buffer: Buffer[int] = Buffer()
        server = MagicMock()
        server.is_healthy.return_value = True
        context = MagicMock(spec=grpc.ServicerContext)

        client = _ErrorClient(server)
        client(
            request_iterator=request_iter,
            output_buffer=output_buffer,
            context=context,
            request_id="req-err",
        )

        self.assertTrue(output_buffer.done)
        context.abort.assert_called_once()

    def test_buffer_request_generator_yields_until_exhausted(self) -> None:
        """RequestIteratorFromBuffer drains a Buffer and stops when done."""
        buffer: Buffer[int] = Buffer()
        buffer.put(1)
        buffer.put(2)
        buffer.done = True

        iterator = RequestIteratorFromBuffer(buffer, poll_timeout=0.01)

        self.assertEqual(list(iterator), [1, 2])

    def test_request_iterator_returns_empty_when_done_and_no_items(self) -> None:
        """Returns immediately when buffer is done with no items."""
        buffer: Buffer[int] = Buffer()
        buffer.done = True

        iterator = RequestIteratorFromBuffer(buffer, poll_timeout=0.01)

        self.assertEqual(list(iterator), [])

    def test_request_iterator_respects_consumer_id(self) -> None:
        """Reads from the specified consumer queue in multi-queue buffers."""
        buffer: Buffer[int] = Buffer(num_queues=2)
        buffer.put(7)
        buffer.done = True

        consumer0_iter = RequestIteratorFromBuffer(buffer, consumer_id=0, poll_timeout=0.01)
        consumer1_iter = RequestIteratorFromBuffer(buffer, consumer_id=1, poll_timeout=0.01)

        self.assertEqual(list(consumer0_iter), [7])
        self.assertEqual(list(consumer1_iter), [7])
