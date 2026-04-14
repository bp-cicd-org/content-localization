# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the common buffers module."""

import queue
import threading
import unittest

from common.buffers import Buffer


class TestBuffer(unittest.TestCase):
    """Unit tests covering Buffer producer/consumer behavior."""

    def test_single_queue_roundtrip(self) -> None:
        """Items put into a single queue are received in order."""
        buf: Buffer[int] = Buffer()
        items = list(range(5))

        for item in items:
            buf.put(item)

        received = [buf.get() for _ in items]

        self.assertEqual(received, items)
        self.assertTrue(buf.empty())

    def test_multi_queue_put_creates_copies(self) -> None:
        """Multi-queue put uses copy_func to duplicate items."""
        copy_calls: list[int] = []

        def copy_func(item: dict[str, int]) -> dict[str, int]:
            copy_calls.append(1)
            return {"value": item["value"]}

        buf: Buffer[dict[str, int]] = Buffer(num_queues=2, copy_func=copy_func)
        buf.put({"value": 7})

        first = buf.get(0)
        second = buf.get(1)

        self.assertEqual(first["value"], 7)
        self.assertEqual(second["value"], 7)
        self.assertIsNot(first, second)
        self.assertEqual(len(copy_calls), 1)

    def test_consumer_bounds_check(self) -> None:
        """Invalid consumer id raises IndexError."""
        buf: Buffer[int] = Buffer()
        with self.assertRaises(IndexError):
            buf.get(consumer_id=1, timeout=0.01)

    def test_full_and_qsize_for_bounded_queue(self) -> None:
        """Bounded queue reports full and correct size."""
        buf: Buffer[str] = Buffer(max_size=1)
        buf.put("a")

        self.assertTrue(buf.full())
        self.assertEqual(buf.qsize(), 1)

        with self.assertRaises(queue.Full):
            buf.put("b", timeout=0.01)

    def test_done_and_is_exhausted(self) -> None:
        """is_exhausted reflects done + empty state."""
        buf: Buffer[int] = Buffer()
        self.assertFalse(buf.is_exhausted())
        buf.done = True
        self.assertTrue(buf.is_exhausted())

    def test_multithreaded_producer_consumer(self) -> None:
        """Concurrent producers/consumers preserve ordering and duplication."""
        buf: Buffer[dict[str, int]] = Buffer(num_queues=2)
        total_items = 200
        produced = [{"value": i} for i in range(total_items)]
        consumed: dict[int, list[dict[str, int]]] = {0: [], 1: []}

        def producer() -> None:
            for item in produced:
                buf.put(item)
            buf.done = True

        def consumer(consumer_id: int) -> None:
            while True:
                try:
                    value = buf.get(consumer_id, timeout=0.05)
                    consumed[consumer_id].append(value)
                except queue.Empty:
                    if buf.is_exhausted(consumer_id):
                        break

        threads = [
            threading.Thread(target=producer),
            threading.Thread(target=consumer, args=(0,)),
            threading.Thread(target=consumer, args=(1,)),
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(consumed[0], produced)
        self.assertEqual(consumed[1], produced)
        for idx in range(total_items):
            self.assertIsNot(consumed[0][idx], consumed[1][idx])
        self.assertTrue(buf.empty(0))
        self.assertTrue(buf.empty(1))
