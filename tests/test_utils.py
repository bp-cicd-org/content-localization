# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common shared utilities"""

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import grpc as real_grpc
import requests as real_requests

from common.servers import GRPCServer
from common.servers import HTTPServer


class TestHTTPServer(unittest.TestCase):
    def test_from_string(self):
        server = HTTPServer.from_string("localhost:1234")
        self.assertEqual(server.host, "localhost")
        self.assertEqual(server.port, 1234)
        self.assertIn("localhost:1234", server.health_url)
        self.assertEqual(server.health_url, "http://localhost:1234/v2/health/ready")

    @patch("common.servers.requests.get")
    def test_is_healthy_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        server = HTTPServer("localhost", 1234)
        self.assertEqual(server.is_healthy(), 200)

    @patch("common.servers.requests.get")
    def test_is_healthy_failure(self, mock_get):
        mock_get.side_effect = real_requests.RequestException("fail")
        server = HTTPServer("localhost", 1234)
        with self.assertRaises(ConnectionError):
            server.is_healthy()

    @patch("common.servers.requests.get")
    def test_call_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        server = HTTPServer("localhost", 1234)
        # __call__ should return host:port and trigger health check
        self.assertEqual(server(), "localhost:1234")

    @patch("common.servers.requests.get")
    def test_call_failure(self, mock_get):
        mock_get.side_effect = real_requests.RequestException("fail")
        server = HTTPServer("localhost", 1234)
        with self.assertRaises(ConnectionError):
            server()


class TestGRPCServer(unittest.TestCase):
    @patch("common.servers.grpc.insecure_channel")
    @patch("common.servers.health_pb2_grpc.HealthStub")
    def test_is_healthy_success(self, mock_stub, mock_channel):
        mock_instance = MagicMock()
        mock_instance.Check.return_value.status = 1  # SERVING
        mock_stub.return_value = mock_instance
        server = GRPCServer("localhost", 50051)
        self.assertTrue(server.is_healthy())

    @patch("common.servers.grpc.insecure_channel")
    @patch("common.servers.health_pb2_grpc.HealthStub")
    def test_is_healthy_failure(self, mock_stub, mock_channel):
        mock_instance = MagicMock()
        mock_instance.Check.return_value.status = 2  # NOT_SERVING
        mock_stub.return_value = mock_instance
        server = GRPCServer("localhost", 50051)
        with self.assertRaises(ConnectionError):
            server.is_healthy()

    @patch("common.servers.grpc.insecure_channel")
    @patch("common.servers.health_pb2_grpc.HealthStub")
    def test_is_healthy_grpc_error(self, mock_stub, mock_channel):
        mock_instance = MagicMock()
        mock_instance.Check.side_effect = real_grpc.RpcError("fail")
        mock_stub.return_value = mock_instance
        server = GRPCServer("localhost", 50051)
        with self.assertRaises(ConnectionError):
            server.is_healthy()

    def test_from_string(self):
        server = GRPCServer.from_string("localhost:50051")
        self.assertEqual(server.host, "localhost")
        self.assertEqual(server.port, 50051)

    @patch("common.servers.grpc.insecure_channel")
    @patch("common.servers.health_pb2_grpc.HealthStub")
    def test_call_success(self, mock_stub, mock_channel):
        mock_instance = MagicMock()
        mock_instance.Check.return_value.status = 1  # SERVING
        mock_stub.return_value = mock_instance
        server = GRPCServer("localhost", 50051)
        # __call__ should return host:port and trigger health check
        self.assertEqual(server(), "localhost:50051")

    @patch("common.servers.grpc.insecure_channel")
    @patch("common.servers.health_pb2_grpc.HealthStub")
    def test_call_failure(self, mock_stub, mock_channel):
        mock_instance = MagicMock()
        mock_instance.Check.side_effect = real_grpc.RpcError("fail")
        mock_stub.return_value = mock_instance
        server = GRPCServer("localhost", 50051)
        with self.assertRaises(ConnectionError):
            server()


if __name__ == "__main__":
    unittest.main()
