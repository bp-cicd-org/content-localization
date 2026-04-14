# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP and gRPC server health-check abstractions."""

import os

import grpc
import requests
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc

from base_utils import logger

# Shared timeout for both HTTP and gRPC health checks.
# Prevents indefinite hangs when a downstream service is unreachable.
HEALTH_CHECK_TIMEOUT: float = float(os.environ.get("HEALTH_CHECK_TIMEOUT", "5.0"))


class HTTPServer:
    """A common abstraction for HTTP Servers."""

    def __init__(
        self,
        host: str,
        port: int,
        health_url: str = "/v2/health/ready",
    ) -> None:
        """Initialize the HTTPServer.

        Args:
            host (str): The host to listen on.
            port (int): The port to listen on.
            health_url (str): The health URL to check.
                Defaults to ``"/v2/health/ready"``.

        Examples:
            >>> server = HTTPServer(host="localhost", port=8080)
            >>> str(server)
            'localhost:8080'
        """
        self.host = host
        self.port = port
        self.health_url = self._construct_health_url(
            health_url=health_url,
        )

    def _construct_health_url(self, health_url: str) -> str:
        """Construct the full health URL.

        Args:
            health_url (str): The health URL path to check.

        Returns:
            str: The full health URL including protocol, host, and port.

        Examples:
            >>> server = HTTPServer(host="localhost", port=8080)
            >>> server._construct_health_url("/v2/health/ready")
            'http://localhost:8080/v2/health/ready'
        """
        return f"http://{self.host}:{self.port}{health_url}"

    @classmethod
    def from_string(cls, url: str) -> "HTTPServer":
        """Create a HTTPServer from a string.

        Args:
            url (str): The URL string in the format ``'host:port'``.

        Returns:
            HTTPServer: A new HTTPServer instance.

        Examples:
            >>> server = HTTPServer.from_string("localhost:8080")
            >>> server.host
            'localhost'
        """
        return cls(url.split(":")[0], int(url.split(":")[1]))

    def is_healthy(self) -> int:
        """Check if the server is healthy.

        Returns:
            int: The status code of the health check.

        Raises:
            ConnectionError: If the server is not healthy.

        Examples:
            >>> server = HTTPServer(host="localhost", port=8080)
            >>> server.is_healthy()  # doctest: +SKIP
            200
        """
        try:
            logger.debug(f"Checking health: {self.health_url}")
            response = requests.get(url=self.health_url, timeout=HEALTH_CHECK_TIMEOUT)
            response.raise_for_status()
            logger.debug(f"Health check passed for {self.health_url}")
        except requests.RequestException as e:
            logger.error(f"Health check failed for {self.health_url}: {e!s}")
            raise ConnectionError(f"Service at {self.health_url} is not healthy: {e!s}") from e
        return response.status_code

    def __call__(self) -> str:
        """Return the server as a string after performing health check.

        Returns:
            str: The server address in ``'host:port'`` format.

        Raises:
            ConnectionError: If the health check fails.

        Examples:
            >>> server = HTTPServer(host="localhost", port=8080)
            >>> server()  # doctest: +SKIP
            'localhost:8080'
        """
        _ = self.is_healthy()
        return f"{self.host}:{self.port}"

    def __str__(self) -> str:
        """Return the server as a string.

        Returns:
            str: The server address in ``'host:port'`` format.

        Examples:
            >>> str(HTTPServer(host="localhost", port=8080))
            'localhost:8080'
        """
        return f"{self.host}:{self.port}"


class GRPCServer:
    """A common abstraction for gRPC Servers."""

    def __init__(
        self,
        host: str,
        port: int,
        health_check_service: str = "",
        channel_credentials: grpc.ChannelCredentials | None = None,
    ) -> None:
        """Initialize the GRPCServer.

        Args:
            host (str): The host to connect to.
            port (int): The port to connect to.
            health_check_service (str): The gRPC health check service
                name. Defaults to ``""``.
            channel_credentials (grpc.ChannelCredentials | None):
                Optional credentials for secure channels.
                Defaults to ``None``.

        Examples:
            >>> server = GRPCServer(host="localhost", port=50051)
            >>> str(server)
            'localhost:50051'
        """
        self.host = host
        self.port = port
        self.health_check_service = health_check_service
        self.channel_credentials = channel_credentials

    @classmethod
    def from_string(cls, url: str) -> "GRPCServer":
        """Create a GRPCServer from a string.

        Args:
            url (str): The URL string in the format ``'host:port'``.

        Returns:
            GRPCServer: A new GRPCServer instance.

        Examples:
            >>> server = GRPCServer.from_string("localhost:50051")
            >>> server.port
            50051
        """
        return cls(url.split(":")[0], int(url.split(":")[1]))

    def is_healthy(self) -> bool:
        """Check if the gRPC server is healthy.

        Uses the standard gRPC health checking protocol.

        Returns:
            bool: ``True`` if the service is healthy.

        Raises:
            ConnectionError: If the server is not healthy.

        Examples:
            >>> server = GRPCServer(host="localhost", port=50051)
            >>> server.is_healthy()  # doctest: +SKIP
            True
        """
        address = f"{self.host}:{self.port}"
        if self.channel_credentials is not None:
            channel = grpc.secure_channel(address, self.channel_credentials)
        else:
            channel = grpc.insecure_channel(address)
        stub = health_pb2_grpc.HealthStub(channel)
        try:
            logger.debug(f"Checking gRPC health: {address} (service='{self.health_check_service}')")
            response = stub.Check(
                health_pb2.HealthCheckRequest(
                    service=self.health_check_service,
                ),
                timeout=HEALTH_CHECK_TIMEOUT,
            )
            if response.status == health_pb2.HealthCheckResponse.SERVING:
                logger.debug(f"gRPC health check passed for {address}")
                return True
            else:
                logger.error(f"gRPC health check failed for {address}: status={response.status}")
                raise ConnectionError(
                    f"gRPC service at {address} not healthy: status={response.status}"
                )
        except grpc.RpcError as e:
            logger.error(f"gRPC health check failed for {address}: {e!s}")
            raise ConnectionError(f"gRPC service at {address} health check failed: {e!s}") from e

    def __call__(self) -> str:
        """Return the server as a string after performing health check.

        Returns:
            str: The server address in ``'host:port'`` format.

        Raises:
            ConnectionError: If the health check fails.

        Examples:
            >>> server = GRPCServer(host="localhost", port=50051)
            >>> server()  # doctest: +SKIP
            'localhost:50051'
        """
        _ = self.is_healthy()
        return f"{self.host}:{self.port}"

    def __str__(self) -> str:
        """Return the server as a string.

        Returns:
            str: The server address in ``'host:port'`` format.

        Examples:
            >>> str(GRPCServer(host="localhost", port=50051))
            'localhost:50051'
        """
        return f"{self.host}:{self.port}"
