# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GRPC inference server abstractions."""

from abc import ABC
from abc import abstractmethod
from collections.abc import Iterator
from typing import Any

import grpc

from base_utils import logger
from common.servers import GRPCServer


class InferenceServer(ABC):
    """Abstract interface for inference server adapters."""

    @abstractmethod
    def create_server(self, *args: Any, **kwargs: Any) -> None:
        """Initialize any client/stub resources needed for inference."""

    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if the inference server is healthy."""


class GRPCInferenceServer(GRPCServer, InferenceServer, ABC):
    """GRPC Server abstractions for NIMs and other model services."""

    def __init__(
        self,
        host: str,
        port: int,
        stub_class: Any,
        health_check_service: str = "",
        channel_credentials: grpc.ChannelCredentials | None = None,
    ) -> None:
        """Initialize the GRPCNIMServer.

        Args:
            host (str): The host to listen on.
            port (int): The port to listen on.
            health_check_service (str): The health check service to use.
            stub_class (Any): The stub class to use.
        """
        super().__init__(
            host=host,
            port=port,
            health_check_service=health_check_service,
            channel_credentials=channel_credentials,
        )

        self.stub_class = stub_class
        self.stub = None
        self.channel = None
        logger.debug(
            f"GRPCInferenceServer initialized: host={host}, port={port}, "
            f"stub_class={stub_class.__name__}, health_check_service={health_check_service}"
        )

    def create_server(
        self,
        channel_options: list | None = None,
        channel_credentials: grpc.ChannelCredentials | None = None,
    ) -> None:
        """Create the server.

        Args:
            channel_options (list): The channel options.
            channel_credentials (grpc.ChannelCredentials | None): Optional credentials for
                secure channels. Defaults to None.
        """
        logger.debug(f"Creating server channel to {self.host}:{self.port}")
        if channel_options is None:
            channel_options = []
        credentials = channel_credentials or self.channel_credentials
        if credentials is not None:
            self.channel = grpc.secure_channel(
                target=f"{self.host}:{self.port}",
                credentials=credentials,
                options=channel_options,
            )
        else:
            self.channel = grpc.insecure_channel(
                target=f"{self.host}:{self.port}", options=channel_options
            )
        self.stub = self.stub_class(self.channel)
        logger.debug(f"Server channel created with stub: {self.stub_class.__name__}")

    @abstractmethod
    def get_response_iterator(self, request_iterator: Iterator[Any]) -> Iterator[Any]:
        """Get a response iterator from the NIM.

        Args:
            request_iterator (Iterator[Any]): The request iterator.
        """
