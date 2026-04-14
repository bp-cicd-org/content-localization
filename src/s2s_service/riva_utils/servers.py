# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RIVA gRPC inference server adapters."""

import riva.client

from common.service import GRPCInferenceServer


class RivaASRServer(GRPCInferenceServer):
    """gRPC inference server adapter for RIVA ASR."""

    def __init__(
        self,
        host: str,
        port: int,
        health_check_service: str = "",
    ) -> None:
        super().__init__(
            host=host,
            port=port,
            health_check_service=health_check_service,
            stub_class=object,
        )

    @classmethod
    def from_string(cls, url: str) -> "RivaASRServer":
        host, port = url.split(":")
        return cls(host=host, port=int(port))

    def create_server(self, channel_options: list | None = None, channel_credentials=None) -> None:
        self.stub = riva.client.ASRService(auth=riva.client.Auth(uri=f"{self.host}:{self.port}"))

    def get_response_iterator(self, request_iterator):
        raise NotImplementedError("Use RIVA ASR client for response iteration.")


class RivaTTSServer(GRPCInferenceServer):
    """gRPC inference server adapter for RIVA TTS."""

    def __init__(
        self,
        host: str,
        port: int,
        health_check_service: str = "",
    ) -> None:
        super().__init__(
            host=host,
            port=port,
            health_check_service=health_check_service,
            stub_class=object,
        )

    @classmethod
    def from_string(cls, url: str) -> "RivaTTSServer":
        host, port = url.split(":")
        return cls(host=host, port=int(port))

    def create_server(self, channel_options: list | None = None, channel_credentials=None) -> None:
        self.stub = riva.client.SpeechSynthesisService(
            auth=riva.client.Auth(uri=f"{self.host}:{self.port}")
        )

    def get_response_iterator(self, request_iterator):
        raise NotImplementedError("Use RIVA TTS client for response iteration.")
