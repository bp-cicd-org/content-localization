# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIMs and tools for interacting with NIMs."""

import traceback
from collections.abc import Iterator
from typing import Any

import grpc
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    DetectActiveSpeakerRequest,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    DetectActiveSpeakerResponse,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2_grpc import (
    ActiveSpeakerDetectionServiceStub,
)
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncRequest
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncResponse
from nvidia.ai4m.lipsync.v1.lipsync_pb2_grpc import LipSyncServiceStub
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechRequest
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechResponse
from nvidia.ai4m.s2s.v1.s2s_pb2_grpc import SpeechToSpeechStub

from base_utils import logger
from common.buffers import Buffer
from common.clients import Client
from common.service import GRPCInferenceServer


class SpeechToSpeechServer(GRPCInferenceServer):
    """Speech to Speech NIM Server."""

    def __init__(
        self,
        host: str,
        port: int,
        health_check_service: str = "",
        channel_credentials: grpc.ChannelCredentials | None = None,
    ) -> None:
        """Initialize the SpeechToSpeechServer.

        Args:
            host (str): The host to listen on.
            port (int): The port to listen on.
            health_check_service (str): The health check service to use.
        """
        super().__init__(
            host=host,
            port=port,
            health_check_service=health_check_service,
            stub_class=SpeechToSpeechStub,
            channel_credentials=channel_credentials,
        )

    def get_response_iterator(
        self, request_iterator: Iterator[SpeechToSpeechRequest]
    ) -> Iterator[SpeechToSpeechResponse]:
        """Get a response iterator from the SpeechToSpeechServer.

        Args:
            request_iterator (Iterator[Any]): The request iterator.

        """
        return self.stub.StreamSpeechToSpeech(request_iterator)


class ActiveSpeakerDetectionServer(GRPCInferenceServer):
    """Active Speaker Detection NIM Server."""

    def __init__(
        self,
        host: str,
        port: int,
        health_check_service: str = "",
        channel_credentials: grpc.ChannelCredentials | None = None,
    ) -> None:
        """Initialize the SpeakerDetectionServer.

        Args:
            host (str): The host to listen on.
            port (int): The port to listen on.
            health_check_service (str): The health check service to use.
        """
        super().__init__(
            host=host,
            port=port,
            health_check_service=health_check_service,
            stub_class=ActiveSpeakerDetectionServiceStub,
            channel_credentials=channel_credentials,
        )

    def get_response_iterator(
        self, request_iterator: Iterator[DetectActiveSpeakerRequest]
    ) -> Iterator[DetectActiveSpeakerResponse]:
        """Get a response iterator from the ActiveSpeakerDetectionServer.

        Args:
            request_iterator (Iterator[Any]): The request iterator.
        """
        return self.stub.DetectActiveSpeaker(request_iterator)


class LipsyncServer(GRPCInferenceServer):
    """Lipsync NIM Server."""

    def __init__(
        self,
        host: str,
        port: int,
        health_check_service: str = "",
        channel_credentials: grpc.ChannelCredentials | None = None,
    ) -> None:
        """Initialize the LipsyncServer.

        Args:
            host (str): The host to listen on.
            port (int): The port to listen on.
            health_check_service (str): The health check service to use.
        """
        super().__init__(
            host=host,
            port=port,
            health_check_service=health_check_service,
            stub_class=LipSyncServiceStub,
            channel_credentials=channel_credentials,
        )

    def get_response_iterator(
        self, request_iterator: Iterator[LipsyncRequest]
    ) -> Iterator[LipsyncResponse]:
        """Get a response iterator from the LipsyncServer.

        Args:
            request_iterator (Iterator[Any]): The request iterator.
        """
        return self.stub.Lipsync(request_iterator)


class SpeechToSpeechClient(Client[SpeechToSpeechRequest, SpeechToSpeechResponse]):
    """Client that streams non-keepalive S2S responses into an output buffer."""

    def _impl(
        self,
        request_iterator: Iterator[SpeechToSpeechRequest],
        output_buffer: Buffer[SpeechToSpeechResponse],
        context: grpc.ServicerContext,
        request_id: str,
        *args: tuple[object, ...],
        **kwargs: Any,
    ) -> None:
        logger.debug(f"Starting SpeechToSpeech client for request_id={request_id}")
        if self.server.stub is None:
            self.server.create_server()
        response_iterator = self.server.get_response_iterator(request_iterator=request_iterator)
        try:
            for response in response_iterator:
                if response.HasField("keepalive"):
                    logger.debug("SpeechToSpeech client: skipping keep-alive response")
                    continue
                output_buffer.put(response)
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Error in SpeechToSpeech client: {e}\n{tb}")
            context.abort(grpc.StatusCode.INTERNAL, f"{type(e).__name__}: {e}\n{tb}")


class ActiveSpeakerDetectionClient(Client[DetectActiveSpeakerRequest, DetectActiveSpeakerResponse]):
    """Client that streams ASD detection results into an output buffer."""

    def _impl(
        self,
        request_iterator: Iterator[DetectActiveSpeakerRequest],
        output_buffer: Buffer[DetectActiveSpeakerResponse],
        context: grpc.ServicerContext,
        request_id: str,
        *args: tuple[object, ...],
        **kwargs: Any,
    ) -> None:
        logger.debug(f"Starting ActiveSpeakerDetection client for request_id={request_id}")
        if self.server.stub is None:
            self.server.create_server()
        response_iterator = self.server.get_response_iterator(request_iterator=request_iterator)
        result_count = 0
        keepalive_count = 0
        config_count = 0
        try:
            for response in response_iterator:
                if response.HasField("keepalive"):
                    keepalive_count += 1
                    continue
                if response.HasField("config"):
                    config_count += 1
                    continue
                result_count += 1
                output_buffer.put(response)
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Error in ActiveSpeakerDetection client: {e}\n{tb}")
            context.abort(grpc.StatusCode.INTERNAL, f"{type(e).__name__}: {e}\n{tb}")
        logger.info(
            f"ASD client finished: results={result_count},"
            f" keepalives={keepalive_count}, configs={config_count}"
        )


class LipsyncClient(Client[LipsyncRequest, LipsyncResponse]):
    """Client that streams non-keepalive LipSync responses into an output buffer."""

    def _impl(
        self,
        request_iterator: Iterator[LipsyncRequest],
        output_buffer: Buffer[LipsyncResponse],
        context: grpc.ServicerContext,
        request_id: str,
        *args: tuple[object, ...],
        **kwargs: Any,
    ) -> None:
        logger.debug(f"Starting LipSync client for request_id={request_id}")
        if self.server.stub is None:
            self.server.create_server()
        response_iterator = self.server.get_response_iterator(request_iterator=request_iterator)
        try:
            for response in response_iterator:
                if response.HasField("keepalive"):
                    logger.debug("LipSync client: skipping keep-alive response")
                    continue
                output_buffer.put(response)
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Error in Lipsync client: {e}\n{tb}")
            context.abort(grpc.StatusCode.INTERNAL, f"{type(e).__name__}: {e}\n{tb}")
