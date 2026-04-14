# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller Service for Content Localization.

This module implements the main Controller Service that orchestrates the content
localization pipeline. It coordinates between client applications and downstream
AI services to provide end-to-end content localization capabilities.

Architecture Overview
=====================

The Controller Service acts as the central orchestrator in a microservices
architecture, coordinating three main AI services:

1. **Speech-to-Speech (S2S) Service**: Handles audio translation and synthesis
2. **Active Speaker Detection (ASD) Service**: Identifies speaking faces in video
3. **LipSync Service**: Synchronizes lip movements with translated audio

Service Components
==================

ControllerService
-----------------
The main service class that implements the content localization logic. It manages:
- Deserializer-based request distribution
- Multi-threaded client processing
- Response streaming
- Error handling and recovery

ControllerServiceServicer
-------------------------
gRPC servicer implementation that handles client communication:
- Request stream processing
- Response streaming
- Context management
- Error propagation

Multi-Threaded Architecture
===========================

The Controller Service uses a multi-threaded pipeline:

Deserializer + Buffer + Client Threads
---------------------------------------
- ContentLocalizationDeserializer consumes gRPC request stream on background thread
- Distributes requests to typed buffers (audio_buffer, video_buffer, etc.)
- Multiple client threads process requests concurrently:
  - S2S Client Thread: Processes audio translation (skipped in bypass-S2S mode)
  - ASD Client Thread: Detects active speakers (skipped when bypass_asd)
  - LipSync Client Thread: Generates lip-synced video
- Supports **bypass-S2S mode**: when ``controller_config.bypass_s2s`` is True,
  S2S is skipped and ``translated_audio_buffer`` feeds LipSync directly
- Main thread yields ContentLocalizationResponse to client

Request Flow
============

.. code-block:: text

    Client Request Stream
      |
    ContentLocalizationDeserializer (background thread)
      |-> controller_config_buffer -> determines bypass_s2s mode
      |-> audio_buffer (queue 0) -> S2S Client Thread -> s2s_output_buffer  [skip if bypass]
      |-> audio_buffer (queue 1) -> ASD Client Thread (audio input)
      |-> video_buffer (queue 0) -> ASD Client Thread (video input) -> asd_output_buffer
      |-> video_buffer (queue 1) -> LipSync Client Thread -> lipsync_output_buffer
      |-> diarization_buffer     -> ASD Client Thread (diarization input)
      |-> background_audio_buffer -> LipSync Client Thread (optional)
      |-> translated_audio_buffer -> LipSync Client Thread (bypass S2S only)
      |
    Main Thread yields ContentLocalizationResponse

Key Features
============

Multi-threading Support
-----------------------
- Deserializer thread for non-blocking gRPC stream consumption
- Concurrent client threads for S2S, ASD, and LipSync services
- Thread-safe buffer-based communication
- Proper thread lifecycle management and cleanup

Error Handling
--------------
- Comprehensive error handling for gRPC communication
- Graceful degradation when services are unavailable
- Proper error propagation to clients

Configuration Management
------------------------
- S2S output audio format derived from ``S2S_SERVICE`` env var
  (fixed: WAV for RIVA, MP3 for ElevenLabs; )
- ``is_speaker_info_provided`` auto-set based on ASD availability
- ASD service optional (bypass_asd per-request)
- gRPC message size limits
- Service endpoint configuration

Monitoring and Logging
----------------------
- Detailed logging for debugging and monitoring
- Request/response tracking
- Performance metrics collection

Usage
=====

The Controller Service is typically deployed as a Docker container and
exposes a gRPC interface for client applications:

.. code-block:: python

    # Client usage example
    import grpc
    from nvidia.ai4m.controller.v1.controller_pb2_grpc import ContentLocalizationControllerStub

    channel = grpc.insecure_channel("localhost:50056")
    stub = ContentLocalizationControllerStub(channel)

    # Stream requests to the service
    responses = stub.StreamContentLocalization(request_iterator)
    for response in responses:
        # Process video data
        if response.HasField("video_file_data"):
            # Handle video data
            pass

Configuration
=============

The service can be configured through environment variables:
- ```CONTROLLER_GRPC_API_PORT```: gRPC service port
- ```S2S_SERVER```: S2S service endpoint (optional, bypass_s2s when not provided)
- ```ASD_SERVER```: ASD service endpoint (optional, bypass_asd when not provided)
- ```LIPSYNC_SERVER```: LipSync service endpoint
- ```S2S_SERVICE```: S2S service type (e.g., "EL_DUBBING" for ElevenLabs)
"""

import argparse
import dataclasses
import os
import threading
import time
import traceback
import uuid
from collections.abc import Iterator

import grpc
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    ActiveSpeakerDetectionConfig,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    DetectActiveSpeakerResponse,
)
from nvidia.ai4m.audio.v1.audio_pb2 import AUDIO_CODEC_WAV
from nvidia.ai4m.controller.v1.controller_pb2 import ContentLocalizationRequest
from nvidia.ai4m.controller.v1.controller_pb2 import ContentLocalizationResponse
from nvidia.ai4m.controller.v1.controller_pb2_grpc import ContentLocalizationControllerServicer
from nvidia.ai4m.controller.v1.controller_pb2_grpc import (
    add_ContentLocalizationControllerServicer_to_server,
)
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncConfig
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncResponse
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechResponse

from base_utils import GRPCServiceBase
from base_utils import logger
from common.buffers import Buffer
from common.buffers import RequestIteratorFromBuffer
from common.nims import ActiveSpeakerDetectionClient
from common.nims import ActiveSpeakerDetectionServer
from common.nims import LipsyncClient
from common.nims import LipsyncServer
from common.nims import SpeechToSpeechClient
from common.nims import SpeechToSpeechServer
from controller_service.conversions import to_asd_audio_data
from controller_service.conversions import to_asd_diarization_data
from controller_service.conversions import to_asd_video_data
from controller_service.conversions import to_lipsync_background_audio
from controller_service.conversions import to_lipsync_video
from controller_service.conversions import to_s2s_request
from controller_service.deserializer import AudioQueueConsumer
from controller_service.deserializer import ContentLocalizationDeserializer
from controller_service.deserializer import DiarizationQueueConsumer
from controller_service.deserializer import VideoQueueConsumer
from controller_service.helpers import _FORMAT_TO_CODEC
from controller_service.helpers import _S2S_OUTPUT_FORMAT
from controller_service.helpers import CONTROLLER_CLEANUP_TIMEOUT
from controller_service.helpers import _audio_codec_to_format_string
from controller_service.helpers import _extract_config
from controller_service.stream_adapters import asd_request_generator
from controller_service.stream_adapters import asd_response_to_lipsync_speaker_info
from controller_service.stream_adapters import lipsync_request_generator
from controller_service.stream_adapters import s2s_audio_to_lipsync_audio
from controller_service.stream_adapters import translated_audio_to_lipsync_audio
from profiler.cprofile_profiler import CProfileProfiler
from profiler.metrics_tracker import MetricsTracker
from profiler.yappi_profiler import YappiProfiler


@dataclasses.dataclass
class _PipelineConfig:
    """Bundled configuration extracted per-request from client buffers.

    Groups the bypass flags, NIM configs, and derived audio format
    so they can be threaded through the pipeline helpers.

    Attributes:
        bypass_s2s: Skip S2S, use translated audio for LipSync.
        bypass_asd: Skip ASD, LipSync uses internal face detection.
        asd_config: ASD protobuf config, or ``None`` when bypassed.
        lipsync_config: LipSync protobuf config (with server
            overrides applied).
        input_audio_format: Derived from ASD config audio encoding.
        s2s_output_format: Resolved S2S output format for this
            request. ``None`` when S2S is bypassed.
    """

    bypass_s2s: bool
    bypass_asd: bool
    asd_config: ActiveSpeakerDetectionConfig | None
    lipsync_config: LipsyncConfig
    input_audio_format: str
    s2s_output_format: str | None


class ControllerServiceServicer(ContentLocalizationControllerServicer):
    """gRPC servicer implementation for the Controller Service.

    This class implements the gRPC servicer interface for the Controller Service,
    handling client communication, request processing, and response streaming.
    It acts as the bridge between gRPC clients and the main ControllerService
    implementation.

    Responsibilities
    ================

    - **Client Communication**: Handles gRPC request/response streaming
    - **Request Routing**: Routes client requests to the main service implementation
    - **Context Management**: Manages gRPC context and metadata
    - **Error Propagation**: Propagates errors from the service to clients
    - **Request Tracking**: Tracks request IDs and client information

    Key Methods
    ===========

    StreamContentLocalization
    -------------------------
    The main gRPC method that handles content localization requests:
    - Accepts streaming requests from clients
    - Generates unique request IDs for tracking
    - Delegates processing to the main ControllerService
    - Streams responses back to clients
    - Handles errors and propagates them appropriately

    Request Lifecycle
    =================

    1. **Request Reception**: Receives streaming requests from gRPC clients
    2. **Request ID Generation**: Generates unique UUID for request tracking
    3. **Service Delegation**: Delegates processing to ControllerService.infer()
    4. **Response Streaming**: Streams processed responses back to client
    5. **Error Handling**: Catches and propagates any processing errors

    Error Handling
    ==============

    The servicer provides robust error handling:
    - Catches exceptions from the main service implementation
    - Converts exceptions to appropriate gRPC status codes
    - Provides detailed error information to clients
    - Ensures proper cleanup on errors

    Thread Safety
    =============

    The servicer is designed to be thread-safe:
    - Each request gets a unique request ID
    - Request handling is isolated per request
    - Proper error isolation between concurrent requests

    Usage
    =====

    The servicer is typically used by the gRPC server framework:

    .. code-block:: python

        # Server setup
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        servicer = ControllerServiceServicer(controller_service)
        add_ContentLocalizationControllerServicer_to_server(servicer, server)
        server.add_insecure_port("[::]:50056")
        server.start()

    Client Interaction
    ==================

    Clients interact with this servicer through the gRPC interface:

    .. code-block:: python

        # Client usage
        channel = grpc.insecure_channel("localhost:50056")
        stub = ContentLocalizationControllerStub(channel)

        # Stream requests
        responses = stub.StreamContentLocalization(request_iterator)
        for response in responses:
            # Process responses
            pass
    """

    def __init__(self, service: "ControllerService") -> None:
        """Initialize the controller service servicer.

        Args:
            service (ControllerService): The controller service instance that provides for an
                content localization.
        """
        super().__init__()
        self.service = service

    def StreamContentLocalization(
        self,
        request_iterator: Iterator[ContentLocalizationRequest],
        context: grpc.ServicerContext,
    ) -> Iterator[ContentLocalizationResponse]:
        # Get the first request to extract the request_id
        logger.debug("Creating request id.")
        request_id = str(uuid.uuid4())
        peer = context.peer() if hasattr(context, "peer") else "unknown"
        logger.debug(f"Request received | id={request_id} | peer={peer}")

        # It is the responsibility of the infer method to handle content localization
        # and yield chunks of video in the ContentLocalizationResponse format.
        try:
            logger.debug("Running content localization call.")
            content_localization_response = self.service.infer(
                request_iterator=request_iterator,
                context=context,
                request_id=request_id,
            )
            logger.debug("self.service.infer completed successfully")
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Error processing request {request_id}: {e}\n{tb}")
            context.abort(grpc.StatusCode.INTERNAL, f"{type(e).__name__}: {e}\n{tb}")
        logger.debug("Content localization response setup passed.")

        # Yield content localization responses
        logger.debug("About to yield from content_localization_response")
        try:
            yield from content_localization_response
        except Exception as e:
            logger.error(f"Error in yield from content_localization_response: {e}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            raise


class ControllerService(GRPCServiceBase):
    """Main Controller Service for orchestrating the content localization pipeline.

    The ControllerService is the central orchestrator that coordinates between client
    applications and downstream AI services to provide end-to-end content localization
    capabilities. It uses a multi-threaded deserializer + buffer + client thread
    architecture.

    Responsibilities
    ================

    - **Service Orchestration**: Coordinates communication with S2S, ASD, and LipSync services
    - **Request Distribution**: Uses ContentLocalizationDeserializer to distribute
      requests to buffers
    - **Multi-threaded Processing**: Manages concurrent client threads for each service
    - **Response Streaming**: Streams processed video data back to clients
    - **Error Handling**: Provides comprehensive error handling and recovery

    Multi-Threaded Architecture
    ===========================

    The service uses the following architecture:

    Deserializer + Buffer + Client Threads
    ---------------------------------------
    - **ContentLocalizationDeserializer**: Background thread consumes gRPC request stream
    - **Typed Buffers**: Distribute requests to appropriate services
      (audio_buffer, video_buffer, translated_audio_buffer, controller_config_buffer, etc.)
    - **Client Threads**: Concurrent processing for each service
      - S2S Client Thread: Processes audio translation (skipped in bypass-S2S mode)
      - ASD Client Thread: Detects active speakers (skipped when bypass_asd)
      - LipSync Client Thread: Generates lip-synced video
    - **Bypass-S2S Mode**: When ``controller_config.bypass_s2s`` is True,
      S2S is skipped and ``translated_audio_buffer`` feeds LipSync directly
    - **Main Thread**: Yields ContentLocalizationResponse to client

    Key Components
    ==============

    Downstream Services
    -------------------
    - ```lipsync_server```: LipSync service for video processing
    - ```s2s_server```: Speech-to-Speech service for audio translation
    - ```asd_server```: Active Speaker Detection service (optional, bypass_asd when None)

    Configuration
    -------------
    - ```s2s_output_audio_format```: Derived from ``S2S_SERVICE`` env var.
      Fixed for EL/RIVA (``"MP3"`` / ``"WAV"``), and ``None`` for CambAI
      (meaning "mirror request input format"). Used as LipSync input codec
      only when S2S is active.
    - ```message_size```: Maximum gRPC message size

    Request Flow
    ============

    .. code-block:: text

        Client Request Stream
          ↓
        ContentLocalizationDeserializer (background thread)
          ├→ controller_config_buffer → determines bypass_s2s / bypass_asd
          ├→ audio_buffer → S2S Client Thread → s2s_output_buffer  [skip if bypass_s2s]
          ├→ video_buffer → ASD Client Thread → asd_output_buffer  [skip if bypass_asd]
          │              └→ LipSync Client Thread → lipsync_output_buffer
          ├→ diarization_buffer → ASD Client Thread  [skip if bypass_asd]
          ├→ background_audio_buffer → LipSync Client Thread (optional)
          └→ translated_audio_buffer → LipSync Client Thread (bypass S2S only)
          ↓
        Main Thread yields ContentLocalizationResponse

    Error Handling
    ==============

    The service provides comprehensive error handling:
    - gRPC communication errors with downstream services
    - Request processing errors in client threads
    - Service unavailability handling
    - Proper error propagation to clients
    - Graceful degradation when ASD is bypassed (bypass_asd)

    Thread Safety
    =============

    The service is designed to be thread-safe:
    - Deserializer runs on dedicated background thread
    - Each service has dedicated client thread(s)
    - Buffer-based communication ensures thread isolation
    - Proper resource cleanup and thread management in finally blocks

    Usage
    =====

    The ControllerService is typically instantiated with downstream service
    connections:

    .. code-block:: python

        service = ControllerService(
            lipsync_server=lipsync_server,
            s2s_server=s2s_server,  # Optional - None for bypass-S2S-only
            asd_server=asd_server,  # Optional — bypass_asd when None
        )
    """

    def __init__(
        self,
        lipsync_server: LipsyncServer,
        s2s_server: SpeechToSpeechServer | None = None,
        asd_server: ActiveSpeakerDetectionServer | None = None,
        message_size: int = 1024 * 1024 * 4,
    ) -> None:
        """Initialize the controller service.

        Args:
            lipsync_server (LipsyncServer): The lipsync server.
            s2s_server (SpeechToSpeechServer | None): The S2S server.
                ``None`` when running in bypass-S2S-only mode (all
                requests must set ``bypass_s2s=True``).
            asd_server (ActiveSpeakerDetectionServer | None): The asd
                server. If None, ASD is disabled.
            message_size (int): The maximum message size in bytes.
                Defaults to 1024 * 1024 * 4.
        """
        super().__init__(message_size=message_size)

        self.lipsync_server = lipsync_server
        self.s2s_server = s2s_server
        self.asd_server = asd_server

        # S2S output format is only needed when S2S is configured.
        if self.s2s_server is not None:
            s2s_service = os.environ.get("S2S_SERVICE", "EL_DUBBING")
            if s2s_service not in _S2S_OUTPUT_FORMAT:
                raise ValueError(
                    f"Unknown S2S_SERVICE={s2s_service!r}. "
                    f"Supported values: {list(_S2S_OUTPUT_FORMAT)}"
                )
            self.s2s_output_audio_format: str = _S2S_OUTPUT_FORMAT[s2s_service]
            logger.info(
                f"S2S backend={s2s_service}, s2s_output_audio_format={self.s2s_output_audio_format}"
            )
        else:
            self.s2s_output_audio_format: str = ""
            logger.info("S2S service not configured — only bypass-S2S requests supported")

        logger.debug("Controller Service initialized (deserializer + buffer + client threads)")

        if self.asd_server is None:
            logger.debug("ASD service disabled - LipSync will use internal face detection")

        if os.environ.get("CONTROLLER_PROFILER", 0) == 1:
            if os.environ.get("CONTROLLER_PROFILER_TYPE", "cprofiler").lower() == "yappi":
                self.profiler = YappiProfiler()
            elif os.environ.get("CONTROLLER_PROFILER_TYPE", "cprofiler").lower() == "cprofiler":
                self.profiler = CProfileProfiler()
            else:
                raise ValueError(
                    f"Invalid profiler type: {os.environ.get('CONTROLLER_PROFILER_TYPE', 'yappi')}"
                )
            logger.info("Profiler is enabled")
        else:
            logger.info("Profiler is disabled")
            self.profiler = None

        if os.environ.get("CONTROLLER_OTEL_PLUGIN", 0) == 1:
            from profiler.otel_plugin import OTelPlugin

            self.otel_plugin = OTelPlugin()
            logger.info("OpenTelemetry plugin is enabled")
        else:
            self.otel_plugin = None
            logger.info("OpenTelemetry plugin is disabled")

        self.metric_tracker = MetricsTracker()

    @staticmethod
    def argsfactory(
        parser: argparse.ArgumentParser | None = None,
    ) -> argparse.ArgumentParser:
        """Parser for command line arguments.

        Args:
            parser (argparse.ArgumentParser | None): Optional existing parser to extend.

        Returns:
            argparse.ArgumentParser: Unparsed command line arguments
        """
        if parser is None:
            parser = argparse.ArgumentParser(description="Controller Service")

        parser = GRPCServiceBase.argsfactory(parser)

        parser.add_argument(
            "--s2s-server",
            type=str,
            required=False,
            help="S2S service URI (host:port). Not required when running in bypass-S2S-only mode.",
        )
        parser.add_argument(
            "--asd-server",
            type=str,
            required=False,
            help="ASD NIM service URI (host:port). When omitted, only "
            "bypass_asd=True requests are supported.",
        )
        parser.add_argument(
            "--lipsync-server",
            type=str,
            required=True,
            help="LipSync NIM service URI (host:port)",
        )

        return parser

    def _check_services_health(
        self,
        bypass_s2s: bool = False,
        bypass_asd: bool = False,
        context: grpc.ServicerContext | None = None,
    ) -> bool:
        """Check health and preconditions for configured services.

        LipSync is always required. S2S and ASD are checked only
        when not bypassed. Aborts with ``FAILED_PRECONDITION`` if a
        required server is not configured.

        Args:
            bypass_s2s (bool): Skip S2S health check.
            bypass_asd (bool): Skip ASD health check.
            context (grpc.ServicerContext | None): gRPC context for
                aborting on precondition failures.

        Returns:
            bool: True if all checks pass, False if a precondition
                failed (abort already called on context).

        Raises:
            Exception: If a required server health check fails.

        Examples:
            >>> svc = ControllerService(...)  # doctest: +SKIP
            >>> svc._check_services_health(
            ...     bypass_s2s=True,
            ...     bypass_asd=False,
            ...     context=ctx,
            ... )
            True
        """
        self.lipsync_server.is_healthy()
        if not bypass_s2s:
            if self.s2s_server is None:
                msg = (
                    "S2S server is not configured but bypass_s2s is "
                    "not set. Provide --s2s-server or set "
                    "bypass_s2s=True in ContentLocalizationConfig."
                )
                logger.error(msg)
                if context is not None:
                    context.abort(
                        code=grpc.StatusCode.FAILED_PRECONDITION,
                        details=msg,
                    )
                return False
            self.s2s_server.is_healthy()
        if not bypass_asd:
            if self.asd_server is None:
                msg = (
                    "ASD server is not configured but bypass_asd is "
                    "not set. Provide --asd-server or set "
                    "bypass_asd=True in ContentLocalizationConfig."
                )
                logger.error(msg)
                if context is not None:
                    context.abort(
                        code=grpc.StatusCode.FAILED_PRECONDITION,
                        details=msg,
                    )
                return False
            self.asd_server.is_healthy()
        return True

    def add_servicer_to_server(self, server: grpc.Server) -> None:
        """Add the controller servicer to the gRPC server."""
        servicer = ControllerServiceServicer(self)
        add_ContentLocalizationControllerServicer_to_server(servicer, server)
        logger.debug("Added controller servicer to gRPC server")

    def _create_services(self) -> None:
        """Create gRPC channels for configured downstream services.

        Only creates channels for services that are configured
        (non-``None``). LipSync is always required; S2S and ASD
        are optional.
        """
        channel_options: list = []
        self.lipsync_server.create_server(channel_options=channel_options)
        if self.s2s_server is not None:
            self.s2s_server.create_server(channel_options=channel_options)
        if self.asd_server is not None:
            self.asd_server.create_server(channel_options=channel_options)

    def infer(
        self,
        request_iterator: Iterator[ContentLocalizationRequest],
        context: grpc.ServicerContext,
        request_id: str,
    ) -> Iterator[ContentLocalizationResponse]:
        """Main inference method for content localization processing.

        This method orchestrates the complete content localization pipeline, from
        client request ingestion through response streaming. It handles service
        initialization, health checks, and processes requests using the
        multi-threaded deserializer + buffer + client thread architecture.

        Processing Pipeline:
            1. Service Initialization - Creates connections to downstream services
            2. Health Check - Verifies all required services are available
            3. Request Processing - Multi-threaded processing via deserializer
            4. Response Streaming - Streams processed video data back to client
            5. Error Handling - Provides comprehensive error handling and recovery

        Architecture:
            - ContentLocalizationDeserializer: Background thread consumes gRPC stream
            - Typed Buffers: Distribute requests (audio_buffer, video_buffer)
            - Client Threads: Concurrent processing for S2S, ASD, and LipSync
            - Main Thread: Yields ContentLocalizationResponse to client

        Error Handling:
            The method provides robust error handling at multiple levels:
            - Service creation errors
            - Service health check failures
            - Request processing errors in client threads
            - gRPC communication errors

            All errors are properly logged and propagated to the client with
            appropriate gRPC status codes.

        Args:
                request_iterator: Stream of client requests containing audio/video data
                context: gRPC context for request metadata and cancellation
                request_id: Unique identifier for this request session

        Yields:
                ContentLocalizationResponse: Stream of processed video data

        Raises:
                grpc.RpcError: If there's an error in processing the stream or service communication
        """
        if self.profiler is not None:
            self.profiler.start(func_name=f"infer_{request_id}")
        if self.otel_plugin is not None:
            self.otel_plugin.register()
        try:
            self._create_services()
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Error creating services: {e}\n{tb}")
            context.abort(code=grpc.StatusCode.INTERNAL, details=f"{type(e).__name__}: {e}\n{tb}")

        try:
            logger.debug("Creating output iterator")
            controller_output_iterator = self._controller_impl(
                request_iterator=request_iterator,
                context=context,
                request_id=request_id,
            )
            logger.debug("Yielding from controller output iterator")
            yield from controller_output_iterator
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Error in yield from controller output iterator: {e}\n{tb}")
            context.abort(code=grpc.StatusCode.INTERNAL, details=f"{type(e).__name__}: {e}\n{tb}")

        if self.profiler is not None:
            self.profiler.stop()
        if self.otel_plugin is not None:
            self.otel_plugin.deregister()
        logger.info("Controller inference finished")

    def _extract_and_apply_configs(
        self,
        deserializer: ContentLocalizationDeserializer,
    ) -> "_PipelineConfig":
        """Extract client configs from buffers and apply server overrides.

        Reads controller, ASD, and LipSync configs from the
        deserializer's config buffers, then applies server-side
        overrides (``is_speaker_info_provided``, input audio codec).

        Args:
            deserializer (ContentLocalizationDeserializer): Active
                deserializer with populated config buffers.

        Returns:
            _PipelineConfig: Bundled pipeline configuration.

        Examples:
            >>> cfg = svc._extract_and_apply_configs(  # doctest: +SKIP
            ...     deserializer=des,
            ... )
        """
        controller_config = _extract_config(
            deserializer.controller_config_buffer, "controller_config"
        )
        bypass_s2s = controller_config.bypass_s2s if controller_config else False
        bypass_asd = controller_config.bypass_asd if controller_config else False

        # Only block on ASD config extraction when ASD is active
        asd_config = None
        if not bypass_asd and self.asd_server is not None:
            asd_config = _extract_config(deserializer.asd_config_buffer, "asd_config")
        lipsync_config = _extract_config(deserializer.lipsync_config_buffer, "lipsync_config")

        if lipsync_config is None:
            lipsync_config = LipsyncConfig()

        # is_speaker_info_provided driven by per-request bypass_asd
        has_asd = not bypass_asd and self.asd_server is not None
        lipsync_config.is_speaker_info_provided = has_asd

        input_audio_format = _audio_codec_to_format_string(
            asd_config.input_audio_config.encoding if asd_config else AUDIO_CODEC_WAV
        )

        # Override lipsync input codec when S2S is active so the
        # LipSync NIM knows what audio container to expect.
        s2s_output_format: str | None = None
        if not bypass_s2s and self.s2s_server is not None:
            s2s_output_format = self.s2s_output_audio_format

            expected_codec = _FORMAT_TO_CODEC[s2s_output_format]
            if lipsync_config.input_audio_codec != expected_codec:
                logger.warning(
                    f"Client lipsync_config.input_audio_codec "
                    f"({lipsync_config.input_audio_codec}) does not "
                    f"match S2S output format "
                    f"({s2s_output_format}). "
                    f"Overriding to {expected_codec}."
                )
                lipsync_config.input_audio_codec = expected_codec

        return _PipelineConfig(
            bypass_s2s=bypass_s2s,
            bypass_asd=bypass_asd,
            asd_config=asd_config,
            lipsync_config=lipsync_config,
            input_audio_format=input_audio_format,
            s2s_output_format=s2s_output_format,
        )

    def _start_s2s_thread(
        self,
        deserializer: ContentLocalizationDeserializer,
        s2s_output_buffer: "Buffer[SpeechToSpeechResponse]",
        input_audio_format: str,
        bypass_s2s: bool,
        context: grpc.ServicerContext,
        request_id: str,
    ) -> threading.Thread | None:
        """Start S2S client thread or drain thread when bypassed.

        Args:
            deserializer (ContentLocalizationDeserializer): Active
                deserializer.
            s2s_output_buffer (Buffer): Output buffer for S2S
                responses.
            input_audio_format (str): Audio format string.
            bypass_s2s (bool): Whether to skip S2S.
            context (grpc.ServicerContext): gRPC context.
            request_id (str): Request identifier.

        Returns:
            threading.Thread | None: Started thread, or ``None``
                if precondition check aborted.

        Examples:
            >>> thr = svc._start_s2s_thread(  # doctest: +SKIP
            ...     deserializer=des,
            ...     s2s_output_buffer=buf,
            ...     input_audio_format="wav",
            ...     bypass_s2s=False,
            ...     context=ctx,
            ...     request_id="r1",
            ... )
        """
        if not bypass_s2s:
            # Precondition validated in _check_services_health
            def _s2s_request_generator() -> Iterator:
                for req in RequestIteratorFromBuffer(deserializer.s2s_config_buffer, consumer_id=0):
                    yield to_s2s_request(req, input_audio_format=input_audio_format)
                for req in RequestIteratorFromBuffer(
                    deserializer.audio_buffer,
                    consumer_id=AudioQueueConsumer.S2S,
                ):
                    yield to_s2s_request(req, input_audio_format=input_audio_format)

            def _run_s2s() -> None:
                logger.debug(f"S2S client thread started: {threading.current_thread().name}")
                s2s_client = SpeechToSpeechClient(server=self.s2s_server)
                s2s_client(
                    request_iterator=_s2s_request_generator(),
                    output_buffer=s2s_output_buffer,
                    context=context,
                    request_id=request_id,
                )

            thread = threading.Thread(
                target=_run_s2s,
                daemon=True,
                name=f"S2S-{request_id}",
            )
            thread.start()
            logger.debug("S2S client thread launched")
            return thread

        # Drain audio_buffer queue 0 in bypass mode
        def _drain_s2s_audio() -> None:
            for _ in RequestIteratorFromBuffer(
                deserializer.audio_buffer,
                consumer_id=AudioQueueConsumer.S2S,
            ):
                pass
            for _ in RequestIteratorFromBuffer(deserializer.s2s_config_buffer, consumer_id=0):
                pass

        thread = threading.Thread(
            target=_drain_s2s_audio,
            daemon=True,
            name=f"S2S-drain-{request_id}",
        )
        thread.start()
        logger.debug("S2S thread skipped (bypass_s2s mode), drain launched")
        return thread

    def _start_asd_thread(
        self,
        deserializer: ContentLocalizationDeserializer,
        asd_config: "ActiveSpeakerDetectionConfig | None",
        bypass_asd: bool,
        context: grpc.ServicerContext,
        request_id: str,
    ) -> tuple[threading.Thread | None, "Buffer | None"]:
        """Start ASD client thread or drain threads when bypassed.

        When ``bypass_asd`` is True or ASD server is not configured,
        drains the four ASD-related buffer queues (video queue 0,
        audio queue 1, diarization queue 0, ASD config) to prevent
        accumulation.

        Args:
            deserializer (ContentLocalizationDeserializer): Active
                deserializer.
            asd_config (ActiveSpeakerDetectionConfig | None): ASD
                config protobuf or ``None``.
            bypass_asd (bool): Whether to skip ASD.
            context (grpc.ServicerContext): gRPC context.
            request_id (str): Request identifier.

        Returns:
            tuple[threading.Thread | None, Buffer | None]: The ASD
                thread (or drain thread) and the ASD output buffer
                (``None`` when bypassed).

        Examples:
            >>> thr, buf = svc._start_asd_thread(  # doctest: +SKIP
            ...     deserializer=des,
            ...     asd_config=cfg,
            ...     bypass_asd=False,
            ...     context=ctx,
            ...     request_id="r1",
            ... )
        """
        if not bypass_asd and self.asd_server is not None:
            asd_output_buffer: Buffer[DetectActiveSpeakerResponse] = Buffer(num_queues=1)
            asd_server = self.asd_server

            def _asd_video_iter() -> Iterator:
                for req in RequestIteratorFromBuffer(
                    deserializer.video_buffer,
                    consumer_id=VideoQueueConsumer.ASD,
                ):
                    yield to_asd_video_data(req)

            def _asd_audio_iter() -> Iterator:
                for req in RequestIteratorFromBuffer(
                    deserializer.audio_buffer,
                    consumer_id=AudioQueueConsumer.ASD,
                ):
                    yield to_asd_audio_data(req)

            def _asd_diarization_iter() -> Iterator:
                for req in RequestIteratorFromBuffer(
                    deserializer.diarization_buffer,
                    consumer_id=DiarizationQueueConsumer.ASD,
                ):
                    yield to_asd_diarization_data(req)

            def _run_asd() -> None:
                logger.debug(f"ASD client thread started: {threading.current_thread().name}")
                asd_client = ActiveSpeakerDetectionClient(server=asd_server)
                asd_request_iter = asd_request_generator(
                    video_iter=_asd_video_iter(),
                    audio_iter=_asd_audio_iter(),
                    asd_config=asd_config,
                    diarization_iter=_asd_diarization_iter(),
                )
                asd_client(
                    request_iterator=asd_request_iter,
                    output_buffer=asd_output_buffer,
                    context=context,
                    request_id=request_id,
                )

            asd_thread = threading.Thread(
                target=_run_asd,
                daemon=True,
                name=f"ASD-{request_id}",
            )
            asd_thread.start()
            logger.debug("ASD client thread launched")
            return asd_thread, asd_output_buffer

        # Drain ASD-related buffers so items don't accumulate
        def _drain_asd_buffers() -> None:
            for _ in RequestIteratorFromBuffer(
                deserializer.video_buffer,
                consumer_id=VideoQueueConsumer.ASD,
            ):
                pass
            for _ in RequestIteratorFromBuffer(
                deserializer.audio_buffer,
                consumer_id=AudioQueueConsumer.ASD,
            ):
                pass
            for _ in RequestIteratorFromBuffer(
                deserializer.diarization_buffer,
                consumer_id=DiarizationQueueConsumer.ASD,
            ):
                pass
            for _ in RequestIteratorFromBuffer(deserializer.asd_config_buffer, consumer_id=0):
                pass

        drain_thread = threading.Thread(
            target=_drain_asd_buffers,
            daemon=True,
            name=f"ASD-drain-{request_id}",
        )
        drain_thread.start()
        logger.debug("ASD bypassed; drain thread launched for ASD buffers")
        return drain_thread, None

    def _start_lipsync_thread(
        self,
        deserializer: ContentLocalizationDeserializer,
        s2s_output_buffer: "Buffer[SpeechToSpeechResponse]",
        asd_output_buffer: "Buffer[DetectActiveSpeakerResponse] | None",
        pipeline_config: "_PipelineConfig",
        context: grpc.ServicerContext,
        request_id: str,
    ) -> tuple[threading.Thread, "Buffer[LipsyncResponse]"]:
        """Start LipSync client thread with appropriate inputs.

        Wires video, audio, speaker info, and background audio
        iterators based on the pipeline configuration.

        Args:
            deserializer (ContentLocalizationDeserializer): Active
                deserializer.
            s2s_output_buffer (Buffer): S2S output buffer.
            asd_output_buffer (Buffer | None): ASD output buffer,
                or ``None`` when ASD is bypassed.
            pipeline_config (_PipelineConfig): Pipeline config.
            context (grpc.ServicerContext): gRPC context.
            request_id (str): Request identifier.

        Returns:
            tuple[threading.Thread, Buffer]: LipSync thread and
                output buffer.

        Examples:
            >>> thr, buf = svc._start_lipsync_thread(  # doctest: +SKIP
            ...     deserializer=des,
            ...     s2s_output_buffer=s2s_buf,
            ...     asd_output_buffer=asd_buf,
            ...     pipeline_config=cfg,
            ...     context=ctx,
            ...     request_id="r1",
            ... )
        """
        lipsync_output_buffer: Buffer[LipsyncResponse] = Buffer(num_queues=1)
        lipsync_config = pipeline_config.lipsync_config

        # Video input
        lipsync_video_iter = (
            to_lipsync_video(req)
            for req in RequestIteratorFromBuffer(
                deserializer.video_buffer,
                consumer_id=VideoQueueConsumer.LIPSYNC,
            )
        )

        # Audio input: translated audio or S2S output
        if pipeline_config.bypass_s2s:
            lipsync_audio_iter = translated_audio_to_lipsync_audio(
                request_iter=RequestIteratorFromBuffer(
                    deserializer.translated_audio_buffer,
                    consumer_id=0,
                ),
            )
        else:
            lipsync_audio_iter = s2s_audio_to_lipsync_audio(
                response_iter=RequestIteratorFromBuffer(s2s_output_buffer),
                audio_format=pipeline_config.s2s_output_format,
            )

        # Speaker info: from ASD output or None when bypassed
        if pipeline_config.bypass_asd:
            lipsync_speaker_info_iter = None
        else:
            lipsync_speaker_info_iter = asd_response_to_lipsync_speaker_info(
                response_iter=RequestIteratorFromBuffer(asd_output_buffer),
            )

        # Background audio
        has_background_audio = (
            lipsync_config.HasField("background_audio_config")
            and lipsync_config.background_audio_config.is_background_audio_provided
        )
        if has_background_audio:
            lipsync_bg_audio_iter = (
                to_lipsync_background_audio(req)
                for req in RequestIteratorFromBuffer(
                    deserializer.background_audio_buffer,
                    consumer_id=0,
                )
            )
        else:
            lipsync_bg_audio_iter = None

        lipsync_request_iter = lipsync_request_generator(
            video_iter=lipsync_video_iter,
            audio_iter=lipsync_audio_iter,
            speaker_info_iter=lipsync_speaker_info_iter,
            lipsync_config=lipsync_config,
            background_audio_iter=lipsync_bg_audio_iter,
        )

        def _run_lipsync() -> None:
            logger.debug(f"LipSync client thread started: {threading.current_thread().name}")
            lipsync_client = LipsyncClient(server=self.lipsync_server)
            lipsync_client(
                request_iterator=lipsync_request_iter,
                output_buffer=lipsync_output_buffer,
                context=context,
                request_id=request_id,
            )

        lipsync_thread = threading.Thread(
            target=_run_lipsync,
            daemon=True,
            name=f"LipSync-{request_id}",
        )
        lipsync_thread.start()
        logger.debug("LipSync client thread launched")
        return lipsync_thread, lipsync_output_buffer

    def _yield_responses(
        self,
        lipsync_output_buffer: "Buffer[LipsyncResponse]",
        request_id: str,
        context: grpc.ServicerContext,
    ) -> Iterator[ContentLocalizationResponse]:
        """Stream LipSync responses to the gRPC client.

        Args:
            lipsync_output_buffer (Buffer): Buffer of LipSync
                responses.
            request_id (str): Request identifier.
            context (grpc.ServicerContext): gRPC context.

        Yields:
            ContentLocalizationResponse: Responses with video data.

        Examples:
            >>> for resp in svc._yield_responses(  # doctest: +SKIP
            ...     lipsync_output_buffer=buf,
            ...     request_id="r1",
            ...     context=ctx,
            ... ):
            ...     pass
        """
        lipsync_response_iter = RequestIteratorFromBuffer(lipsync_output_buffer)
        logger.info("Starting to read LipSync responses.")
        response_count = 0
        _keepalive_count = 0
        for lipsync_response in lipsync_response_iter:
            self.metric_tracker.record_metric("lipsync_response_received", time.time())
            response_count += 1
            logger.debug(f"Processing LipSync response #{response_count}")
            try:
                if lipsync_response.HasField("video_file_data"):
                    self.metric_tracker.record_metric("outgoing_response_yield", time.time())
                    yield ContentLocalizationResponse(
                        video_file_data=lipsync_response.video_file_data,
                        request_id=request_id,
                    )
                elif lipsync_response.HasField("keepalive"):
                    logger.debug(f"Lipsync transmitted keep-alive response {_keepalive_count}.")
                    _keepalive_count += 1
                    continue
                else:
                    logger.debug("Ignoring non-video LipSync responses.")
            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"Error converting lipsync response: {e}\n{tb}")
                context.abort(
                    code=grpc.StatusCode.INTERNAL,
                    details=f"{type(e).__name__}: {e}\n{tb}",
                )
        logger.debug(
            f"LipSync response iterator finished after "
            f"{response_count} responses and "
            f"{_keepalive_count} keep-alive responses"
        )

    def _cleanup_threads(
        self,
        deserializer: ContentLocalizationDeserializer,
        threads: list[threading.Thread],
    ) -> None:
        """Stop deserializer and join all pipeline threads.

        Args:
            deserializer (ContentLocalizationDeserializer): Active
                deserializer to stop.
            threads (list[threading.Thread]): Threads to join.

        Examples:
            >>> svc._cleanup_threads(  # doctest: +SKIP
            ...     deserializer=des,
            ...     threads=[t1, t2],
            ... )
        """
        logger.debug("Cleanup: stopping deserializer and joining threads")
        try:
            deserializer.stop(timeout=CONTROLLER_CLEANUP_TIMEOUT)
        except Exception as e:
            logger.error(f"Error stopping deserializer: {e}")

        for thr in threads:
            try:
                thr.join(timeout=CONTROLLER_CLEANUP_TIMEOUT)
                if thr.is_alive():
                    logger.warning(f"Thread {thr.name} did not stop within timeout")
            except Exception as e:
                logger.error(f"Error joining thread {thr.name}: {e}")

        logger.debug("Cleanup completed")

        if os.environ.get("CONTROLLER_METRIC_TRACKER", "0") == "1":
            self.metric_tracker.dump_metrics_to_file(
                file_name=(f"raw_data_{time.strftime('%Y-%m-%d_%H-%M-%S')}"),
                raw_format=True,
            )
            self.metric_tracker.dump_metrics_to_file(
                file_name=(f"metrics_{time.strftime('%Y-%m-%d_%H-%M-%S')}"),
                raw_format=False,
            )
            self.metric_tracker.clear_metrics()

    def _controller_impl(
        self,
        request_iterator: Iterator[ContentLocalizationRequest],
        context: grpc.ServicerContext,
        request_id: str,
    ) -> Iterator[ContentLocalizationResponse]:
        """Core controller orchestrator for content localization.

        Thin orchestrator that delegates to helper methods:

        1. Start deserializer to consume the gRPC request stream
        2. Extract and apply configs
           (``_extract_and_apply_configs``)
        3. Health-check required services
           (``_check_services_health``)
        4. Launch S2S, ASD, and LipSync threads
        5. Yield responses (``_yield_responses``)
        6. Clean up threads (``_cleanup_threads``)

        Both ``bypass_s2s`` and ``bypass_asd`` are per-request
        flags read from ``ContentLocalizationConfig``.

        .. code-block:: text

            Client Request Stream
              |
            Deserializer (background thread)
              |-> S2S thread   [skip if bypass_s2s]
              |-> ASD thread   [skip if bypass_asd]
              |-> LipSync thread (always)
              |
            Main Thread yields ContentLocalizationResponse

        Args:
            request_iterator (Iterator[ContentLocalizationRequest]):
                Client request stream.
            context (grpc.ServicerContext): gRPC context.
            request_id (str): Unique request identifier.

        Yields:
            ContentLocalizationResponse: Processed video data.

        Raises:
            grpc.RpcError: On service communication errors.
        """
        logger.info(f"Service invoked for request id: {request_id}")
        logger.debug("Using Deserializer + Client thread pipeline")

        # --- 1. Deserializer: consume gRPC stream into buffers ---
        deserializer = ContentLocalizationDeserializer(request_iterator)
        deserializer.start(request_id=request_id)
        logger.debug("Deserializer thread started")

        # --- 2. Extract configs and apply server overrides ---
        cfg = self._extract_and_apply_configs(deserializer=deserializer)

        if cfg.bypass_s2s:
            logger.info("Bypass-S2S mode: using translated audio for LipSync")
        else:
            logger.info(
                f"Audio formats: input={cfg.input_audio_format}, "
                f"s2s_output={cfg.s2s_output_format or 'unresolved'}"
            )
        if cfg.bypass_asd:
            logger.info("Bypass-ASD mode: LipSync will use internal face detection")

        # --- 3. Health check (validates preconditions too) ---
        if not self._check_services_health(
            bypass_s2s=cfg.bypass_s2s,
            bypass_asd=cfg.bypass_asd,
            context=context,
        ):
            return

        # --- 4. Launch pipeline threads ---
        s2s_output_buffer: Buffer[SpeechToSpeechResponse] = Buffer(num_queues=1)
        s2s_thread = self._start_s2s_thread(
            deserializer=deserializer,
            s2s_output_buffer=s2s_output_buffer,
            input_audio_format=cfg.input_audio_format,
            bypass_s2s=cfg.bypass_s2s,
            context=context,
            request_id=request_id,
        )
        asd_thread, asd_output_buffer = self._start_asd_thread(
            deserializer=deserializer,
            asd_config=cfg.asd_config,
            bypass_asd=cfg.bypass_asd,
            context=context,
            request_id=request_id,
        )
        lipsync_thread, lipsync_output_buffer = self._start_lipsync_thread(
            deserializer=deserializer,
            s2s_output_buffer=s2s_output_buffer,
            asd_output_buffer=asd_output_buffer,
            pipeline_config=cfg,
            context=context,
            request_id=request_id,
        )

        # --- 5. Yield responses to gRPC client ---
        try:
            yield from self._yield_responses(
                lipsync_output_buffer=lipsync_output_buffer,
                request_id=request_id,
                context=context,
            )
        except Exception as e:
            logger.error(f"Exception in LipSync response processing: {e}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            context.abort(
                grpc.StatusCode.INTERNAL, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            )
        finally:
            # --- 6. Cleanup ---
            all_threads = [t for t in [s2s_thread, asd_thread, lipsync_thread] if t is not None]
            self._cleanup_threads(
                deserializer=deserializer,
                threads=all_threads,
            )
