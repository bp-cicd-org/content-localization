# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller service gRPC entrypoint."""

import sys

from base_utils import logger
from common.nims import ActiveSpeakerDetectionServer
from common.nims import LipsyncServer
from common.nims import SpeechToSpeechServer
from common.servers import GRPCServer
from controller_service.service import ControllerService


def main() -> None:
    """Main function to run the controller service.

    This function:
    1. Parses command line arguments
    2. Sets up logging
    3. Creates and initializes the controller service
    4. Starts the gRPC server
    5. Handles the service lifecycle
    """
    parser = ControllerService.argsfactory()
    args = parser.parse_args()

    logger.debug(f"args: {args}")

    try:
        # Parse service URIs
        service_uri = GRPCServer.from_string(url=args.service_uri)
        lipsync_server = GRPCServer.from_string(url=args.lipsync_server)

        # S2S is optional — not needed when running bypass-S2S-only
        s2s_server = None
        if args.s2s_server:
            s2s_server = GRPCServer.from_string(url=args.s2s_server)

        # ASD is optional — when omitted, only bypass_asd requests work
        asd_server = None
        if args.asd_server:
            asd_server = GRPCServer.from_string(url=args.asd_server)

        logger.info(f"Starting Controller Service on {service_uri.host}:{service_uri.port}")
        if s2s_server is not None:
            logger.info(f"S2S Server: {s2s_server.host}:{s2s_server.port}")
        else:
            logger.info("S2S Server: NOT CONFIGURED (bypass-S2S only)")
        if asd_server is not None:
            logger.info(f"ASD NIM Server: {asd_server.host}:{asd_server.port}")
        else:
            logger.info("ASD NIM Server: NOT CONFIGURED (only bypass_asd=True requests supported)")
        logger.info(f"LipSync NIM Server: {lipsync_server.host}:{lipsync_server.port}")

        # Create service instances
        lipsync_nim = LipsyncServer(host=lipsync_server.host, port=lipsync_server.port)
        s2s_nim = None
        if s2s_server is not None:
            s2s_nim = SpeechToSpeechServer(host=s2s_server.host, port=s2s_server.port)

        # Only create ASD service if not disabled
        asd_nim = None
        if asd_server is not None:
            asd_nim = ActiveSpeakerDetectionServer(host=asd_server.host, port=asd_server.port)

        # Create and start the controller service
        controller_service = ControllerService(
            lipsync_server=lipsync_nim,
            s2s_server=s2s_nim,
            asd_server=asd_nim,
            message_size=args.message_size,
        )

        # Start the service
        controller_service.serve(
            service_uri=args.service_uri,
            max_concurrency=args.max_concurrency,
            use_ssl=False,
            ssl_server_key_path=None,
            ssl_server_cert_path=None,
            ssl_root_cert_path=None,
            concurrency_mode=args.concurrency_mode,
            threads_per_process=args.threads_per_process,
        )

    except Exception as e:
        logger.error(f"Failed to start controller service: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
