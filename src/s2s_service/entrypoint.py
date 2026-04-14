#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speech-to-Speech (S2S) gRPC entrypoint."""

import argparse

from base_utils import logger
from s2s_service.camb_utils.dubbing import CambDubbingService
from s2s_service.el_utils.dubbing import ELDubbingService
from s2s_service.riva_utils.s2s import S2SRIVAStreamingService
from s2s_service.riva_utils.s2s import S2SRIVATransactionalService
from s2s_service.riva_utils.servers import RivaASRServer
from s2s_service.riva_utils.servers import RivaTTSServer


def main() -> None:
    """Main function to run the S2S service.

    This function:
    1. Parses command line arguments
    2. Sets up logging
    3. Creates and initializes the S2S service
    4. Starts the gRPC server
    5. Handles the service lifecycle
    """
    parser = argparse.ArgumentParser(
        description="Speech-to-Speech (S2S) gRPC entrypoint supporting RIVA and "
        "ElevenLabs backends."
    )
    subparsers = parser.add_subparsers(dest="service", required=True, help="S2S backend to use")

    # RIVA Transactional subcommand
    riva_transactional_parser = subparsers.add_parser(
        name="riva_transactional", help="Run with NVIDIA RIVA transactional backend"
    )
    S2SRIVATransactionalService.argsfactory(parser=riva_transactional_parser)

    # RIVA Streaming subcommand
    riva_streaming_parser = subparsers.add_parser(
        name="riva_streaming", help="Run with NVIDIA RIVA streaming backend"
    )
    S2SRIVAStreamingService.argsfactory(parser=riva_streaming_parser)

    # EL Dubbing subcommand
    el_dubbing_parser = subparsers.add_parser("el_dubbing", help="Run with ElevenLabs backend")
    ELDubbingService.argsfactory(parser=el_dubbing_parser)

    # CambAI Dubbing subcommand
    camb_dubbing_parser = subparsers.add_parser(
        "camb_dubbing", help="Run with CambAI dubbing backend"
    )
    CambDubbingService.argsfactory(parser=camb_dubbing_parser)

    args = parser.parse_args()

    logger.debug(f"args: {args}")

    if args.service == "riva_transactional":
        # Choosing to the run the RIVA service
        service = S2SRIVATransactionalService(
            ast_server=RivaASRServer.from_string(url=args.ast_server),
            tts_server=RivaTTSServer.from_string(url=args.tts_server),
            sample_rate_hz=args.sample_rate_hz,
            default_voice_name=args.default_voice_name,
            default_source_language=args.default_source_language,
            default_target_language=args.default_target_language,
            message_size=args.message_size,
            audio_format=args.audio_format,
        )
    elif args.service == "riva_streaming":
        service = S2SRIVAStreamingService(
            ast_server=RivaASRServer.from_string(url=args.ast_server),
            tts_server=RivaTTSServer.from_string(url=args.tts_server),
            sample_rate_hz=args.sample_rate_hz,
            default_voice_name=args.default_voice_name,
            default_source_language=args.default_source_language,
            default_target_language=args.default_target_language,
            message_size=args.message_size,
            audio_format=args.audio_format,
        )
    elif args.service == "el_dubbing":
        # Choosing to the run the EL service
        service = ELDubbingService(
            sample_rate_hz=args.sample_rate_hz,
            default_source_language=args.default_source_language,
            default_target_language=args.default_target_language,
            message_size=args.message_size,
            audio_format=args.audio_format,
        )
    elif args.service == "camb_dubbing":
        service = CambDubbingService(
            sample_rate_hz=args.sample_rate_hz,
            default_source_language=args.default_source_language,
            default_target_language=args.default_target_language,
            message_size=args.message_size,
            audio_format=args.audio_format,
        )
    else:
        parser.error(f"Unknown service: {args.service}")

    # TODO: Add full-support for SSL in a future MR.
    service.serve(
        service_uri=args.service_uri,
        max_concurrency=args.max_concurrency,
        use_ssl=False,
        ssl_server_key_path=None,
        ssl_server_cert_path=None,
        ssl_root_cert_path=None,
        concurrency_mode=args.concurrency_mode,
        threads_per_process=args.threads_per_process,
    )


if __name__ == "__main__":
    main()
