# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Argument parsing for the batch processing client."""

import argparse

from client.asd.args import add_asd_config_args_to_parser
from client.lipsync.args import add_lipsync_config_args_to_parser
from client.s2s.args import add_s2s_config_args_to_parser

MB = 1024 * 1024


def argsfactory() -> argparse.ArgumentParser:
    """Create argument parser for batch processing.

    Registers batch-processing-specific arguments (input/output directories)
    and delegates NIM-specific config arguments to the reusable helpers from
    the S2S, ASD, and LipSync client modules.

    Returns:
        argparse.ArgumentParser: Configured parser instance.

    Examples:
        >>> parser = argsfactory()
        >>> args = parser.parse_args(["--input-dir", "videos/"])
        >>> args.input_dir
        'videos/'
    """
    parser = argparse.ArgumentParser(
        description=(
            "Batch processing: run the end-to-end content "
            "localization pipeline on every video in a directory."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing input MP4 video files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/batch_processing",
        help=("Directory for output files (default: outputs/batch_processing)."),
    )
    parser.add_argument(
        "--controller-server",
        type=str,
        default="localhost:50056",
        help=("Controller gRPC server address (default: localhost:50056)."),
    )
    parser.add_argument(
        "--chunk-size-audio-secs",
        type=float,
        default=1.0,
        help="Audio chunk size in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--chunk-size-video-bytes",
        type=int,
        default=1 * MB,
        help="Video chunk size in bytes (default: 1 MB).",
    )
    parser.add_argument(
        "--s2s-service",
        type=str,
        default="EL_DUBBING",
        choices=["EL_DUBBING", "CAMB_DUBBING", "RIVA_TRANSACTIONAL"],
        help=(
            "S2S backend service. Controls diarization provider: "
            "CAMB_DUBBING uses Camb AI, others use ElevenLabs "
            "(default: EL_DUBBING)."
        ),
    )

    # Delegate NIM-specific config args
    add_s2s_config_args_to_parser(parser)
    add_asd_config_args_to_parser(parser)
    add_lipsync_config_args_to_parser(parser)

    return parser
