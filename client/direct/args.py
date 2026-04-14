# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Argument parsing for the Direct client.

Delegates S2S, ASD, and LipSync config arguments to their respective
shared helpers so all clients use identical configuration knobs.
"""

import argparse

from client.asd.args import add_asd_config_args_to_parser
from client.lipsync.args import add_lipsync_config_args_to_parser
from client.s2s.args import add_s2s_config_args_to_parser

KB = 1024


def argsfactory() -> argparse.ArgumentParser:
    """Factory function for creating an ArgumentParser instance.

    Returns:
        argparse.ArgumentParser: An ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(description="Direct client for S2S, ASD, and LipSync services")
    parser.add_argument(
        "--s2s-server",
        type=str,
        default="localhost:50050",
        help="Port of the S2S gRPC service (default: localhost:50050)",
    )
    parser.add_argument(
        "--lipsync-server",
        type=str,
        default="localhost:50054",
        help="Port of the LipSync gRPC service (default: localhost:50054)",
    )
    parser.add_argument(
        "--asd-server",
        type=str,
        default="localhost:50055",
        help="Port of the ASD gRPC service (default: localhost:50055)",
    )

    parser.add_argument(
        "--input-audio",
        type=str,
        default="assets/sample_audio.wav",
        help="Path to input file (default: assets/sample_audio.wav)",
    )
    parser.add_argument(
        "--output-audio",
        type=str,
        default="outputs/sample_audio_output.mp3",
        help="Path to output file, can be wav or mp3 (default: outputs/sample_audio_output.mp3)",
    )
    parser.add_argument(
        "--translated-audio",
        type=str,
        default=None,
        help="Path to a pre-translated WAV audio file. When provided, "
        "S2S is bypassed and this file is fed directly to LipSync. "
        "The --input-audio is still used for ASD speaker detection.",
    )
    parser.add_argument(
        "--chunk-size-audio-secs",
        type=float,
        default=1,
        help="Chunk size for streaming audio in seconds (default: 1)",
    )
    parser.add_argument(
        "--latency-plot",
        type=str,
        default="outputs/latency.png",
        help="Path to the latency plot (default: outputs/latency.png)",
    )

    parser.add_argument(
        "--input-mp4",
        type=str,
        default="assets/sample_video_streamable.mp4",
        help="Path to input video file, streamable mp4 only. "
        "If your file is not streamable mp4, you can convert to streamable mp4 using the script "
        "provided at: scripts/convert_to_streamable_mp4.sh "
        "(default: assets/sample_video_streamable.mp4)",
    )

    parser.add_argument(
        "--chunk-size-video-bytes",
        type=int,
        default=1024 * KB,
        help="Chunk size for streaming video (default: 1 MB)",
    )

    parser.add_argument(
        "--output-mp4",
        type=str,
        default="outputs/direct_output.mp4",
        help="Path to output video file, mp4 only. (default: outputs/direct_output.mp4)",
    )

    parser.add_argument(
        "--output-speaker-info",
        type=str,
        default="assets/asd_speaker_info.csv",
        help="Path to output speaker info CSV file. (default: assets/asd_speaker_info.csv)",
    )
    parser.add_argument(
        "--bypass-asd",
        action="store_true",
        default=False,
        help="Bypass ASD (Active Speaker Detection) service. "
        "Auto-enabled when no --diarization-file is provided.",
    )
    parser.add_argument(
        "--diarization-file",
        type=str,
        default=None,
        help="Path to JSON diarization file for speaker segments. "
        "Supports flat ASD format, native RIVA diarization JSON, or "
        "ElevenLabs STT JSON. Use --diarization-format to select the parser "
        "(default: auto-detect). "
        "Generate with: scripts/riva_parakeet_diarize.py, "
        "scripts/el_diarize.py, or scripts/camb_diarize.py",
    )
    parser.add_argument(
        "--diarization-format",
        type=str,
        default="elevenlabs",
        choices=["flat", "riva", "elevenlabs", "elevenlabs-studio", "camb"],
        help="Format of the diarization JSON file. "
        "'camb' for Camb AI transcription JSON "
        "(generate with scripts/camb_diarize.py) (default: elevenlabs).",
    )

    parser.add_argument(
        "--background-audio-input",
        type=str,
        default=None,
        help="Path to background audio file (WAV or MP3) for mixing "
        "with the LipSync output (optional)",
    )

    # Delegate S2S, ASD, and LipSync config args to shared helpers
    add_s2s_config_args_to_parser(parser)
    add_asd_config_args_to_parser(parser, default_audio_source="separate_stream")
    add_lipsync_config_args_to_parser(parser)

    return parser
