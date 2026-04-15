# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Argument parsing for S2S client.

Provides reusable ``add_s2s_config_args_to_parser`` and ``s2s_config_from_args``
so that any client (S2S, Direct, Controller) can share the same S2S config
arguments and build a ``SpeechToSpeechConfig`` identically.

Provider-specific arguments are modularised in separate files:

- ``elevenlabs_args.py`` — ElevenLabs dubbing flags
- ``camb_args.py`` — CambAI dubbing flags (placeholder for future use)

Both are composed into the S2S helpers automatically.
"""

import argparse

from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechConfig
from s2s.camb_args import add_camb_config_args_to_parser
from s2s.camb_args import apply_camb_args_to_config
from s2s.elevenlabs_args import add_elevenlabs_config_args_to_parser
from s2s.elevenlabs_args import apply_elevenlabs_args_to_config

KB = 1024

# Re-export for backward compatibility — callers may import these names
# directly from this module.
_apply_elevenlabs_args_to_config = apply_elevenlabs_args_to_config


def add_s2s_config_args_to_parser(
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    """Add S2S config arguments (including provider-specific) to a parser.

    Args:
        parser (argparse.ArgumentParser | None): Existing parser to extend.
            Creates a new one if ``None``.

    Returns:
        argparse.ArgumentParser: The parser with S2S config arguments added.

    Examples:
        >>> parser = argparse.ArgumentParser()
        >>> add_s2s_config_args_to_parser(parser)  # adds --source-language, --elevenlabs-*, etc.
    """
    if parser is None:
        parser = argparse.ArgumentParser("Speech-to-speech arguments.")
    parser.add_argument(
        "--source-language",
        type=str,
        default="en",
        help="Source language for speech-to-speech translation (default: en)",
    )
    parser.add_argument(
        "--target-language",
        type=str,
        default="de",
        help="Target language for speech-to-speech translation (default: de). "
        "Note: when using the RIVA Magpie Multilingual TTS model, "
        "only en-US is supported as an output language.",
    )
    add_elevenlabs_config_args_to_parser(parser)
    add_camb_config_args_to_parser(parser)
    return parser


def s2s_config_from_args(args: argparse.Namespace) -> SpeechToSpeechConfig:
    """Build a ``SpeechToSpeechConfig`` from parsed CLI arguments.

    Populates base S2S fields and delegates provider-specific fields
    to ``apply_elevenlabs_args_to_config`` and ``apply_camb_args_to_config``.

    Args:
        args (argparse.Namespace): Parsed argument namespace with
            ``source_language``, ``target_language``,
            and ``elevenlabs_*`` attributes.

    Returns:
        SpeechToSpeechConfig: Populated protobuf config message.

    Examples:
        >>> args = argparse.Namespace(
        ...     source_language="en",
        ...     target_language="de",
        ...     elevenlabs_num_speakers=0,
        ...     elevenlabs_drop_background_audio=False,
        ...     elevenlabs_use_profanity_filter=False,
        ...     elevenlabs_target_accent=None,
        ...     elevenlabs_highest_resolution=False,
        ...     elevenlabs_watermark=False,
        ...     elevenlabs_dubbing_studio=False,
        ... )
        >>> cfg = s2s_config_from_args(args)
        >>> cfg.target_language
        'de'
    """
    config = SpeechToSpeechConfig()
    if args.source_language:
        config.source_language = args.source_language
    if args.target_language:
        config.target_language = args.target_language
    # voice_name is not exposed as a CLI argument; populate from the
    # namespace only when set programmatically by a caller.
    voice_name = getattr(args, "voice_name", None)
    if voice_name:
        config.voice_name = voice_name
    apply_elevenlabs_args_to_config(args, config)
    apply_camb_args_to_config(args, config)
    return config


def argsfactory() -> argparse.ArgumentParser:
    """Factory function for creating an ArgumentParser instance.

    Returns:
        argparse.ArgumentParser: An ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(description="Speech-to-Speech (S2S) client")
    parser.add_argument(
        "--s2s-server",
        type=str,
        default="localhost:50050",
        help="Port of the S2S gRPC service (default: localhost:50050)",
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
    add_s2s_config_args_to_parser(parser)
    return parser
