# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration dataclass for the Direct client pipeline."""

import argparse
from dataclasses import dataclass

from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    ActiveSpeakerDetectionConfig,
)
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncConfig
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechConfig

from client.asd.args import asd_config_from_args
from client.direct.pipeline import _is_wav_file
from client.lipsync.args import lipsync_config_from_args
from client.lipsync.constants import AUDIO_CODEC_CONFIGS
from client.s2s.args import s2s_config_from_args

KB = 1024
MB = 1024 * KB


@dataclass
class DirectPipelineConfig:
    """Grouped configuration for a direct-client pipeline run.

    Bundles per-service server addresses, protobuf NIM configs, and
    streaming chunk sizes into a single object so they can be
    constructed once and threaded through the direct client pipeline.

    Attributes:
        s2s_server: S2S gRPC address (``host:port``).
        asd_server: ASD gRPC address (``host:port``).
        lipsync_server: LipSync gRPC address (``host:port``).
        s2s_config: Speech-to-Speech protobuf configuration,
            or ``None`` when using pre-translated audio.
        asd_config: Active Speaker Detection protobuf configuration,
            or ``None`` when ASD is bypassed.
        lipsync_config: LipSync protobuf configuration.
        chunk_size_audio_secs: Audio chunk duration in seconds.
        chunk_size_video_bytes: Video chunk size in bytes.
        bypass_asd: If ``True``, skip ASD and use LipSync internal
            face detection instead.

    Examples:
        >>> from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechConfig
        >>> from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
        ...     ActiveSpeakerDetectionConfig,
        ... )
        >>> from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncConfig
        >>> cfg = DirectPipelineConfig(
        ...     s2s_server="localhost:50050",
        ...     asd_server="localhost:50055",
        ...     lipsync_server="localhost:50054",
        ...     s2s_config=SpeechToSpeechConfig(),
        ...     asd_config=ActiveSpeakerDetectionConfig(),
        ...     lipsync_config=LipsyncConfig(),
        ... )
        >>> cfg.s2s_server
        'localhost:50050'
    """

    s2s_server: str
    asd_server: str
    lipsync_server: str
    s2s_config: SpeechToSpeechConfig | None
    asd_config: ActiveSpeakerDetectionConfig | None
    lipsync_config: LipsyncConfig
    chunk_size_audio_secs: float = 1.0
    chunk_size_video_bytes: int = 1 * MB
    bypass_asd: bool = False
    background_audio_input: str | None = None
    translated_audio: str | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "DirectPipelineConfig":
        """Build a ``DirectPipelineConfig`` from parsed CLI arguments.

        Delegates to ``s2s_config_from_args``, ``asd_config_from_args``,
        and ``lipsync_config_from_args`` to construct the protobuf
        config messages.

        Args:
            args (argparse.Namespace): Parsed argument namespace with
                direct-client, S2S, ASD, and LipSync attributes.

        Returns:
            DirectPipelineConfig: Populated configuration instance.

        Examples:
            >>> import argparse
            >>> args = argparse.Namespace(
            ...     s2s_server="localhost:50050",
            ...     asd_server="localhost:50055",
            ...     lipsync_server="localhost:50054",
            ...     chunk_size_audio_secs=1.0,
            ...     chunk_size_video_bytes=1048576,
            ...     bypass_asd=False,
            ...     source_language="en",
            ...     target_language="de",
            ...     voice_name=None,
            ...     elevenlabs_num_speakers=0,
            ...     elevenlabs_drop_background_audio=False,
            ...     elevenlabs_use_profanity_filter=False,
            ...     elevenlabs_target_accent=None,
            ...     elevenlabs_highest_resolution=False,
            ...     elevenlabs_watermark=False,
            ...     elevenlabs_dubbing_studio=False,
            ...     asd_input_audio_codec="WAV",
            ...     asd_input_video_codec=None,
            ...     lipsync_input_audio_codec="MP3",
            ...     lipsync_extend_audio="unspecified",
            ...     lipsync_extend_video="unspecified",
            ...     lipsync_output_bitrate_mbps=20,
            ...     lipsync_output_idr_interval=8,
            ...     lipsync_head_movement_speed=None,
            ...     lipsync_output_audio_codec=None,
            ...     lipsync_is_speaker_info_provided=False,
            ... )
            >>> cfg = DirectPipelineConfig.from_args(args)
            >>> cfg.s2s_server
            'localhost:50050'
        """
        lipsync_config = lipsync_config_from_args(args)

        # Auto-detect bypass_asd when no diarization file is provided
        bypass_asd = getattr(args, "bypass_asd", False)
        diarization_file = getattr(args, "diarization_file", None)
        if not bypass_asd and diarization_file is None:
            print("ASD bypassed — LipSync will use internal face detection")
            bypass_asd = True

        # When ASD is enabled, LipSync must know to expect speaker info
        if not bypass_asd:
            lipsync_config.is_speaker_info_provided = True

        translated_audio = getattr(args, "translated_audio", None)

        # Auto-detect actual audio codec from file content (not just
        # extension) because ElevenLabs sometimes returns MP3 data
        # inside a .wav filename.
        if translated_audio:
            actual_codec = "wav" if _is_wav_file(translated_audio) else "mp3"
            lipsync_config.input_audio_codec = AUDIO_CODEC_CONFIGS[actual_codec]

        s2s_config = None if translated_audio else s2s_config_from_args(args)

        asd_config = None if bypass_asd else asd_config_from_args(args)

        return cls(
            s2s_server=args.s2s_server,
            asd_server=args.asd_server,
            lipsync_server=args.lipsync_server,
            s2s_config=s2s_config,
            asd_config=asd_config,
            lipsync_config=lipsync_config,
            chunk_size_audio_secs=args.chunk_size_audio_secs,
            chunk_size_video_bytes=args.chunk_size_video_bytes,
            bypass_asd=bypass_asd,
            background_audio_input=getattr(args, "background_audio_input", None),
            translated_audio=translated_audio,
        )
