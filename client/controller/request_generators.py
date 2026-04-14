# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller client request stream generators and related constants."""

import time
from collections.abc import Iterator
from itertools import zip_longest

from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    ActiveSpeakerDetectionConfig,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import AudioDiarizationInfo
from nvidia.ai4m.controller.v1.controller_pb2 import ContentLocalizationConfig
from nvidia.ai4m.controller.v1.controller_pb2 import ContentLocalizationRequest
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncConfig
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechConfig

from client.source_simulators.audio import AudioSourceSimulator
from client.source_simulators.audio import simulated_audio_chunk_generator_raw
from client.source_simulators.base import BaseFileSimulator
from client.source_simulators.video import VideoSourceSimulator
from client.source_simulators.video import simulated_video_chunk_generator_raw

BACKPRESSURE_DELAY_SECS = 0
FORWARD_PRESSURE_DELAY_SECS = 0
LOG_FOR_EVERY_N_CHUNKS = 1


def chunk_diarization_info(
    diarization_info: AudioDiarizationInfo,
    rows_per_chunk: int | None = None,
) -> list[AudioDiarizationInfo]:
    """Split diarization info into chunks of N segments each.

    Args:
        diarization_info (AudioDiarizationInfo): The full diarization info to chunk.
        rows_per_chunk (int | None): Number of segment rows per chunk.
            ``None`` sends all segments in a single message.

    Returns:
        list[AudioDiarizationInfo]: List of chunked diarization info messages.
            Empty list if no segments.

    Examples:
        >>> info = AudioDiarizationInfo(segments=[seg1, seg2, seg3])
        >>> chunks = chunk_diarization_info(info, rows_per_chunk=2)
        >>> len(chunks)
        2
        >>> chunks_all = chunk_diarization_info(info)
        >>> len(chunks_all)
        1
    """
    segments = list(diarization_info.segments)
    if not segments:
        return []

    # Send all segments in one message when rows_per_chunk is None
    if rows_per_chunk is None:
        return [diarization_info]

    chunks = []
    for i in range(0, len(segments), rows_per_chunk):
        chunk_segments = segments[i : i + rows_per_chunk]
        chunks.append(AudioDiarizationInfo(segments=chunk_segments))
    return chunks


def create_controller_request_generator(
    audio_source: AudioSourceSimulator,
    video_source: VideoSourceSimulator,
    chunk_size_audio_secs: float,
    chunk_size_video_bytes: int,
    s2s_config: SpeechToSpeechConfig | None,
    asd_config: ActiveSpeakerDetectionConfig | None,
    lipsync_config: LipsyncConfig,
    diarization_info: AudioDiarizationInfo | None = None,
    background_audio_source: BaseFileSimulator | None = None,
    translated_audio_source: BaseFileSimulator | None = None,
    bypass_asd: bool = False,
    diarization_rows_per_chunk: int | None = 10,
) -> Iterator[ContentLocalizationRequest]:
    """Create a generator that yields ContentLocalizationRequest objects.

    This generator sends NIM config messages first (controller config,
    optionally S2S, optionally ASD, LipSync), then interleaves audio,
    video, and optional diarization/translated audio chunks into a single
    request stream.

    When ``translated_audio_source`` is provided, S2S is bypassed: the
    controller config signals ``bypass_s2s=True`` and translated audio
    chunks are streamed alongside original audio (needed for ASD).

    Args:
        audio_source (AudioSourceSimulator): Audio source simulator.
        video_source (VideoSourceSimulator): Video source simulator.
        chunk_size_audio_secs (float): Audio chunk size in seconds.
        chunk_size_video_bytes (int): Video chunk size in bytes.
        s2s_config (SpeechToSpeechConfig | None): S2S config protobuf
            message, or ``None`` when S2S is bypassed.
        asd_config (ActiveSpeakerDetectionConfig | None): ASD config
            protobuf message, or ``None`` when ASD is disabled.
        lipsync_config (LipsyncConfig): LipSync config protobuf message.
        diarization_info (AudioDiarizationInfo | None): Optional diarization
            metadata.
        background_audio_source (AudioSourceSimulator | None): Optional
            background audio source for LipSync mixing. ``None`` when no
            background audio is provided.
        translated_audio_source (AudioSourceSimulator | None): Optional
            pre-translated audio source. When provided, S2S is bypassed
            and this audio feeds directly into LipSync.
        bypass_asd (bool): When True, tells the controller to skip ASD
            and use LipSync's internal face detection. Defaults to False.
        diarization_rows_per_chunk (int | None): Number of diarization
            segment rows per chunk. ``None`` sends all segments in a
            single message. Defaults to ``10``.

    Yields:
        ContentLocalizationRequest: Requests containing configs, diarization,
            audio and video data.

    Examples:
        >>> gen = create_controller_request_generator(
        ...     audio_source=audio_src,
        ...     video_source=video_src,
        ...     chunk_size_audio_secs=1.0,
        ...     chunk_size_video_bytes=1048576,
        ...     s2s_config=s2s_cfg,
        ...     asd_config=asd_cfg,
        ...     lipsync_config=ls_cfg,
        ... )  # doctest: +SKIP
    """
    bypass_s2s = translated_audio_source is not None

    # --- 1. Send all configs first ---
    # Controller config so the server knows the mode
    yield ContentLocalizationRequest(
        controller_config=ContentLocalizationConfig(
            bypass_s2s=bypass_s2s,
            bypass_asd=bypass_asd,
        ),
    )

    if asd_config is not None:
        yield ContentLocalizationRequest(asd_config=asd_config)
    yield ContentLocalizationRequest(lipsync_config=lipsync_config)

    if s2s_config is not None:
        print(
            f"Controller | sending S2S config: "
            f"source_language={s2s_config.source_language}, "
            f"target_language={s2s_config.target_language}, "
            f"voice_name={s2s_config.voice_name or 'None'}"
        )
        yield ContentLocalizationRequest(s2s_config=s2s_config)
    else:
        print("Controller | S2S bypassed — using translated audio")

    # --- 2. Send all diarization chunks before any data ---
    diarization_chunks = (
        chunk_diarization_info(
            diarization_info,
            rows_per_chunk=diarization_rows_per_chunk,
        )
        if diarization_info is not None
        else []
    )
    diarization_chunk_count = 0
    for diar_chunk in diarization_chunks:
        diarization_chunk_count += 1
        yield ContentLocalizationRequest(diarization_info=diar_chunk)
    if diarization_chunk_count:
        print(f"Controller | sent {diarization_chunk_count} diarization chunks")

    # --- 3. Create generators for audio, video, and optional streams ---
    audio_generator = simulated_audio_chunk_generator_raw(
        simulator=audio_source, chunk_size_secs=chunk_size_audio_secs
    )
    video_generator = simulated_video_chunk_generator_raw(
        simulator=video_source, chunk_size=chunk_size_video_bytes
    )

    # Background audio: only create generator when source is provided
    if background_audio_source is not None:
        bg_audio_generator = simulated_audio_chunk_generator_raw(
            simulator=background_audio_source,
            chunk_size_secs=chunk_size_audio_secs,
        )
    else:
        bg_audio_generator = iter([])

    # Translated audio: only create generator when source is provided
    if translated_audio_source is not None:
        translated_audio_generator = simulated_audio_chunk_generator_raw(
            simulator=translated_audio_source,
            chunk_size_secs=chunk_size_audio_secs,
        )
    else:
        translated_audio_generator = iter([])

    # --- 4. Interleave data streams (audio, video, background, translated) ---
    chunk_count = 0
    audio_chunk_count = 0
    video_chunk_count = 0
    bg_audio_chunk_count = 0
    translated_audio_chunk_count = 0
    audio_size_accumulator = 0
    video_size_accumulator = 0

    for (
        audio_chunk,
        video_chunk,
        bg_audio_chunk,
        translated_chunk,
    ) in zip_longest(
        audio_generator,
        video_generator,
        bg_audio_generator,
        translated_audio_generator,
        fillvalue=None,
    ):
        # Yield audio chunk if present (still needed for ASD even in bypass mode)
        if audio_chunk is not None:
            chunk_count += 1
            audio_chunk_count += 1
            audio_size_accumulator += len(audio_chunk)
            yield ContentLocalizationRequest(audio_data=audio_chunk)
            time.sleep(BACKPRESSURE_DELAY_SECS)

        # Yield video chunk if present
        if video_chunk is not None:
            chunk_count += 1
            video_chunk_count += 1
            video_size_accumulator += len(video_chunk)
            yield ContentLocalizationRequest(video_file_data=video_chunk)
            time.sleep(BACKPRESSURE_DELAY_SECS)

        # Yield background audio chunk if present
        if bg_audio_chunk is not None:
            chunk_count += 1
            bg_audio_chunk_count += 1
            yield ContentLocalizationRequest(background_audio_data=bg_audio_chunk)
            time.sleep(BACKPRESSURE_DELAY_SECS)

        # Yield translated audio chunk if present (bypass S2S mode)
        if translated_chunk is not None:
            chunk_count += 1
            translated_audio_chunk_count += 1
            yield ContentLocalizationRequest(translated_audio_data=translated_chunk)
            time.sleep(BACKPRESSURE_DELAY_SECS)

        # Print progress every N chunks or when all streams finish
        if chunk_count % LOG_FOR_EVERY_N_CHUNKS == 0 or (
            audio_chunk is None
            and video_chunk is None
            and bg_audio_chunk is None
            and translated_chunk is None
        ):
            print(f"Controller | sent {chunk_count} data requests to controller service")
            print(f"Controller | audio chunks: {audio_chunk_count}")
            print(f"Controller | video chunks: {video_chunk_count}")
            print(f"Controller | diarization chunks: {diarization_chunk_count}")
            print(f"Controller | bg audio chunks: {bg_audio_chunk_count}")
            print(f"Controller | translated audio chunks: {translated_audio_chunk_count}")
            print(f"Controller | total audio transmitted: {audio_size_accumulator} bytes")
            print(f"Controller | total video transmitted: {video_size_accumulator} bytes")

    print("Controller | finished transmitting requests to controller service")
    print(f"Controller | audio chunks: {audio_chunk_count}")
    print(f"Controller | video chunks: {video_chunk_count}")
    print(f"Controller | diarization chunks: {diarization_chunk_count}")
    print(f"Controller | bg audio chunks: {bg_audio_chunk_count}")
    print(f"Controller | translated audio chunks: {translated_audio_chunk_count}")
