# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ASD request stream generators for the standalone ASD client."""

from collections.abc import Iterator
from itertools import zip_longest

from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    ActiveSpeakerDetectionConfig,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    ActiveSpeakerDetectionData,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import AudioDiarizationInfo
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    DetectActiveSpeakerRequest,
)

from client.source_simulators.audio import AudioSourceSimulator
from client.source_simulators.video import VideoSourceSimulator


def asd_request_generator(
    video_source: VideoSourceSimulator,
    audio_source: AudioSourceSimulator,
    chunk_size_video_bytes: int,
    chunk_size_audio_secs: float,
    asd_config: ActiveSpeakerDetectionConfig,
    diarization_info: AudioDiarizationInfo | None = None,
) -> Iterator[DetectActiveSpeakerRequest]:
    """Generate a stream of DetectActiveSpeakerRequest messages for the ASD service.

    Sends config first, then interleaves video and audio data chunks.
    Optionally includes diarization info with the first data message.

    Args:
        video_source (VideoSourceSimulator): Video source simulator for
            reading video chunks.
        audio_source (AudioSourceSimulator): Audio source simulator for
            reading audio chunks.
        chunk_size_video_bytes (int): Size of each video chunk in bytes.
        chunk_size_audio_secs (float): Duration of each audio chunk in
            seconds.
        asd_config (ActiveSpeakerDetectionConfig): Pre-built
            ``ActiveSpeakerDetectionConfig`` protobuf message.
        diarization_info (AudioDiarizationInfo | None): Optional
            diarization metadata.

    Yields:
        DetectActiveSpeakerRequest: Messages containing config or data.

    Examples:
        >>> gen = asd_request_generator(
        ...     video_source=video_src,
        ...     audio_source=audio_src,
        ...     chunk_size_video_bytes=65536,
        ...     chunk_size_audio_secs=1.0,
        ...     asd_config=config,
        ... )  # doctest: +SKIP
    """
    # 1. Send config as the first message
    yield DetectActiveSpeakerRequest(config=asd_config)
    print(f"ASD: sent config: {asd_config}")

    # 2. Create iterators for video and audio data
    video_iter = video_source.read(chunk_size=chunk_size_video_bytes)
    audio_iter = audio_source.read(chunk_duration_secs=chunk_size_audio_secs)

    # 3. Send diarization info with the first data message if provided
    diarization_sent = diarization_info is None  # True means "no need to send"

    # 4. Interleave video and audio data
    chunk_counters = {"video": 0, "audio": 0}
    for video_chunk, audio_chunk in zip_longest(video_iter, audio_iter, fillvalue=None):
        if video_chunk is not None:
            chunk_counters["video"] += 1
            data_kwargs = {"video_data": video_chunk}
            # Attach diarization info to the first video data message
            if not diarization_sent:
                data_kwargs["diarization_info"] = diarization_info
                diarization_sent = True
            yield DetectActiveSpeakerRequest(data=ActiveSpeakerDetectionData(**data_kwargs))

        if audio_chunk is not None:
            chunk_counters["audio"] += 1
            yield DetectActiveSpeakerRequest(
                data=ActiveSpeakerDetectionData(audio_data=audio_chunk)
            )

        total = chunk_counters["video"] + chunk_counters["audio"]
        if total % 100 == 0:
            print(f"ASD progress: video={chunk_counters['video']}, audio={chunk_counters['audio']}")

    print(
        f"ASD data sending complete: video={chunk_counters['video']}, "
        f"audio={chunk_counters['audio']}"
    )
