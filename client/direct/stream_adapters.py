# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream adapters for the direct client (ASD, LipSync request/response conversion).

NOTE: These adapters are client-side counterparts to the server-side adapters in
``controller_service.stream_adapters``.  They have diverged (print vs logger,
different backpressure logic) and are maintained separately.
"""

import time
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
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    DetectActiveSpeakerResponse,
)
from nvidia.ai4m.common.v1.common_pb2 import BoundingBox
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncConfig
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncInputData
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncRequest
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import SpeakerInfo as LipsyncSpeakerInfo
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import SpeakerInfoPerFrame

# This is the delay between sending chunks to the service to avoid backpressure.
# If this is too small, the gRPC server can't keep up and will drop the connection.
BACKPRESSURE_DELAY_SECS = 0.1


def asd_request_generator_with_audio(
    video_iter: Iterator[ActiveSpeakerDetectionData],
    audio_iter: Iterator[ActiveSpeakerDetectionData],
    asd_config: ActiveSpeakerDetectionConfig,
    diarization_info: AudioDiarizationInfo | None = None,
) -> Iterator[DetectActiveSpeakerRequest]:
    """Merge video and audio streams into a DetectActiveSpeakerRequest stream.

    Emits a config message first, then interleaves video and audio data chunks.
    Optionally includes diarization info with the first video data message.

    Args:
        video_iter (Iterator[ActiveSpeakerDetectionData]): ASD video data chunks.
        audio_iter (Iterator[ActiveSpeakerDetectionData]): ASD audio data chunks.
        asd_config (ActiveSpeakerDetectionConfig): Pre-built
            ``ActiveSpeakerDetectionConfig`` protobuf message.
        diarization_info (AudioDiarizationInfo | None): Optional diarization
            metadata to attach to the first video message.

    Yields:
        DetectActiveSpeakerRequest: Messages ready for the ASD NIM.

    Examples:
        >>> gen = asd_request_generator_with_audio(
        ...     video_iter=video_data,
        ...     audio_iter=audio_data,
        ...     asd_config=config,
        ... )  # doctest: +SKIP
    """

    # 1. Emit config as the first message
    yield DetectActiveSpeakerRequest(config=asd_config)
    print(f"ASD: sent config: {asd_config}")

    # 2. Interleave video and audio data
    diarization_sent = diarization_info is None  # True means "no need to send"
    chunk_counters = {"video": 0, "audio": 0}
    for video_data, audio_data in zip_longest(video_iter, audio_iter, fillvalue=None):
        if video_data is not None:
            chunk_counters["video"] += 1
            # Attach diarization info to the first video data message
            if not diarization_sent:
                enriched_data = ActiveSpeakerDetectionData(
                    video_data=video_data.video_data,
                    diarization_info=diarization_info,
                )
                diarization_sent = True
                print(
                    f"ASD: attached diarization info "
                    f"({len(diarization_info.segments)} segments) to first video message"
                )
                yield DetectActiveSpeakerRequest(data=enriched_data)
            else:
                yield DetectActiveSpeakerRequest(data=video_data)

        if audio_data is not None:
            chunk_counters["audio"] += 1
            yield DetectActiveSpeakerRequest(data=audio_data)

        total = chunk_counters["video"] + chunk_counters["audio"]
        if total % 100 == 0:
            print(f"ASD progress: video={chunk_counters['video']}, audio={chunk_counters['audio']}")

    print(f"ASD complete: video={chunk_counters['video']}, audio={chunk_counters['audio']}")


def speaker_info_from_asd_response(
    response_iter: Iterator[DetectActiveSpeakerResponse],
) -> Iterator[LipsyncInputData]:
    """Create a LipsyncInputData iterator from the ASD response stream.

    Converts ASD ActiveSpeakerDetectionResult to LipSync SpeakerInfoPerFrame,
    mapping bounding boxes and speaker metadata.

    Args:
        response_iter (Iterator[DetectActiveSpeakerResponse]): Iterator
            of DetectActiveSpeakerResponse objects.

    Yields:
        LipsyncInputData: Objects with per_frame_speaker_infos populated.

    Examples:
        >>> gen = speaker_info_from_asd_response(
        ...     response_iter=asd_responses,
        ... )  # doctest: +SKIP
    """
    for response in response_iter:
        result = response.active_speaker_detection_result
        speaker_infos = []
        for speaker in result.speaker_data:
            speaker_infos.append(
                LipsyncSpeakerInfo(
                    speaker_bbox=BoundingBox(
                        x=speaker.speaker_bbox.x,
                        y=speaker.speaker_bbox.y,
                        width=speaker.speaker_bbox.width,
                        height=speaker.speaker_bbox.height,
                    ),
                    speaker_id=speaker.face_id,
                    is_speaking=speaker.is_speaking,
                )
            )

        yield LipsyncInputData(
            per_frame_speaker_infos=[
                SpeakerInfoPerFrame(
                    frame_id=result.frame_id,
                    speaker_infos=speaker_infos,
                )
            ]
        )


def lipsync_input_request_generator(
    video_iterator: Iterator[LipsyncInputData],
    audio_iterator: Iterator[LipsyncInputData],
    speaker_info_iterator: Iterator[LipsyncInputData] | None,
    lipsync_config: LipsyncConfig,
    background_audio_iterator: Iterator[LipsyncInputData] | None = None,
) -> Iterator[LipsyncRequest]:
    """Generate a stream of LipsyncRequest messages for the LipSync service.

    Uses the audio_iterator output from the S2S service to generate the audio
    input for the LipSync service. Uses a brand new video source to generate
    the video input for the LipSync service to avoid re-encoding. Uses the
    speaker_info_iterator output from the ASD service to generate the speaker
    info input for the LipSync service. Optionally interleaves background
    audio data when provided.

    Args:
        video_iterator (Iterator[LipsyncInputData]): Iterator of
            LipsyncInputData for video.
        audio_iterator (Iterator[LipsyncInputData]): Iterator of
            LipsyncInputData for audio.
        speaker_info_iterator (Iterator[LipsyncInputData] | None): Iterator
            of LipsyncInputData for speaker info. If None, no speaker info
            input will be sent.
        lipsync_config (LipsyncConfig): Pre-built ``LipsyncConfig`` protobuf
            message.
        background_audio_iterator (Iterator[LipsyncInputData] | None):
            Optional iterator of LipsyncInputData for background audio.
            ``None`` when no background audio is provided.

    Yields:
        LipsyncRequest: Messages containing either configuration or chunks
            of input data.

    Examples:
        >>> gen = lipsync_input_request_generator(
        ...     video_iterator=video_iter,
        ...     audio_iterator=audio_iter,
        ...     speaker_info_iterator=None,
        ...     lipsync_config=config,
        ... )  # doctest: +SKIP
    """
    print(f"lipsync_input_request_generator called with config: {lipsync_config}")

    # Send configuration
    yield LipsyncRequest(config=lipsync_config)

    # Send input chunks
    chunk_counters = {"video": 0, "audio": 0, "speaker_info": 0, "background_audio": 0}

    # Track completion flags
    audio_done = False
    background_audio_done = background_audio_iterator is None

    # Prime audio early so the server initializes sample rate/resampler
    # before video/speaker-info arrives.
    # This helps avoid None sample rate for MP3 streams.
    primed_audio_chunk: LipsyncInputData | None = None
    try:
        primed_audio_chunk = next(audio_iterator)
        # Skip keepalive priming; only send real audio bytes
        if hasattr(primed_audio_chunk, "keepalive") and primed_audio_chunk.keepalive is not None:
            primed_audio_chunk = None
        else:
            chunk_counters["audio"] += 1
            yield LipsyncRequest(input=primed_audio_chunk)
    except StopIteration:
        primed_audio_chunk = None
        audio_done = True
    except Exception as e:  # pragma: no cover - defensive
        print(f"Audio priming failed, continuing without prime: {e}")
        primed_audio_chunk = None

    if speaker_info_iterator:
        # Use a streaming approach that processes data as it becomes available
        video_done = False
        speaker_info_done = False

        while not (video_done and audio_done and speaker_info_done and background_audio_done):
            # Process video chunks
            time.sleep(BACKPRESSURE_DELAY_SECS)
            if not video_done:
                try:
                    video_chunk = next(video_iterator)
                    chunk_counters["video"] += 1
                    yield LipsyncRequest(input=video_chunk)
                except StopIteration:
                    video_done = True

            # Process audio chunks - skip if not ready yet
            if not audio_done:
                try:
                    audio_chunk = next(audio_iterator)
                    chunk_counters["audio"] += 1
                    if hasattr(audio_chunk, "keepalive") and audio_chunk.keepalive is not None:
                        print(f"lipsync | sent keep-alive chunk: {chunk_counters['audio']}")
                        chunk_counters["audio"] -= 1
                        continue
                    yield LipsyncRequest(input=audio_chunk)
                except StopIteration:
                    audio_done = True
                except Exception as e:
                    # If audio stream is not ready yet, skip this iteration
                    print(f"Audio stream not ready yet, skipping: {e}")
                    time.sleep(BACKPRESSURE_DELAY_SECS)
                    continue

            # Process speaker info chunks - skip if not ready yet
            if not speaker_info_done:
                try:
                    speaker_info_chunk = next(speaker_info_iterator)
                    chunk_counters["speaker_info"] += 1
                    if (
                        hasattr(speaker_info_chunk, "keepalive")
                        and speaker_info_chunk.keepalive is not None
                    ):
                        print(f"lipsync | sent keep-alive chunk: {chunk_counters['speaker_info']}")
                        chunk_counters["speaker_info"] -= 1
                        continue
                    yield LipsyncRequest(input=speaker_info_chunk)
                except StopIteration:
                    speaker_info_done = True
                except Exception as e:
                    print(f"Speaker info stream not ready yet, skipping: {e}")
                    time.sleep(BACKPRESSURE_DELAY_SECS)
                    continue

            # Process background audio chunks if provided
            if not background_audio_done:
                try:
                    bg_chunk = next(background_audio_iterator)
                    chunk_counters["background_audio"] += 1
                    yield LipsyncRequest(input=bg_chunk)
                except StopIteration:
                    background_audio_done = True

            # Print progress every 100 chunks
            total = sum(chunk_counters.values())
            if total % 100 == 0:
                print(
                    f"lipsync | sent chunks: video: {chunk_counters['video']}, "
                    f"audio: {chunk_counters['audio']}, "
                    f"speaker_info: {chunk_counters['speaker_info']}, "
                    f"background_audio: {chunk_counters['background_audio']}"
                )
    else:  # This is needed for handling the case where ASD is disabled.
        # Use a streaming approach that processes data as it becomes available
        video_done = False
        if audio_done:
            # Reset to ensure we still attempt to consume audio in the no-speaker-info path.
            audio_done = False

        while not (video_done and audio_done and background_audio_done):
            # Process video chunks
            if not video_done:
                try:
                    video_chunk = next(video_iterator)
                    chunk_counters["video"] += 1
                    yield LipsyncRequest(input=video_chunk)
                except StopIteration:
                    video_done = True

            # Process audio chunks
            if not audio_done:
                try:
                    audio_chunk = next(audio_iterator)
                    chunk_counters["audio"] += 1
                    if hasattr(audio_chunk, "keepalive") and audio_chunk.keepalive is not None:
                        print(f"lipsync | sent keep-alive chunk: {chunk_counters['audio']}")
                        chunk_counters["audio"] -= 1
                        continue
                    yield LipsyncRequest(input=audio_chunk)
                except StopIteration:
                    audio_done = True

            # Process background audio chunks if provided
            if not background_audio_done:
                try:
                    bg_chunk = next(background_audio_iterator)
                    chunk_counters["background_audio"] += 1
                    yield LipsyncRequest(input=bg_chunk)
                except StopIteration:
                    background_audio_done = True

            # Print progress every 100 chunks
            total = sum(chunk_counters.values())
            if total % 100 == 0:
                print(
                    f"lipsync | sent chunks: "
                    f"video: {chunk_counters['video']}, "
                    f"audio: {chunk_counters['audio']}, "
                    f"background_audio: {chunk_counters['background_audio']}"
                )
    print(
        f"Transmission complete: video: {chunk_counters['video']}, "
        f"audio: {chunk_counters['audio']}, "
        f"speaker_info: {chunk_counters['speaker_info']}, "
        f"background_audio: {chunk_counters['background_audio']}"
    )
