# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream adapter utilities for the Controller Service pipeline.

This module provides generator functions that adapt between different service
response streams in the multi-threaded architecture. These adapters:

* Generate ASD requests by merging video and audio streams
* Transform ASD speaker detection results into LipSync speaker info input
* Transform S2S audio responses into LipSync audio input
* Merge video, audio, and speaker info streams into LipSync requests

Architecture:
    (video + audio + diarization) → asd_request_generator() → ASD requests
    ASD responses → asd_response_to_lipsync_speaker_info() → LipSync speaker info input
    S2S responses → s2s_audio_to_lipsync_audio() → LipSync audio input
    Translated audio → translated_audio_to_lipsync_audio() → LipSync audio input (bypass S2S)
    (video + audio + speaker_info) → lipsync_request_generator() → LipSync requests
"""

from collections.abc import Iterator
from itertools import zip_longest

from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    ActiveSpeakerDetectionConfig,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    ActiveSpeakerDetectionData,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    DetectActiveSpeakerRequest,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    DetectActiveSpeakerResponse,
)
from nvidia.ai4m.common.v1.common_pb2 import BoundingBox
from nvidia.ai4m.controller.v1.controller_pb2 import ContentLocalizationRequest
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncConfig
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncInputData
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncRequest
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import SpeakerInfo as LipsyncSpeakerInfo
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import SpeakerInfoPerFrame
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechResponse

from base_utils import logger
from controller_service.conversions import create_wav_header
from controller_service.conversions import to_lipsync_translated_audio


def asd_request_generator(
    video_iter: Iterator[ActiveSpeakerDetectionData],
    audio_iter: Iterator[ActiveSpeakerDetectionData],
    asd_config: ActiveSpeakerDetectionConfig,
    diarization_iter: Iterator[ActiveSpeakerDetectionData] | None = None,
) -> Iterator[DetectActiveSpeakerRequest]:
    """Merge video, audio, and diarization streams into a ``DetectActiveSpeakerRequest`` stream.

    Emits the client-provided config as the first message, then interleaves
    video, audio, and optional diarization data chunks using
    :func:`itertools.zip_longest`.

    Args:
        video_iter: ASD video data (from video_buffer consumer).
        audio_iter: ASD audio data (from audio_buffer consumer).
        asd_config: Client-provided ``ActiveSpeakerDetectionConfig`` to send
            as the first message.
        diarization_iter: Optional ASD diarization data (from diarization_buffer consumer).

    Yields:
        ``DetectActiveSpeakerRequest`` messages ready for the ASD NIM.

    Examples:
        >>> from nvidia.ai4m.audio.v1.audio_pb2 import AUDIO_CODEC_WAV, AudioConfig
        >>> cfg = ActiveSpeakerDetectionConfig(
        ...     input_audio_config=AudioConfig(encoding=AUDIO_CODEC_WAV)
        ... )
        >>> reqs = asd_request_generator(
        ...     video_iter=iter([]),
        ...     audio_iter=iter([]),
        ...     asd_config=cfg,
        ... )
        >>> first = next(reqs)
        >>> first.HasField("config")
        True
    """
    # 1. Emit client-provided config as the first message
    yield DetectActiveSpeakerRequest(config=asd_config)
    logger.debug("asd_request_generator: sent config (pass-through from client)")

    # 2. Interleave video, audio, and diarization data
    chunk_counters = {"video": 0, "audio": 0, "diarization": 0}

    if diarization_iter is not None:
        iterators = (video_iter, audio_iter, diarization_iter)
    else:
        iterators = (video_iter, audio_iter)

    for items in zip_longest(*iterators, fillvalue=None):
        if diarization_iter is not None:
            video_data, audio_data, diarization_data = items
        else:
            video_data, audio_data = items
            diarization_data = None

        if video_data is not None:
            chunk_counters["video"] += 1
            yield DetectActiveSpeakerRequest(data=video_data)

        if audio_data is not None:
            chunk_counters["audio"] += 1
            yield DetectActiveSpeakerRequest(data=audio_data)

        if diarization_data is not None:
            chunk_counters["diarization"] += 1
            yield DetectActiveSpeakerRequest(data=diarization_data)

        total = sum(chunk_counters.values())
        if total % 100 == 0:
            logger.debug(
                f"asd_request_generator progress: video={chunk_counters['video']}, "
                f"audio={chunk_counters['audio']}, "
                f"diarization={chunk_counters['diarization']}"
            )

    logger.info(
        f"asd_request_generator complete: video={chunk_counters['video']}, "
        f"audio={chunk_counters['audio']}, diarization={chunk_counters['diarization']}"
    )


def asd_response_to_lipsync_speaker_info(
    response_iter: Iterator[DetectActiveSpeakerResponse],
) -> Iterator[LipsyncInputData]:
    """Yield ``LipsyncInputData`` with speaker info from an ASD response stream.

    Converts ASD ``ActiveSpeakerDetectionResult`` to LipSync ``SpeakerInfoPerFrame``,
    mapping bounding boxes and speaker metadata.

    Args:
        response_iter: Any iterator of ``DetectActiveSpeakerResponse``.

    Yields:
        ``LipsyncInputData`` with ``per_frame_speaker_infos`` populated.
    """
    count = 0
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
        count += 1
        yield LipsyncInputData(
            per_frame_speaker_infos=[
                SpeakerInfoPerFrame(
                    frame_id=result.frame_id,
                    speaker_infos=speaker_infos,
                )
            ]
        )
    logger.info(f"asd_response_to_lipsync_speaker_info: yielded {count} speaker_info frames")


def s2s_audio_to_lipsync_audio(
    response_iter: Iterator[SpeechToSpeechResponse],
    audio_format: str = "mp3",
) -> Iterator[LipsyncInputData]:
    """Yield ``LipsyncInputData`` audio chunks from an S2S response stream.

    On the first audio response the format is validated against
    *audio_format*.  For WAV, a synthetic header is emitted before the
    first audio-data chunk **only when the data is raw PCM** (no
    existing header).  If the first chunk already starts with a RIFF
    header the synthetic header is skipped to avoid a duplicate header
    with a wrong sample-rate.

    Args:
        response_iter: Any iterator of ``SpeechToSpeechResponse``.
        audio_format: Expected audio format (``"mp3"`` or ``"wav"``).

    Yields:
        ``LipsyncInputData`` with ``audio_file_data`` populated.

    """
    first_response = True
    for response in response_iter:
        if response.HasField("audio_data"):
            if first_response:
                audio_format_from_s2s = (
                    response.audio_format.lower() if response.audio_format else "mp3"
                )
                if audio_format_from_s2s != audio_format.lower():
                    logger.warning(
                        f"Audio format from S2S service is "
                        f"{audio_format_from_s2s}, but expected "
                        f"{audio_format.lower()}. Continuing with "
                        f"detected format."
                    )
                    audio_format = audio_format_from_s2s
                first_response = False

                # For WAV: only prepend a synthetic header when the
                # data is raw PCM (no header). Some backends stream a
                # complete WAV file whose first bytes are already a
                # RIFF header — adding a second header with a guessed
                # sample-rate makes the audio play at the wrong speed.
                data_already_has_header = response.audio_data[:4] == b"RIFF"
                if audio_format_from_s2s == "wav" and not data_already_has_header:
                    wav_header = create_wav_header(
                        n_channels=response.audio_num_channels or 1,
                        sample_width=2,  # Assuming 16-bit PCM
                        frame_rate=response.audio_sample_rate or 16000,
                        n_frames=0,
                    )
                    logger.debug("s2s_audio_to_lipsync_audio: yielding WAV header")
                    yield LipsyncInputData(audio_file_data=wav_header)
                elif data_already_has_header:
                    logger.debug(
                        "s2s_audio_to_lipsync_audio: first chunk "
                        "already contains a WAV header, skipping "
                        "synthetic header"
                    )
            yield LipsyncInputData(audio_file_data=response.audio_data)


def lipsync_request_generator(
    video_iter: Iterator[LipsyncInputData],
    audio_iter: Iterator[LipsyncInputData],
    speaker_info_iter: Iterator[LipsyncInputData] | None,
    lipsync_config: LipsyncConfig,
    background_audio_iter: Iterator[LipsyncInputData] | None = None,
) -> Iterator[LipsyncRequest]:
    """Merge video, audio, speaker info, and background audio into a ``LipsyncRequest`` stream.

    Emits the client-provided config as the first message, then interleaves
    data chunks using :func:`itertools.zip_longest`.

    Args:
        video_iter (Iterator[LipsyncInputData]): LipSync video input data.
        audio_iter (Iterator[LipsyncInputData]): LipSync audio input data
            (from S2S output).
        speaker_info_iter (Iterator[LipsyncInputData] | None): LipSync
            speaker info data (from ASD output), or ``None`` when ASD is
            disabled.
        lipsync_config (LipsyncConfig): Client-provided ``LipsyncConfig``
            to send as the first message.
        background_audio_iter (Iterator[LipsyncInputData] | None): Optional
            background audio data for LipSync mixing. ``None`` when no
            background audio is provided.

    Yields:
        LipsyncRequest: Messages ready for the LipSync service.

    Examples:
        >>> from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncConfig
        >>> cfg = LipsyncConfig()
        >>> reqs = lipsync_request_generator(iter([]), iter([]), None, cfg)
        >>> first = next(reqs)
        >>> first.HasField("config")
        True
    """
    # 1. Emit client-provided config as the first message
    yield LipsyncRequest(config=lipsync_config)
    logger.debug("lipsync_request_generator: sent config (pass-through from client)")

    # 2. Build dynamic iterators list based on available streams
    chunk_counters = {"video": 0, "audio": 0, "speaker_info": 0, "background_audio": 0}
    iterators: list[Iterator[LipsyncInputData]] = [video_iter, audio_iter]
    # Track which slot index maps to which stream type
    slot_names = ["video", "audio"]

    if speaker_info_iter is not None:
        iterators.append(speaker_info_iter)
        slot_names.append("speaker_info")

    if background_audio_iter is not None:
        iterators.append(background_audio_iter)
        slot_names.append("background_audio")

    for items in zip_longest(*iterators, fillvalue=None):
        for slot_idx, item in enumerate(items):
            if item is not None:
                name = slot_names[slot_idx]
                chunk_counters[name] += 1
                yield LipsyncRequest(input=item)

        total = sum(chunk_counters.values())
        if total % 100 == 0:
            logger.debug(
                f"lipsync_request_generator progress: "
                f"video={chunk_counters['video']}, "
                f"audio={chunk_counters['audio']}, "
                f"speaker_info={chunk_counters['speaker_info']}, "
                f"background_audio={chunk_counters['background_audio']}"
            )

    logger.info(
        f"lipsync_request_generator complete: "
        f"video={chunk_counters['video']}, "
        f"audio={chunk_counters['audio']}, "
        f"speaker_info={chunk_counters['speaker_info']}, "
        f"background_audio={chunk_counters['background_audio']}"
    )


def translated_audio_to_lipsync_audio(
    request_iter: Iterator[ContentLocalizationRequest],
) -> Iterator[LipsyncInputData]:
    """Yield ``LipsyncInputData`` audio chunks from pre-translated audio requests.

    Used in no-S2S mode: the client sends already-translated audio in
    ``translated_audio_data``, which is passed directly to LipSync
    without any S2S processing.

    Args:
        request_iter: Iterator of ``ContentLocalizationRequest`` with
            ``translated_audio_data`` populated.

    Yields:
        ``LipsyncInputData`` with ``audio_file_data`` from the
            translated audio bytes.

    Examples:
        >>> from nvidia.ai4m.controller.v1.controller_pb2 import (
        ...     ContentLocalizationRequest,
        ... )
        >>> reqs = [ContentLocalizationRequest(translated_audio_data=b"\\x00")]
        >>> items = list(translated_audio_to_lipsync_audio(iter(reqs)))
        >>> len(items)
        1
    """
    count = 0
    for request in request_iter:
        count += 1
        yield to_lipsync_translated_audio(request)
    logger.info(f"translated_audio_to_lipsync_audio: yielded {count} audio chunks")
