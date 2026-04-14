# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol buffer conversion utilities for the Controller Service.

This module provides functions to convert between different protobuf message
formats used in the content localization pipeline:

* ContentLocalizationRequest → SpeechToSpeechRequest (S2S)
* ContentLocalizationRequest → ActiveSpeakerDetectionData (ASD video)
* ContentLocalizationRequest → ActiveSpeakerDetectionData (ASD audio)
* ContentLocalizationRequest → ActiveSpeakerDetectionData (ASD diarization)
* ContentLocalizationRequest → LipsyncInputData (LipSync video)
* ContentLocalizationRequest → LipsyncInputData (LipSync background audio)
* ContentLocalizationRequest → LipsyncInputData (LipSync translated audio,
  bypass S2S mode)

These conversions are used in the multi-threaded pipeline where the
ContentLocalizationDeserializer distributes incoming requests to different
service clients (S2S, ASD, LipSync).

Functions:
    to_s2s_request: Convert to S2S service format
    to_asd_video_data: Convert to ASD video data
    to_asd_audio_data: Convert to ASD audio data
    to_asd_diarization_data: Convert to ASD diarization data
    to_lipsync_video: Convert to LipSync video input format
    to_lipsync_background_audio: Convert to LipSync background audio input
    to_lipsync_translated_audio: Convert to LipSync audio input (bypass S2S)

Example:
    from controller_service.conversions import to_s2s_request

    s2s_req = to_s2s_request(content_localization_req)
"""

import traceback

from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    ActiveSpeakerDetectionData,
)
from nvidia.ai4m.controller.v1.controller_pb2 import ContentLocalizationRequest
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncInputData
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechRequest

from base_utils import logger
from common.audio_utils import create_wav_header  # noqa: F401  # re-exported
from controller_service.constants import AUDIO_CONFIG_DEFAULTS


def to_s2s_request(
    request: ContentLocalizationRequest,
    input_audio_format: str = AUDIO_CONFIG_DEFAULTS["audio_format"],
) -> SpeechToSpeechRequest:
    """Convert a ``ContentLocalizationRequest`` to a ``SpeechToSpeechRequest``.

    Uses :data:`AUDIO_CONFIG_DEFAULTS` for sample rate/channel defaults and
    uses ``input_audio_format`` for the request ``audio_format``.

    Args:
        request: Incoming content-localisation request.
        input_audio_format: Source/input audio format (for example, ``"WAV"`` or
            ``"MP3"``) to send to S2S.

    Returns:
        Populated ``SpeechToSpeechRequest``.

    Raises:
        ValueError: If neither ``audio_data`` nor ``s2s_config`` is present.

    Examples:
        >>> req = ContentLocalizationRequest(audio_data=b"\\x00\\x01")
        >>> s2s_req = to_s2s_request(req, input_audio_format="WAV")
        >>> s2s_req.audio_format
        'WAV'
    """
    try:
        s2s_request = SpeechToSpeechRequest()

        if request.HasField("audio_data"):
            s2s_request.audio_data = request.audio_data
            s2s_request.audio_sample_rate = AUDIO_CONFIG_DEFAULTS["audio_sample_rate"]
            s2s_request.audio_num_channels = AUDIO_CONFIG_DEFAULTS["audio_num_channels"]
            s2s_request.audio_format = input_audio_format.upper()

        if request.HasField("s2s_config"):
            s2s_request.config.CopyFrom(request.s2s_config)

        if not request.HasField("s2s_config") and not request.HasField("audio_data"):
            raise ValueError("S2S config or audio data must be provided in the request")
    except Exception as e:
        logger.error(f"Error generating S2S request: {e}")
        logger.error(f"Exception traceback: {traceback.format_exc()}")
        raise

    logger.debug(
        f"Generated S2S request: audio_data_present={request.HasField('audio_data')}, "
        f"audio_data_size={len(request.audio_data) if request.HasField('audio_data') else 0}, "
        f"s2s_config_present={request.HasField('s2s_config')}"
    )
    return s2s_request


def to_asd_video_data(request: ContentLocalizationRequest) -> ActiveSpeakerDetectionData:
    """Convert a ``ContentLocalizationRequest`` to ASD video data.

    Args:
        request: Incoming content-localisation request.

    Returns:
        ``ActiveSpeakerDetectionData`` with ``video_data`` populated.
    """
    if not request.HasField("video_file_data"):
        raise ValueError("Video data not found in request")
    return ActiveSpeakerDetectionData(video_data=request.video_file_data)


def to_asd_audio_data(request: ContentLocalizationRequest) -> ActiveSpeakerDetectionData:
    """Convert a ``ContentLocalizationRequest`` to ASD audio data.

    Args:
        request: Incoming content-localisation request.

    Returns:
        ``ActiveSpeakerDetectionData`` with ``audio_data`` populated.
    """
    if not request.HasField("audio_data"):
        raise ValueError("Audio data not found in request")
    return ActiveSpeakerDetectionData(audio_data=request.audio_data)


def to_asd_diarization_data(request: ContentLocalizationRequest) -> ActiveSpeakerDetectionData:
    """Convert a ``ContentLocalizationRequest`` to ASD diarization data.

    Args:
        request: Incoming content-localisation request.

    Returns:
        ``ActiveSpeakerDetectionData`` with ``diarization_info`` populated.
    """
    if not request.HasField("diarization_info"):
        raise ValueError("Diarization info not found in request")
    return ActiveSpeakerDetectionData(diarization_info=request.diarization_info)


def to_lipsync_video(request: ContentLocalizationRequest) -> LipsyncInputData:
    """Convert a ``ContentLocalizationRequest`` to ``LipsyncInputData`` (video).

    Args:
        request: Incoming content-localisation request.

    Returns:
        ``LipsyncInputData`` containing the video bytes.
    """
    if not request.HasField("video_file_data"):
        raise ValueError("Video data not found in request")
    return LipsyncInputData(video_file_data=request.video_file_data)


def to_lipsync_background_audio(
    request: ContentLocalizationRequest,
) -> LipsyncInputData:
    """Convert a ``ContentLocalizationRequest`` to ``LipsyncInputData`` (background audio).

    Args:
        request: Incoming content-localisation request with
            ``background_audio_data``.

    Returns:
        ``LipsyncInputData`` with ``background_audio_file_data`` populated.

    Raises:
        ValueError: If ``background_audio_data`` is not present.

    Examples:
        >>> req = ContentLocalizationRequest(background_audio_data=b"\\x00")
        >>> lip = to_lipsync_background_audio(req)
        >>> lip.background_audio_file_data
        b'\\x00'
    """
    if not request.HasField("background_audio_data"):
        raise ValueError("Background audio data not found in request")
    return LipsyncInputData(background_audio_file_data=request.background_audio_data)


def to_lipsync_translated_audio(
    request: ContentLocalizationRequest,
) -> LipsyncInputData:
    """Convert a ``ContentLocalizationRequest`` to ``LipsyncInputData`` (translated audio).

    Used in no-S2S mode: the client provides pre-translated audio that
    bypasses S2S and feeds directly into LipSync.

    Args:
        request: Incoming content-localisation request with
            ``translated_audio_data``.

    Returns:
        ``LipsyncInputData`` with ``audio_file_data`` populated.

    Raises:
        ValueError: If ``translated_audio_data`` is not present.

    Examples:
        >>> req = ContentLocalizationRequest(translated_audio_data=b"\\x00")
        >>> lip = to_lipsync_translated_audio(req)
        >>> lip.audio_file_data
        b'\\x00'
    """
    if not request.HasField("translated_audio_data"):
        raise ValueError("Translated audio data not found in request")
    return LipsyncInputData(audio_file_data=request.translated_audio_data)
