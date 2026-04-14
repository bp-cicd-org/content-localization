# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a single video through the controller pipeline."""

import grpc
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import AudioDiarizationInfo
from nvidia.ai4m.controller.v1.controller_pb2_grpc import ContentLocalizationControllerStub

from client.controller.config import ControllerConfig
from client.controller.request_generators import create_controller_request_generator
from client.controller.response_writers import write_output_from_response
from client.source_simulators.audio import AudioSourceSimulator
from client.source_simulators.video import VideoSourceSimulator


def run_single_video(
    audio_path: str,
    video_path: str,
    output_path: str,
    config: ControllerConfig,
    diarization_info: AudioDiarizationInfo | None = None,
) -> None:
    """Run one video through the controller content-localization pipeline.

    Creates input sources from the given audio and video files, streams
    them to the controller service, and writes the output video.

    Args:
        audio_path (str): Path to the extracted WAV audio file.
        video_path (str): Path to the input MP4 video file.
        output_path (str): Path for the output MP4 video file.
        config (ControllerConfig): Pipeline configuration bundle.
        diarization_info (AudioDiarizationInfo | None): Optional
            diarization metadata for ASD.

    Raises:
        grpc.RpcError: If the controller service returns an error.
        RuntimeError: If input files are invalid or output fails.

    Examples:
        >>> run_single_video(
        ...     audio_path="audio.wav",
        ...     video_path="video.mp4",
        ...     output_path="output.mp4",
        ...     config=cfg,
        ... )  # doctest: +SKIP
    """
    input_audio = AudioSourceSimulator(file_path=audio_path)
    input_video = VideoSourceSimulator(file_path=video_path)

    channel = grpc.insecure_channel(config.controller_server)
    stub = ContentLocalizationControllerStub(channel)

    try:
        request_generator = create_controller_request_generator(
            audio_source=input_audio,
            video_source=input_video,
            chunk_size_audio_secs=config.chunk_size_audio_secs,
            chunk_size_video_bytes=config.chunk_size_video_bytes,
            s2s_config=config.s2s_config,
            asd_config=config.asd_config,
            lipsync_config=config.lipsync_config,
            diarization_info=diarization_info,
        )

        response_iter = stub.StreamContentLocalization(request_generator)

        write_output_from_response(
            response_iter=response_iter,
            output_mp4_path=output_path,
            chunk_size_video_bytes=config.chunk_size_video_bytes,
        )
    finally:
        if input_audio.is_open():
            input_audio.close()
        if input_video.is_open():
            input_video.close()
        channel.close()
