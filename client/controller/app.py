# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller client implementation.

Streams audio, video, and optional diarization data to the Controller
gRPC service and writes the lip-synced output video.

Supports a **bypass-S2S mode** via ``--translated-audio``: when a
pre-translated audio file is provided, S2S is skipped and the
translated audio is sent directly to LipSync alongside the original
audio (still needed for ASD) and video.
"""

import time

import grpc
from nvidia.ai4m.controller.v1.controller_pb2_grpc import ContentLocalizationControllerStub

from client.asd.diarization import load_diarization_info
from client.controller.args import argsfactory
from client.controller.config import ControllerConfig
from client.controller.request_generators import create_controller_request_generator
from client.controller.response_writers import write_output_from_response
from client.source_simulators.audio import AudioSourceSimulator
from client.source_simulators.file import FileSourceSimulator
from client.source_simulators.video import VideoSourceSimulator
from client.utils import check_service_health


def _is_wav_file(file_path: str) -> bool:
    """Check whether *file_path* is a real WAV (RIFF) file.

    Reads the first 4 bytes and looks for the ``RIFF`` magic.
    Returns ``False`` for non-WAV files (e.g. MP3 with a ``.wav``
    extension, which ElevenLabs sometimes produces).

    Args:
        file_path (str): Path to the audio file.

    Returns:
        bool: ``True`` if the file starts with a RIFF header.

    Examples:
        >>> _is_wav_file("real.wav")  # doctest: +SKIP
        True
    """
    with open(file_path, "rb") as f:
        return f.read(4) == b"RIFF"


def main() -> None:
    """Main function for the controller client."""
    args = argsfactory().parse_args()

    # Build and validate configuration
    cfg = ControllerConfig.from_args(args)
    cfg.validate_io()
    print(cfg)

    start_time = time.time()
    check_service_health(server=cfg.controller_server)
    print("Controller service is healthy")

    # Create input sources — WAV uses AudioSourceSimulator, non-WAV uses FileSourceSimulator
    if _is_wav_file(cfg.input_audio):
        input_audio_source = AudioSourceSimulator(file_path=cfg.input_audio)
    else:
        input_audio_source = FileSourceSimulator(file_path=cfg.input_audio)
    input_video_source = VideoSourceSimulator(file_path=cfg.input_mp4)

    # Create background audio source only when provided
    # WAV files use AudioSourceSimulator; non-WAV uses FileSourceSimulator
    # (raw byte streaming, works for MP3 or mis-labelled files).
    bg_audio_source = None
    if cfg.background_audio_input:
        if _is_wav_file(cfg.background_audio_input):
            bg_audio_source = AudioSourceSimulator(file_path=cfg.background_audio_input)
        else:
            bg_audio_source = FileSourceSimulator(file_path=cfg.background_audio_input)
        print(f"Background audio source: {cfg.background_audio_input}")

    # Create translated audio source for no-S2S bypass mode.
    # WAV files use AudioSourceSimulator; non-WAV uses FileSourceSimulator
    # (raw byte streaming, works for MP3 or mis-labelled files).
    translated_audio_source = None
    if cfg.translated_audio:
        if _is_wav_file(cfg.translated_audio):
            translated_audio_source = AudioSourceSimulator(file_path=cfg.translated_audio)
        else:
            translated_audio_source = FileSourceSimulator(file_path=cfg.translated_audio)
        print(f"Translated audio source: {cfg.translated_audio} (S2S bypassed)")

    # Load optional diarization info
    diarization_info = load_diarization_info(
        diarization_file=cfg.diarization_file,
        diarization_format=args.diarization_format,
    )
    if diarization_info:
        print(f"Loaded diarization info with {len(diarization_info.segments)} segments")
    if cfg.bypass_asd:
        print("ASD bypassed — LipSync will use internal face detection")

    # Connect to the controller service
    channel = grpc.insecure_channel(cfg.controller_server)
    stub = ContentLocalizationControllerStub(channel)

    # Create request generator
    controller_request_generator = create_controller_request_generator(
        audio_source=input_audio_source,
        video_source=input_video_source,
        chunk_size_audio_secs=cfg.chunk_size_audio_secs,
        chunk_size_video_bytes=cfg.chunk_size_video_bytes,
        s2s_config=cfg.s2s_config,
        asd_config=cfg.asd_config,
        lipsync_config=cfg.lipsync_config,
        diarization_info=diarization_info,
        background_audio_source=bg_audio_source,
        translated_audio_source=translated_audio_source,
        bypass_asd=cfg.bypass_asd,
        diarization_rows_per_chunk=cfg.diarization_rows_per_chunk,
    )

    # Stream requests to the controller service
    controller_response_iter = stub.StreamContentLocalization(controller_request_generator)

    # Process responses and write output
    try:
        write_output_from_response(
            response_iter=controller_response_iter,
            output_mp4_path=cfg.output_mp4,
            chunk_size_video_bytes=cfg.chunk_size_video_bytes,
        )
    except grpc.RpcError as e:
        print(f"gRPC error occurred: {e.code()}: {e.details()}")
        print("This might be due to a timeout or connection issue. Try running the client again.")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

    # Close resources
    if input_audio_source.is_open():
        input_audio_source.close()
    if input_video_source.is_open():
        input_video_source.close()
    if bg_audio_source is not None and bg_audio_source.is_open():
        bg_audio_source.close()
    if translated_audio_source is not None and translated_audio_source.is_open():
        translated_audio_source.close()
    channel.close()

    elapsed = time.time() - start_time
    print(f"Controller client processing completed successfully in {elapsed} seconds")


if __name__ == "__main__":
    main()
