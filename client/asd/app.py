# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple ASD (Active Speaker Detection) client implementation."""

import threading

from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    DetectActiveSpeakerResponse,
)

from client.asd.args import argsfactory
from client.asd.args import asd_config_from_args
from client.asd.config import ASDConfig
from client.asd.diarization import load_diarization_info
from client.asd.request_generators import asd_request_generator
from client.asd.response_writers import write_asd_outputs_from_response
from client.context import LocalContext
from client.source_simulators.audio import AudioSourceSimulator
from client.source_simulators.file import FileSourceSimulator
from client.source_simulators.video import VideoSourceSimulator
from client.utils import check_service_health
from common.buffers import Buffer
from common.buffers import RequestIteratorFromBuffer
from common.nims import ActiveSpeakerDetectionClient
from common.nims import ActiveSpeakerDetectionServer


def main() -> None:
    """Main function for the ASD client.

    This function:
    1. Checks service health
    2. Streams both video and audio to the ASD service
    3. Writes the speaker detection data to a CSV file
    4. Prints processing statistics
    """
    args = argsfactory().parse_args()

    # Build and validate configuration
    asd_cfg = ASDConfig.from_args(args)
    asd_cfg.validate_asd_config()
    print(asd_cfg)

    # Check service health
    check_service_health(server=args.asd_server)
    print("ASD service is healthy")

    # Create the input video and audio sources
    input_video_source = VideoSourceSimulator(file_path=args.input_mp4)
    if args.asd_input_audio_codec == "MP3":
        input_audio_source = FileSourceSimulator(file_path=args.input_audio)
    else:
        input_audio_source = AudioSourceSimulator(file_path=args.input_audio)

    # Load optional diarization info
    diarization_info = load_diarization_info(
        diarization_file=args.diarization_file,
        diarization_format=args.diarization_format,
    )
    if diarization_info:
        print(f"Loaded diarization info with {len(diarization_info.segments)} segments")

    # Build ASD config from shared args
    asd_config = asd_config_from_args(args)

    # Connect to the ASD Service client abstraction
    host, port = args.asd_server.split(":", 1)
    server = ActiveSpeakerDetectionServer(host=host, port=int(port))
    client = ActiveSpeakerDetectionClient(server=server)

    # Generate the request stream with both video and audio
    request_generator = asd_request_generator(
        video_source=input_video_source,
        audio_source=input_audio_source,
        chunk_size_video_bytes=args.chunk_size_video_bytes,
        chunk_size_audio_secs=args.chunk_size_audio_secs,
        asd_config=asd_config,
        diarization_info=diarization_info,
    )

    output_buffer: Buffer[DetectActiveSpeakerResponse] = Buffer()
    context = LocalContext()

    def run_client() -> None:
        print(f"ASD client running on thread: {threading.current_thread().name}")
        client(
            request_iterator=request_generator,
            output_buffer=output_buffer,
            context=context,
            request_id="asd-client",
        )

    client_thread = threading.Thread(target=run_client, daemon=True)
    client_thread.start()

    # Get responses from the ASD service
    asd_response_iter = RequestIteratorFromBuffer(output_buffer, poll_timeout=0.1)

    # Write the speaker detection data to CSV file
    write_asd_outputs_from_response(
        response_iter=asd_response_iter,
        output_csv_path=args.output_speaker_info,
    )

    # Close the input sources
    if input_video_source.is_open():
        input_video_source.close()
    if input_audio_source.is_open():
        input_audio_source.close()

    client_thread.join()

    print("ASD processing completed successfully")


if __name__ == "__main__":
    main()
