# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Main S2S client implementation."""

import threading

import grpc
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    ActiveSpeakerDetectionData,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    DetectActiveSpeakerResponse,
)
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechRequest
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechResponse

from client.asd.diarization import load_diarization_info
from client.context import LocalContext
from client.direct.args import argsfactory
from client.direct.config import DirectPipelineConfig
from client.direct.pipeline import audio_iterator_from_file
from client.direct.pipeline import audio_iterator_from_s2s_response_with_format
from client.direct.pipeline import background_audio_iterator_from_file
from client.direct.pipeline import video_iterator_from_source
from client.direct.stream_adapters import asd_request_generator_with_audio
from client.direct.stream_adapters import lipsync_input_request_generator
from client.direct.stream_adapters import speaker_info_from_asd_response
from client.source_simulators.audio import AudioSinkSimulator
from client.source_simulators.audio import AudioSourceSimulator
from client.source_simulators.audio import simulated_audio_chunk_generator
from client.source_simulators.video import VideoSinkSimulator
from client.source_simulators.video import VideoSourceSimulator
from client.source_simulators.video import simulated_video_chunk_generator_raw
from client.utils import check_service_health
from common.buffers import Buffer
from common.buffers import RequestIteratorFromBuffer
from common.nims import ActiveSpeakerDetectionClient
from common.nims import ActiveSpeakerDetectionServer
from common.nims import LipsyncClient
from common.nims import LipsyncServer
from common.nims import SpeechToSpeechClient
from common.nims import SpeechToSpeechServer


def main() -> None:
    """Main function for the S2S client."""
    args = argsfactory().parse_args()

    # Build pipeline config (server addresses + NIM configs + streaming params)
    pipeline_cfg = DirectPipelineConfig.from_args(args)

    # Check service health — skip S2S when using pre-translated audio
    using_translated_audio = pipeline_cfg.translated_audio is not None
    if not using_translated_audio:
        check_service_health(server=pipeline_cfg.s2s_server)
        print("S2S service is healthy")
    else:
        print(f"Using pre-translated audio: {pipeline_cfg.translated_audio}")
        print("Skipping S2S service")
    check_service_health(server=pipeline_cfg.lipsync_server)
    print("LipSync service is healthy")

    # I. S2S Input stream generation (skipped in translated-audio mode)
    input_s2s_audio_source = None
    output_s2s_audio_sink = None
    if not using_translated_audio:
        input_s2s_audio_source = AudioSourceSimulator(file_path=args.input_audio)
        output_s2s_audio_sink = AudioSinkSimulator(
            frame_rate=input_s2s_audio_source.frame_rate,
            sample_width=input_s2s_audio_source.sample_width,
            n_channels=input_s2s_audio_source.n_channels,
            n_frames=input_s2s_audio_source.n_frames,
            file_path=args.output_audio,
            chunk_duration_secs=pipeline_cfg.chunk_size_audio_secs,
            audio_format=args.output_audio.split(".")[-1],
        )

    def s2s_audio_request_generator():
        yield SpeechToSpeechRequest(config=pipeline_cfg.s2s_config)
        yield from simulated_audio_chunk_generator(
            simulator=input_s2s_audio_source,
            chunk_size_secs=pipeline_cfg.chunk_size_audio_secs,
        )

    # II. ASD Input stream generation
    if not pipeline_cfg.bypass_asd:
        check_service_health(server=pipeline_cfg.asd_server)
        print("ASD service is healthy")

        diarization_info = load_diarization_info(
            diarization_file=args.diarization_file,
            diarization_format=args.diarization_format,
        )
        if diarization_info:
            print(f"Loaded diarization info with {len(diarization_info.segments)} segments")

        input_asd_video_source = VideoSourceSimulator(file_path=args.input_mp4)
        input_asd_audio_source = AudioSourceSimulator(file_path=args.input_audio)

        def asd_video_data_iter():
            for chunk in input_asd_video_source.read(
                chunk_size=pipeline_cfg.chunk_size_video_bytes
            ):
                yield ActiveSpeakerDetectionData(video_data=chunk)

        def asd_audio_data_iter():
            for chunk in input_asd_audio_source.read(
                chunk_duration_secs=pipeline_cfg.chunk_size_audio_secs
            ):
                yield ActiveSpeakerDetectionData(audio_data=chunk)

        asd_request_gen = asd_request_generator_with_audio(
            video_iter=asd_video_data_iter(),
            audio_iter=asd_audio_data_iter(),
            asd_config=pipeline_cfg.asd_config,
            diarization_info=diarization_info,
        )

    # III. Setup the S2S Service client and output buffer (skipped in
    # translated-audio mode).
    s2s_thread = None
    if not using_translated_audio:
        s2s_host, s2s_port = pipeline_cfg.s2s_server.split(":", 1)
        s2s_server = SpeechToSpeechServer(
            host=s2s_host,
            port=int(s2s_port),
        )
        s2s_client = SpeechToSpeechClient(server=s2s_server)
        s2s_output_buffer: Buffer[SpeechToSpeechResponse] = Buffer()

        def run_s2s_client() -> None:
            print(f"S2S client running on thread: {threading.current_thread().name}")
            context = LocalContext()
            s2s_client(
                request_iterator=s2s_audio_request_generator(),
                output_buffer=s2s_output_buffer,
                context=context,
                request_id="direct-s2s",
            )

        s2s_thread = threading.Thread(target=run_s2s_client, daemon=True)
        s2s_thread.start()
        s2s_response_iter = RequestIteratorFromBuffer(
            s2s_output_buffer,
            poll_timeout=0.1,
        )
        print("S2S response iterator created - starting streaming")

    # IV. Setup the ASD Service client and output buffer.
    if pipeline_cfg.bypass_asd:
        asd_response_iter = None
        asd_thread = None
    else:
        asd_host, asd_port = pipeline_cfg.asd_server.split(":", 1)
        asd_server = ActiveSpeakerDetectionServer(host=asd_host, port=int(asd_port))
        asd_client = ActiveSpeakerDetectionClient(server=asd_server)
        asd_output_buffer: Buffer[DetectActiveSpeakerResponse] = Buffer()

        def run_asd_client() -> None:
            print(f"ASD client running on thread: {threading.current_thread().name}")
            context = LocalContext()
            asd_client(
                request_iterator=asd_request_gen,
                output_buffer=asd_output_buffer,
                context=context,
                request_id="direct-asd",
            )

        asd_thread = threading.Thread(target=run_asd_client, daemon=True)
        asd_thread.start()
        asd_response_iter = RequestIteratorFromBuffer(asd_output_buffer, poll_timeout=0.1)

    # IV. Setup the LipSync Inputs.
    # Note: Lipsync input audio is iterated from the S2S Service output.

    # Create video source and sink for LipSync processing
    input_lipsync_video_source = VideoSourceSimulator(file_path=args.input_mp4)
    output_lipsync_video_sink = VideoSinkSimulator(
        file_path=args.output_mp4,
        chunk_size=pipeline_cfg.chunk_size_video_bytes,
    )

    video_lipsync_iterator = simulated_video_chunk_generator_raw(
        simulator=input_lipsync_video_source,
        chunk_size=pipeline_cfg.chunk_size_video_bytes,
    )

    # Setup LipSync client and output buffer
    lipsync_host, lipsync_port = pipeline_cfg.lipsync_server.split(":", 1)
    lipsync_server = LipsyncServer(host=lipsync_host, port=int(lipsync_port))
    lipsync_client = LipsyncClient(server=lipsync_server)
    lipsync_output_buffer: Buffer = Buffer()

    # Now create LipSync request generator using the collected S2S and ASD responses
    # video_iterator for lipsync
    video_iterator = video_iterator_from_source(source_iterator=video_lipsync_iterator)
    # audio_iterator for lipsync — from file or S2S output
    if using_translated_audio:
        audio_iterator = audio_iterator_from_file(
            file_path=pipeline_cfg.translated_audio,
            chunk_size_secs=pipeline_cfg.chunk_size_audio_secs,
        )
    else:
        audio_iterator = audio_iterator_from_s2s_response_with_format(
            response_iter=s2s_response_iter,
            audio_format=args.output_audio.split(".")[-1],
            output_sink=output_s2s_audio_sink,
        )
    if not pipeline_cfg.bypass_asd:
        speaker_info_iter = speaker_info_from_asd_response(response_iter=asd_response_iter)
    else:
        speaker_info_iter = None

    # Background audio iterator — only when file is provided
    bg_audio_iter = None
    if pipeline_cfg.background_audio_input:
        bg_audio_iter = background_audio_iterator_from_file(
            file_path=pipeline_cfg.background_audio_input,
        )
        print(f"Background audio source: {pipeline_cfg.background_audio_input}")

    video_lipsync_request_generator = lipsync_input_request_generator(
        video_iterator=video_iterator,
        audio_iterator=audio_iterator,
        speaker_info_iterator=speaker_info_iter,
        lipsync_config=pipeline_cfg.lipsync_config,
        background_audio_iterator=bg_audio_iter,
    )

    def run_lipsync_client() -> None:
        print(f"LipSync client running on thread: {threading.current_thread().name}")
        context = LocalContext()
        lipsync_client(
            request_iterator=video_lipsync_request_generator,
            output_buffer=lipsync_output_buffer,
            context=context,
            request_id="direct-lipsync",
        )

    lipsync_thread = threading.Thread(target=run_lipsync_client, daemon=True)
    lipsync_thread.start()
    lipsync_response_iter = RequestIteratorFromBuffer(lipsync_output_buffer, poll_timeout=0.1)
    print("LipSync response iterator created - starting streaming")

    # Process LipSync responses
    chunk = 0
    try:
        for lipsync_response in lipsync_response_iter:
            if lipsync_response.video_file_data:
                chunk += 1
                output_lipsync_video_sink.write(video_bytes=lipsync_response.video_file_data)
                if chunk % 1000 == 0:
                    print(f"lipsync | received chunk: {chunk}")
    except grpc.RpcError as e:
        print(f"gRPC error occurred: {e.code()}: {e.details()}")
        print("This might be due to a timeout or connection issue. Try running the client again.")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

    print(f"Streaming completed successfully! Processed {chunk} LipSync response chunks.")

    # Close the input and output file generators.
    if input_s2s_audio_source and input_s2s_audio_source.is_open():
        input_s2s_audio_source.close()
    if output_s2s_audio_sink and output_s2s_audio_sink.is_open():
        output_s2s_audio_sink.close()
    if not pipeline_cfg.bypass_asd:
        if input_asd_video_source.is_open():
            input_asd_video_source.close()
        if input_asd_audio_source.is_open():
            input_asd_audio_source.close()
    if input_lipsync_video_source.is_open():
        input_lipsync_video_source.close()
    if output_lipsync_video_sink.is_open():
        output_lipsync_video_sink.flush()
        output_lipsync_video_sink.close()

    # Wait for background threads to finish
    lipsync_thread.join()
    if s2s_thread is not None:
        s2s_thread.join()
    if asd_thread is not None:
        asd_thread.join()


if __name__ == "__main__":
    main()
