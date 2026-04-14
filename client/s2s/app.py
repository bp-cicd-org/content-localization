# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Main S2S client implementation."""

import threading
from collections.abc import Iterator

from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechRequest
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechResponse

from client.context import LocalContext
from client.s2s.args import argsfactory
from client.s2s.args import s2s_config_from_args
from client.s2s.config import S2SConfig
from client.s2s.latency_analysis import calculate_output_stream_latencies
from client.s2s.latency_analysis import calculate_per_chunk_latencies
from client.s2s.latency_analysis import plot_latency
from client.s2s.response_writers import write_outputs_from_response
from client.source_simulators.audio import AudioSinkSimulator
from client.source_simulators.audio import AudioSourceSimulator
from client.source_simulators.audio import simulated_audio_chunk_generator
from client.utils import check_service_health
from common.buffers import Buffer
from common.buffers import RequestIteratorFromBuffer
from common.nims import SpeechToSpeechClient
from common.nims import SpeechToSpeechServer


def main() -> None:
    """Main function for the S2S client.

    This function:
    1. Checks service health
    2. Streams audio to the service
    3. Writes the output audio to a file.
    4. Calculates the latencies for the input and output streams.
    5. Prints the latencies.
    6. Checks if the output stream is real-time.
    """
    args = argsfactory().parse_args()

    # Build and validate configuration
    s2s_cfg = S2SConfig.from_args(args)
    s2s_cfg.validate_s2s_config()
    print(s2s_cfg)

    # Check service health
    check_service_health(server=args.s2s_server)
    print("S2S service is healthy")

    # Create the input and output file generators.
    input_file_generator = AudioSourceSimulator(file_path=args.input_audio)
    output_file_generator = AudioSinkSimulator(
        frame_rate=16000,
        sample_width=input_file_generator.sample_width,
        n_channels=input_file_generator.n_channels,
        n_frames=input_file_generator.n_frames,
        file_path=args.output_audio,
        chunk_duration_secs=args.chunk_size_audio_secs,
        audio_format=args.output_audio.split(".")[-1],
    )

    # Connect to the S2SService client abstraction
    host, port = args.s2s_server.split(":", 1)
    server = SpeechToSpeechServer(host=host, port=int(port))
    client = SpeechToSpeechClient(server=server)

    # Build protobuf config for the first gRPC request
    config = s2s_config_from_args(args)

    # Create a request generator that sends config first, then audio data
    def s2s_request_generator() -> Iterator[SpeechToSpeechRequest]:
        # Send config as the first request
        yield SpeechToSpeechRequest(config=config)

        # Then send audio data
        for audio_request in simulated_audio_chunk_generator(
            simulator=input_file_generator, chunk_size_secs=args.chunk_size_audio_secs
        ):
            yield audio_request

    output_buffer: Buffer[SpeechToSpeechResponse] = Buffer()
    context = LocalContext()

    def run_client() -> None:
        print(f"S2S client running on thread: {threading.current_thread().name}")
        client(
            request_iterator=s2s_request_generator(),
            output_buffer=output_buffer,
            context=context,
            request_id="s2s-client",
        )

    client_thread = threading.Thread(target=run_client, daemon=True)
    client_thread.start()

    response_iter = RequestIteratorFromBuffer(output_buffer, poll_timeout=0.1)
    write_outputs_from_response(
        response_iter=response_iter,
        output_file_generator=output_file_generator,
    )
    client_thread.join()

    latencies = calculate_per_chunk_latencies(
        input_ledger=input_file_generator.ledger, output_ledger=output_file_generator.ledger
    )
    output_stream_latencies = calculate_output_stream_latencies(
        input_ledger=input_file_generator.ledger, output_ledger=output_file_generator.ledger
    )

    # Generate latency analysis plot
    if args.latency_plot:
        plot_latency(
            output_stream_latencies=output_stream_latencies,
            per_chunk_latencies=latencies,
            chunk_size_secs=args.chunk_size_audio_secs,
            output_path=args.latency_plot,
        )

    is_realtime = all(latency < args.chunk_size_audio_secs for latency in output_stream_latencies)
    print(f"Is realtime output stream: {is_realtime}")


if __name__ == "__main__":
    main()
