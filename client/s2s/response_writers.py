# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""S2S response writers (audio output from the response iterator)."""

from collections.abc import Iterator

from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechResponse

from client.source_simulators.audio import AudioSinkSimulator


def write_outputs_from_response(
    response_iter: Iterator[SpeechToSpeechResponse],
    output_file_generator: AudioSinkSimulator,
) -> None:
    """Write the output audio file from the S2S response iterator.

    Args:
        response_iter (Iterator[SpeechToSpeechResponse]): The response
            iterator.
        output_file_generator (AudioSinkSimulator): The output file
            generator.

    Raises:
        Exception: Re-raises any exception encountered while consuming
            the response iterator.

    Examples:
        >>> write_outputs_from_response(
        ...     response_iter=responses,
        ...     output_file_generator=sink,
        ... )  # doctest: +SKIP
    """
    try:
        for response_idx, response in enumerate(response_iter):
            if response.HasField("audio_data"):
                audio_size = len(response.audio_data)
                print(f"Chunk {response_idx}: Audio chunk received of {audio_size} bytes.")
                output_file_generator.write(wave_bytes=response.audio_data)

    except Exception as e:
        print(f"{e=}")
        raise e
    finally:
        # Close the output files.
        if output_file_generator.is_open():
            output_file_generator.close()
