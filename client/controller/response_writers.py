# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller client response writers (video output from controller responses)."""

import os
import time
from collections.abc import Iterator

from nvidia.ai4m.controller.v1.controller_pb2 import ContentLocalizationResponse

from client.controller.request_generators import FORWARD_PRESSURE_DELAY_SECS
from client.source_simulators.video import VideoSinkSimulator


def write_output_from_response(
    response_iter: Iterator[ContentLocalizationResponse],
    output_mp4_path: str,
    chunk_size_video_bytes: int = 64 * 1024,
) -> None:
    """Write video output from controller service responses using VideoSinkSimulator.

    Args:
        response_iter (Iterator[ContentLocalizationResponse]): Response
            iterator from controller.
        output_mp4_path (str): Path to output MP4 file.
        chunk_size_video_bytes (int): Chunk size for video processing.
            Defaults to ``65536``.

    Examples:
        >>> write_output_from_response(
        ...     response_iter=responses,
        ...     output_mp4_path="/tmp/output.mp4",
        ... )  # doctest: +SKIP
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_mp4_path), exist_ok=True)

    # Create video sink simulator
    output_video_sink = VideoSinkSimulator(
        file_path=output_mp4_path,
        chunk_size=chunk_size_video_bytes,
    )

    chunk_count = 0
    try:
        for response in response_iter:
            time.sleep(FORWARD_PRESSURE_DELAY_SECS)
            if response.HasField("keepalive"):
                continue  # Skip processing keep-alive responses
            elif response.HasField("video_file_data"):
                chunk_count += 1
                output_video_sink.write(video_bytes=response.video_file_data)
                if chunk_count % 100 == 0:
                    print(f"Controller | received chunk: {chunk_count}")
    except Exception as e:
        print(f"Error writing video output: {e}")
        raise
    finally:
        # Ensure we flush and close the sink
        if output_video_sink.is_open():
            output_video_sink.flush()
            output_video_sink.close()

    print(f"Controller | processed {chunk_count} video chunks")
    print(f"Controller | output written to: {output_mp4_path}")
