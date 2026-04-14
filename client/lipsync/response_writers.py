# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LipSync response writers (video output from inference responses)."""

import os
import time
from collections.abc import Iterator

from nvidia.ai4m.lipsync.v1 import lipsync_pb2

from client.lipsync.config import LipSyncConfig


def write_output_file_from_response(
    response_iter: Iterator[lipsync_pb2.LipsyncResponse],
    output_filepath: os.PathLike,
) -> None:
    """Write the video data from LipsyncResponse messages to an output file.

    Args:
        response_iter (Iterator[lipsync_pb2.LipsyncResponse]): Iterator of
            LipsyncResponse messages from the LipSync service.
        output_filepath (os.PathLike): Path where the output video will be saved.

    Raises:
        RuntimeError: If there are errors writing the output file.

    Examples:
        >>> write_output_file_from_response(
        ...     response_iter=responses,
        ...     output_filepath="/tmp/output.mp4",
        ... )  # doctest: +SKIP
    """
    try:
        chunk_number = 0
        with open(output_filepath, "wb") as fd:
            for response in response_iter:
                if response.HasField("video_file_data"):
                    if chunk_number == 0:
                        print(f"Writing output file {output_filepath}")
                    chunk_number += 1
                    fd.write(response.video_file_data)
        print(f"Output file written successfully: {output_filepath} ({chunk_number} chunks)")
    except OSError as e:
        raise RuntimeError(f"Error writing output file: {e}")


def process_response_iter(
    response_iter: Iterator[lipsync_pb2.LipsyncResponse],
    lipsync_config: LipSyncConfig,
) -> None:
    """Process gRPC response iterator and write output.

    Args:
        response_iter (Iterator[lipsync_pb2.LipsyncResponse]): Iterator
            of LipsyncResponse messages.
        lipsync_config (LipSyncConfig): Configuration for the LipSync service.

    Raises:
        Exception: If any errors occur during processing.

    Examples:
        >>> process_response_iter(
        ...     response_iter=responses,
        ...     lipsync_config=cfg,
        ... )  # doctest: +SKIP
    """
    try:
        start_time = time.time()

        # Skip first response (usually configuration acknowledgment)
        first_response = next(response_iter, None)
        if first_response is None:
            raise RuntimeError("No responses received from LipSync service")

        # Process video data
        write_output_file_from_response(
            response_iter=response_iter,
            output_filepath=lipsync_config.output_filepath,
        )

        end_time = time.time()
        print(f"Function invocation completed in {end_time - start_time:.2f}s")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise
