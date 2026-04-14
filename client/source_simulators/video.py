# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video simulator classes for S2S client."""

import os
import time
from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Iterator
from pathlib import Path

from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    ActiveSpeakerDetectionData,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    DetectActiveSpeakerRequest,
)
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechRequest

from client.source_simulators.base import BaseFileSimulator
from client.utils import check_streamable

KB = 1024


class VideoSourceSimulator(BaseFileSimulator):
    """Simulate a video source as an iterator."""

    def __init__(self, file_path: os.PathLike):
        """Initialize VideoSourceSimulator with a video file path."""
        super().__init__(file_path=file_path)

        # Open video file for reading
        self._file_opened = open(file_path, "rb")

        # Check if video is streamable (for MP4 files)
        self.is_streamable = check_streamable(file_path)

        # Get file size for tracking
        self.file_size = os.path.getsize(file_path)

        self._chunk_count = 0

    def validate_file_path(self, value: os.PathLike) -> None:
        """Validate the file path of the video file."""
        if not os.path.exists(value):
            raise FileNotFoundError(f"File not found: {value}")

    def frames(
        self, chunk_size: int = 64 * 1024, process_video_callback: Callable | None = None
    ) -> Generator[bytes, None, None]:
        """Generator that yields video frames from the video file.

        Args:
            chunk_size (int): The size of each chunk in bytes.
                Once set, will not be changed until entire file is read, but can be
                changed for a second read.
            process_video_callback (Callable): A callback function to process the video frames
                once read.

        Returns:
            Generator[bytes, None, None]: A generator that yields video frames in bytes.
        """
        while True:
            frame_data = self._file_opened.read(chunk_size)
            if not frame_data:  # End of file
                break
            # Note down the timestamp of the frame taken out.
            self.ledger[self._chunk_count] = time.time()
            if process_video_callback:
                process_video_callback(frame_data)
            self._chunk_count += 1
            yield frame_data

    def read(
        self, chunk_size: int = 64 * 1024, process_video_callback: Callable | None = None
    ) -> Generator[bytes, None, None]:
        """Yield a video file as a generator.

        Args:
            chunk_size (int): The size of each chunk in bytes.
            process_video_callback (Callable | None): A callback function to process
                the video frames once read.

        Yields:
            Generator[bytes, None, None]: A generator that yields video frames in bytes.
        """
        print(f"Generating video from file: {self.file_path}")
        print(f"Video streamable: {self.is_streamable}")

        # generate frames
        yield from self.frames(chunk_size=chunk_size, process_video_callback=process_video_callback)


class VideoSinkSimulator(BaseFileSimulator):
    """Simulate a video sink to consume video frames from a generator."""

    def validate_file_path(self, value: os.PathLike) -> None:
        """Validate the file path of the video file."""
        if not os.path.exists(os.path.dirname(value)):
            raise FileNotFoundError(f"Directory not found: {os.path.dirname(value)}")

    def __init__(
        self,
        file_path: str = "output.mp4",
        chunk_size: int = 64 * 1024,
    ):
        """Initialize VideoSinkSimulator.

        Args:
            file_path (str): The path to the output video file.
            chunk_size (int): The size of each chunk in bytes.
        """
        super().__init__(file_path=file_path)

        # Open file in write mode.
        self._file_opened = open(file_path, "wb")

        # Chunk tracking
        self._chunk_count = 0
        self._buffer = b""
        self.chunk_size = chunk_size

    def write(
        self,
        video_bytes: bytes,
        process_video_callback: Callable | None = None,
    ) -> None:
        """Write video frames from a generator to a video file.

        Args:
            video_bytes (bytes): The bytes of the video file from the output of the generator.
            process_video_callback (Callable | None): A callback function to
                process the video frames once read.
        """
        self._buffer += video_bytes
        removed_from_buffer = None

        while len(self._buffer) >= self.chunk_size:
            removed_from_buffer = self._buffer[: self.chunk_size]
            # update buffer with rest of the frames overflown.
            self._buffer = self._buffer[self.chunk_size :]
            if process_video_callback:
                process_video_callback(removed_from_buffer)
            # write frames
            if self._file_opened is None:
                raise RuntimeError("Output video file is closed")
            self._file_opened.write(removed_from_buffer)
            self.ledger[self._chunk_count] = time.time()
            # Only print every 10th chunk to reduce output
            if self._chunk_count % 10 == 0:
                print(
                    f"Video sink | received chunk: {self._chunk_count} "
                    f"(buffer size: {len(self._buffer)})"
                )
            self._chunk_count += 1

    def flush(self) -> None:
        """Flush any remaining buffered data to the file."""
        if self._buffer and self._file_opened is not None:
            self._file_opened.write(self._buffer)
            self.ledger[self._chunk_count] = time.time()
            print(f"Flushing remaining video data: {len(self._buffer)} bytes")
            self._chunk_count += 1
            self._buffer = b""


def video_chunk_generator(
    mp4_path: Path | str,
    chunk_size: int = 64 * KB,
) -> Iterator[SpeechToSpeechRequest]:
    """Generate SpeechToSpeechRequest messages from a video file and return a request chunk.

    Args:
        mp4_path (str): The path to the video file.
        chunk_size (int): The chunk size for streaming video. Default is 64KB.
    """
    chunk_count = 0
    with Path(mp4_path).open(mode="rb") as fd:
        while True:
            chunk = fd.read(chunk_size)
            if chunk == b"":
                break
            yield SpeechToSpeechRequest(
                video_file_data=chunk,
            )
            chunk_count += 1


def simulated_asd_video_chunk_generator(
    simulator: VideoSourceSimulator, chunk_size: int = 64 * KB
) -> Iterator[DetectActiveSpeakerRequest]:
    """Generate DetectActiveSpeakerRequest messages with video data from a simulated video source.

    Args:
        simulator (VideoSourceSimulator): The simulated video source.
        chunk_size (int): The chunk size for streaming video. Default is 64KB.
    """
    for chunk in simulator.read(chunk_size=chunk_size):
        yield DetectActiveSpeakerRequest(
            data=ActiveSpeakerDetectionData(video_data=chunk),
        )


def simulated_video_chunk_generator_raw(
    simulator: VideoSourceSimulator, chunk_size: int = 64 * KB
) -> Iterator[bytes]:
    """Generate raw video chunks from a simulated video source and return a request chunk.

    Args:
        simulator (VideoSourceSimulator): The simulated video source.
        chunk_size (int): The chunk size for streaming video. Default is 64KB.
    """
    yield from simulator.read(chunk_size=chunk_size)
