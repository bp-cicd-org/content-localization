# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LipSync request stream generators for the standalone LipSync client."""

import argparse
import csv
from collections.abc import Iterator
from contextlib import nullcontext

from nvidia.ai4m.common.v1.common_pb2 import BoundingBox
from nvidia.ai4m.lipsync.v1 import lipsync_pb2

from client.lipsync.args import _build_background_audio_config
from client.lipsync.config import LipSyncConfig
from client.lipsync.constants import AUDIO_CODEC_CONFIGS
from client.lipsync.constants import DATA_CHUNK_SIZE
from client.lipsync.constants import EXTEND_AUDIO_CONFIGS
from client.lipsync.constants import EXTEND_VIDEO_CONFIGS
from client.lipsync.constants import SPEAKER_INFO_FRAME_COUNT
from client.lipsync.encoding import create_output_video_encoding
from client.utils import speaker_info_csv_reader


def _speaker_info_from_row(row: list) -> tuple[int, lipsync_pb2.SpeakerInfo]:
    """Parse a single CSV row into a frame ID and SpeakerInfo protobuf.

    Args:
        row (list): List containing speaker info columns in one of these formats:
            - [frame_id, x, y, width, height]
            - [frame_id, x, y, width, height, diarized_speaker_id, face_id,
              is_speaking, ...]
            frame_id is used as the frame identifier.
            x, y, width, height define the speaker bounding box.
            face_id and is_speaking are consumed when provided.

    Returns:
        tuple[int, lipsync_pb2.SpeakerInfo]: The frame ID and a SpeakerInfo
            protobuf message for one detected face.

    Examples:
        >>> fid, info = _speaker_info_from_row(
        ...     row=["0", "10", "20", "30", "40"],
        ... )  # doctest: +SKIP
    """
    frame_id, x, y, width, height = row[0], *map(float, row[1:5])
    speaker_info = lipsync_pb2.SpeakerInfo(
        speaker_bbox=BoundingBox(
            x=float(x),
            y=float(y),
            width=float(width),
            height=float(height),
        )
    )

    # Optional metadata written by ASD CSV output:
    # [frame_id,x,y,w,h,diarized_speaker_id,face_id,is_speaking,confidence]
    if len(row) >= 7 and row[6] != "":
        speaker_info.speaker_id = int(row[6])

    if len(row) >= 8 and row[7] != "":
        speaker_info.is_speaking = row[7].strip().lower() in {
            "1",
            "true",
            "t",
            "yes",
            "y",
        }

    return int(frame_id), speaker_info


def group_rows_into_per_frame_infos(
    rows: list[list],
) -> list[lipsync_pb2.SpeakerInfoPerFrame]:
    """Group CSV rows by frame ID into SpeakerInfoPerFrame messages.

    Multiple speakers in the same frame are grouped into a single
    ``SpeakerInfoPerFrame`` with all their ``SpeakerInfo`` entries,
    matching how the direct client streams ASD results to LipSync.

    Args:
        rows (list[list]): Batch of CSV rows from the ASD output.

    Returns:
        list[lipsync_pb2.SpeakerInfoPerFrame]: One message per unique
            frame ID, each containing all speakers detected in that frame.

    Examples:
        >>> infos = group_rows_into_per_frame_infos(
        ...     rows=[["0", "10", "20", "30", "40"]],
        ... )  # doctest: +SKIP
    """
    # Preserve frame ordering while grouping speakers per frame
    frames: dict[int, list[lipsync_pb2.SpeakerInfo]] = {}
    for row in rows:
        frame_id, speaker_info = _speaker_info_from_row(row)
        frames.setdefault(frame_id, []).append(speaker_info)

    return [
        lipsync_pb2.SpeakerInfoPerFrame(
            frame_id=frame_id,
            speaker_infos=speaker_infos,
        )
        for frame_id, speaker_infos in frames.items()
    ]


def generate_request_for_inference(
    lipsync_config: LipSyncConfig,
) -> Iterator[lipsync_pb2.LipsyncRequest]:
    """Generate a stream of LipsyncRequest messages for the LipSync service.

    Args:
        lipsync_config (LipSyncConfig): Configuration object containing all
            LipSync parameters.

    Yields:
        lipsync_pb2.LipsyncRequest: Messages containing either configuration
            or chunks of input data.

    Raises:
        RuntimeError: If there are errors reading input files.

    Examples:
        >>> gen = generate_request_for_inference(
        ...     lipsync_config=cfg,
        ... )  # doctest: +SKIP
    """
    print("Generating request for inference")

    # Create output video encoding configuration
    output_video_encoding = create_output_video_encoding(config=lipsync_config)

    # Prepare configuration parameters
    if lipsync_config.audio_codec is None:
        raise RuntimeError("Audio codec is not set. Validate config before generating requests.")

    params = {
        "input_audio_codec": AUDIO_CODEC_CONFIGS[lipsync_config.audio_codec],
        "extend_audio": EXTEND_AUDIO_CONFIGS[lipsync_config.extend_audio],
        "extend_video": EXTEND_VIDEO_CONFIGS[lipsync_config.extend_video],
        "output_video_encoding": output_video_encoding,
        "is_speaker_info_provided": lipsync_config.is_speaker_info_provided,
    }

    # Build BackgroundAudioConfig when a background audio file is provided
    has_bg_audio = lipsync_config.background_audio_filepath is not None
    if has_bg_audio:
        # Construct a minimal namespace for _build_background_audio_config
        bg_args = argparse.Namespace(
            lipsync_background_audio_codec=None,
            lipsync_background_audio_volume=None,
        )
        bg_config = _build_background_audio_config(
            args=bg_args,
            file_path=str(lipsync_config.background_audio_filepath),
        )
        params["background_audio_config"] = bg_config

    # Send configuration
    yield lipsync_pb2.LipsyncRequest(config=lipsync_pb2.LipsyncConfig(**params))

    print("Sending data for inference")

    # Initialize file handles and state
    video_done = audio_done = False
    speaker_info_done = lipsync_config.speaker_info_filepath is None
    background_audio_done = not has_bg_audio

    with (
        open(lipsync_config.video_filepath, "rb") as video_file,
        open(lipsync_config.audio_filepath, "rb") as audio_file,
        (
            open(lipsync_config.speaker_info_filepath)
            if lipsync_config.speaker_info_filepath
            else nullcontext()
        ) as speaker_info_file,
        (
            open(lipsync_config.background_audio_filepath, "rb") if has_bg_audio else nullcontext()
        ) as background_audio_file,
    ):
        # Set up speaker info reader if file is provided
        speaker_info_iterator = None
        if speaker_info_file:
            speaker_info_reader = csv.reader(speaker_info_file)
            next(speaker_info_reader)  # Skip header row
            speaker_info_iterator = speaker_info_csv_reader(
                speaker_info_reader, SPEAKER_INFO_FRAME_COUNT
            )

        # Process data chunks
        chunk_counters = {
            "video": 0,
            "audio": 0,
            "speaker_info": 0,
            "background_audio": 0,
        }

        while not (video_done and audio_done and speaker_info_done and background_audio_done):
            # Send video chunk if not done
            if not video_done:
                try:
                    video_buffer = video_file.read(DATA_CHUNK_SIZE)
                    if video_buffer:
                        chunk_counters["video"] += 1
                        yield lipsync_pb2.LipsyncRequest(
                            input=lipsync_pb2.LipsyncInputData(video_file_data=video_buffer)
                        )
                    else:
                        video_done = True
                except OSError as e:
                    raise RuntimeError(f"Failed to read video file: {e}")

            # Send audio chunk if not done
            if not audio_done:
                try:
                    audio_buffer = audio_file.read(DATA_CHUNK_SIZE)
                    if audio_buffer:
                        chunk_counters["audio"] += 1
                        yield lipsync_pb2.LipsyncRequest(
                            input=lipsync_pb2.LipsyncInputData(audio_file_data=audio_buffer)
                        )
                    else:
                        audio_done = True
                except OSError as e:
                    raise RuntimeError(f"Failed to read audio file: {e}")

            # Send speaker info batch if not done
            if not speaker_info_done:
                assert speaker_info_iterator is not None
                try:
                    rows = next(speaker_info_iterator, None)
                    if rows is not None:
                        chunk_counters["speaker_info"] += 1
                        speaker_info_batch = group_rows_into_per_frame_infos(rows)
                        yield lipsync_pb2.LipsyncRequest(
                            input=lipsync_pb2.LipsyncInputData(
                                per_frame_speaker_infos=speaker_info_batch
                            )
                        )
                    else:
                        speaker_info_done = True
                except Exception as e:
                    raise RuntimeError(f"Failed to process speaker info data: {e}")

            # Send background audio chunk if not done
            if not background_audio_done:
                try:
                    bg_buffer = background_audio_file.read(DATA_CHUNK_SIZE)
                    if bg_buffer:
                        chunk_counters["background_audio"] += 1
                        yield lipsync_pb2.LipsyncRequest(
                            input=lipsync_pb2.LipsyncInputData(background_audio_file_data=bg_buffer)
                        )
                    else:
                        background_audio_done = True
                except OSError as e:
                    raise RuntimeError(f"Failed to read background audio file: {e}") from e

    print(
        f"Data sending completed - Video: {chunk_counters['video']}, "
        f"Audio: {chunk_counters['audio']}, "
        f"Speaker info: {chunk_counters['speaker_info']}, "
        f"Background audio: {chunk_counters['background_audio']} chunks"
    )
