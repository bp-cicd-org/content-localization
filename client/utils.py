# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for S2S client."""

import argparse
import csv
import io
import itertools
import os
import wave
from collections.abc import Iterator

import grpc
from google.protobuf import any_pb2
from google.protobuf import wrappers_pb2
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc


def create_wav_header(n_channels: int, sample_width: int, frame_rate: int, n_frames: int) -> bytes:
    """Create a dummy header for a WAV file.

    Args:
        n_channels (int): The number of channels in the WAV file.
        sample_width (int): The sample width in bytes, usually 2 for 16-bit PCM.
        frame_rate (int): The frame rate in Hz. (sample rate).
        n_frames (int): The number of frames in the WAV file.

    Returns:
        bytes: The header of the WAV file created from wav file parameters.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, mode="wb") as wf:
        wf.setnchannels(nchannels=n_channels)
        wf.setsampwidth(sampwidth=sample_width)
        wf.setframerate(framerate=frame_rate)
        wf.setnframes(nframes=n_frames)
    return buffer.getvalue()


def check_service_health(server: str) -> bool:
    """Check gRPC health using the standard gRPC health checking protocol.

    Args:
        server (str): The gRPC server address (host:port).

    Returns:
        bool: True if the service is healthy, raises ConnectionError otherwise.
    """
    channel = grpc.insecure_channel(server)
    stub = health_pb2_grpc.HealthStub(channel)
    try:
        response = stub.Check(health_pb2.HealthCheckRequest(service=""))
        if response.status == health_pb2.HealthCheckResponse.SERVING:
            return True
        else:
            raise ConnectionError(f"Service not healthy: {response.status}")
    except grpc.RpcError as e:
        raise ConnectionError(f"Health check failed for {server}: {e}")


def is_file_available(file_path: os.PathLike, file_types: list[str]) -> bool:
    """Check if the file exists and has one of the specified file types.

    Args:
        file_path: Path to input file
        file_types: List of allowed file extensions (without the dot, e.g., ['txt', 'csv'])

    Returns:
        True if file exists and has one of the allowed extensions, False otherwise

    Raises:
        FileNotFoundError: If the file does not exist
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found")
    for file_type in file_types:
        if os.path.splitext(file_path)[1].lower() == f".{file_type}":
            return True
    return False


def read_file_content(file_path: os.PathLike) -> bytes:
    """Read file content as bytes.

    Args:
        file_path: Path to input file

    Returns:
        File contents as bytes
    """
    with open(file_path, "rb") as file:
        return file.read()


def speaker_info_csv_reader(reader: csv.reader, row_count: int) -> Iterator[list]:
    """Read CSV data in batches of multiple rows.

    Args:
        reader: CSV reader object to read from
        row_count: Number of rows to include in each batch

    Yields:
        List of CSV rows in batches of the specified row count
    """
    while True:
        rows = list(itertools.islice(reader, row_count))
        if not rows:
            break
        yield rows


def check_streamable(file_path: os.PathLike) -> bool:
    """Checks if the video is streamable by checking if the moov atom follows immediately after
    the ftyp atom in an MP4 file.

    For streamable MP4s, the moov atom must come immediately after:
    [4 bytes: size][4 bytes: "ftyp"][... ftyp data ...]
    [4 bytes: size][4 bytes: "moov"][... moov data ...]

    For non-streamable MP4s, other atoms like mdat may come between ftyp and moov:
    [4 bytes: size][4 bytes: "ftyp"][... ftyp data ...][4 bytes: size][4 bytes: "mdat"]
    [... mdat data ...][moov atom]

    Args:
        file_path: Path to the MP4 file to check

    Returns:
        bool: True if the file is streamable, False otherwise
    """
    # Read first 40 bytes of the file
    with open(file_path, "rb") as f:
        mp4_header_data = f.read(40)
        if len(mp4_header_data) < 40:
            return False

    # Read the first atom size
    ftyp_size = int.from_bytes(mp4_header_data[0:4], byteorder="big")

    # Check if it's a ftyp atom
    if mp4_header_data[4:8] != b"ftyp":
        return False

    next_atom_type = bytes(mp4_header_data[ftyp_size + 4 : ftyp_size + 8])

    # Check if the next atom is a moov atom
    return next_atom_type == b"moov"


def create_channel_credentials(args: argparse.Namespace) -> grpc.ChannelCredentials:
    """Create channel credentials based on SSL mode.

    Args:
        args: Command line arguments containing SSL configuration

    Returns:
        Configured channel credentials

    Raises:
        RuntimeError: If required SSL files are missing
    """
    channel_credentials = None
    if args.ssl_mode == "MTLS":
        if not (args.ssl_key and args.ssl_cert and args.ssl_root_cert):
            raise RuntimeError(
                "If --ssl-mode is MTLS, --ssl-key, --ssl-cert and --ssl-root-cert are required."
            )
        private_key = read_file_content(args.ssl_key)
        certificate_chain = read_file_content(args.ssl_cert)
        root_certificates = read_file_content(args.ssl_root_cert)
        channel_credentials = grpc.ssl_channel_credentials(
            root_certificates=root_certificates,
            private_key=private_key,
            certificate_chain=certificate_chain,
        )
    else:
        if not (args.ssl_root_cert):
            raise RuntimeError("If --ssl-mode is TLS, --ssl-root-cert is required.")
        root_certificates = read_file_content(args.ssl_root_cert)
        channel_credentials = grpc.ssl_channel_credentials(root_certificates=root_certificates)
    return channel_credentials


def create_protobuf_any_value(value: bool | int | float | str) -> any_pb2.Any:
    """Create a google.protobuf.Any message from a Python value.

    Args:
        value: The value to convert (bool, int, float, or str)

    Returns:
        google.protobuf.Any message
    """
    any_message = any_pb2.Any()

    if isinstance(value, bool):
        wrapper = wrappers_pb2.BoolValue(value=value)
        any_message.Pack(wrapper)
    elif isinstance(value, int):
        if value > 2147483647 or value < -2147483648:  # int32 range
            wrapper = wrappers_pb2.Int64Value(value=value)
        else:
            wrapper = wrappers_pb2.Int32Value(value=value)
        any_message.Pack(wrapper)
    elif isinstance(value, float):
        wrapper = wrappers_pb2.FloatValue(value=value)
        any_message.Pack(wrapper)
    elif isinstance(value, str):
        wrapper = wrappers_pb2.StringValue(value=value)
        any_message.Pack(wrapper)
    else:
        raise ValueError(f"Unsupported type: {type(value)}")

    return any_message
