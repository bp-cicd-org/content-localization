#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end functional test for the Direct client.

This test runs the actual Direct client and validates the complete pipeline:
1. Runs the direct client with sample inputs
2. Validates output video generation
3. Checks file formats and sizes
4. Verifies the direct service communication pipeline
"""

import subprocess
import sys
import time
from pathlib import Path

import pytest

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from client.direct.args import argsfactory
from client.utils import check_service_health


def test_services_health():
    """Check if all required services are running and healthy."""
    print("Checking service health...")

    services = [
        ("S2S", "localhost:50050"),
        ("LipSync", "localhost:50054"),
        ("ASD", "localhost:50055"),
    ]

    all_healthy = True
    for service_name, service_addr in services:
        try:
            check_service_health(service_addr)
            print(f"OK: {service_name} service is healthy")
        except Exception as e:
            pytest.fail(f"ERROR: {service_name} service not available: {e}")

    return all_healthy


def test_input_files_exist():
    """Check if required input files exist."""
    print("Checking input files...")

    project_root = Path(__file__).parent.parent
    defaults = argsfactory().parse_args([])
    audio_file = project_root / defaults.input_audio
    video_file = project_root / defaults.input_mp4

    if not audio_file.exists():
        pytest.fail(f"ERROR: Audio file not found: {audio_file}")

    if not video_file.exists():
        pytest.fail(f"ERROR: Video file not found: {video_file}")

    print(f"OK: Audio file found: {audio_file}")
    print(f"OK: Video file found: {video_file}")
    return True


def cleanup_previous_outputs():
    """Clean up any previous test outputs."""
    outputs_dir = Path(__file__).parent / "outputs"
    if outputs_dir.exists():
        for file in outputs_dir.glob("direct_*"):
            try:
                file.unlink()
                print(f"CLEANUP: Cleaned up previous output: {file.name}")
            except Exception as e:
                print(f"WARNING: Could not clean up {file.name}: {e}")


def test_direct_client_comprehensive(source_language, target_language, audio_format):
    """Comprehensive Direct client test covering basic and complex functionality."""
    print("\nStarting comprehensive Direct client test...")
    # Clean up previous outputs
    cleanup_previous_outputs()

    # Create outputs directory
    outputs_dir = Path(__file__).parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    output_file = outputs_dir / "direct_comprehensive_output.mp4"

    project_root = Path(__file__).parent.parent
    defaults = argsfactory().parse_args([])
    resolved_source_language = source_language or defaults.source_language
    resolved_target_language = target_language or defaults.target_language
    resolved_audio_format = audio_format or Path(defaults.output_audio).suffix.lstrip(".")
    if resolved_audio_format not in {"mp3", "wav"}:
        pytest.fail(f"ERROR: Unsupported audio format: {resolved_audio_format}")

    output_audio = outputs_dir / f"direct_comprehensive_audio.{resolved_audio_format}"
    audio_file = project_root / defaults.input_audio
    video_file = project_root / defaults.input_mp4

    # Build command with optimized chunk sizes for better streaming
    cmd = [
        sys.executable,
        "client/direct/app.py",
        "--s2s-server",
        "localhost:50050",
        "--lipsync-server",
        "localhost:50054",
        "--asd-server",
        "localhost:50055",
        "--input-audio",
        str(audio_file),
        "--input-mp4",
        str(video_file),
        "--output-mp4",
        str(output_file),
        "--output-audio",
        str(output_audio),
        "--source-language",
        resolved_source_language,
        "--target-language",
        resolved_target_language,
        "--chunk-size-audio-secs",
        "2.0",  # Optimized chunk size for better streaming
        "--chunk-size-video-bytes",
        "1048576",  # 1MB chunk size for optimal streaming
    ]

    print(f"Running comprehensive Direct test: {' '.join(cmd)}")

    # Run the direct client
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        end_time = time.time()
        processing_time = end_time - start_time

        if result.returncode != 0:
            pytest.fail(
                f"ERROR: Direct client failed with return code: {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )

        print(f"OK: Direct client completed successfully in {processing_time:.2f} seconds")
        print(f"STDOUT: {result.stdout}")

        # Validate video output
        if not output_file.exists():
            pytest.fail(f"ERROR: Output video file not created: {output_file}")

        if output_file.stat().st_size == 0:
            pytest.fail(f"ERROR: Output video file is empty: {output_file}")

        # Validate audio output
        if not output_audio.exists():
            pytest.fail(f"ERROR: Output audio file not created: {output_audio}")

        if output_audio.stat().st_size == 0:
            pytest.fail(f"ERROR: Output audio file is empty: {output_audio}")

        # Validate MP4 format for video
        with open(output_file, "rb") as f:
            header = f.read(12)
            if len(header) < 8:
                pytest.fail(f"ERROR: Output video file too small: {len(header)} bytes")

            # Check for MP4 signature patterns
            is_valid_mp4 = (
                header[4:8] == b"ftyp"  # ftyp atom
                or header[4:8] == b"moov"  # moov atom
                or header[4:8] == b"mdat"  # mdat atom
            )

            if not is_valid_mp4:
                pytest.fail(f"ERROR: Output video file is not valid MP4, header: {header}")

        # Validate audio format header based on requested audio_format
        if resolved_audio_format == "mp3":
            with open(output_audio, "rb") as f:
                header = f.read(10)
                if len(header) < 3:
                    pytest.fail(f"ERROR: Output audio file too small: {len(header)} bytes")
                is_valid_mp3 = header.startswith(b"ID3") or (
                    header[0] == 0xFF and (header[1] & 0xE0) == 0xE0
                )
                if not is_valid_mp3:
                    pytest.fail(f"ERROR: Output audio file is not valid MP3, header: {header[:10]}")
        else:
            with open(output_audio, "rb") as f:
                header = f.read(12)
                if len(header) < 12:
                    pytest.fail(f"ERROR: Output audio file too small: {len(header)} bytes")
                is_valid_wav = header[0:4] == b"RIFF" and header[8:12] == b"WAVE"
                if not is_valid_wav:
                    pytest.fail(f"ERROR: Output audio file is not valid WAV, header: {header[:12]}")

        # Check file sizes are reasonable.
        # LipSync re-encodes video at a low bitrate (default 3 Mbps) so the
        # output will be much smaller than the high-bitrate input (~20 Mbps).
        # S2S similarly re-encodes audio at a different bitrate.
        # Use absolute minimums instead of input-relative thresholds.
        output_video_size = output_file.stat().st_size
        output_audio_size = output_audio.stat().st_size
        min_video_bytes = 1 * 1024 * 1024  # 1 MB
        min_audio_bytes = 10 * 1024  # 10 KB

        if output_video_size < min_video_bytes:
            pytest.fail(
                f"ERROR: Output video seems too small: "
                f"{output_video_size} bytes (minimum: {min_video_bytes} bytes)"
            )

        if output_audio_size < min_audio_bytes:
            pytest.fail(
                f"ERROR: Output audio seems too small: "
                f"{output_audio_size} bytes (minimum: {min_audio_bytes} bytes)"
            )

        print(f"OK: Output video file created successfully: {output_file}")
        print(f"OK: Output video file size: {output_video_size} bytes")
        print("OK: Output video file is valid MP4 format")
        print(f"OK: Output audio file created successfully: {output_audio}")
        print(f"OK: Output audio file size: {output_audio_size} bytes")
        print(f"OK: Output audio file is valid {resolved_audio_format.upper()} format")
        print("OK: File size validation passed")
        print("OK: Comprehensive Direct test completed successfully")

    except subprocess.TimeoutExpired:
        pytest.fail("ERROR: Direct client timed out after 5 minutes")
    except Exception as e:
        pytest.fail(f"ERROR: Direct client failed with exception: {e}")


# Pytest will automatically discover and run all test_* functions
