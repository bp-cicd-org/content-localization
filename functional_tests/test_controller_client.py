#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end functional test for the Controller client.

This test runs the actual Controller client and validates the complete pipeline:
1. Runs the controller client with sample inputs
2. Validates output video generation
3. Checks file formats and sizes
4. Verifies the complete content localization pipeline
"""

import subprocess
import sys
import time
from pathlib import Path

import pytest

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from client.controller.args import argsfactory
from client.utils import check_service_health


def test_controller_service_health():
    """Check if the Controller service is running and healthy."""
    print("Checking Controller service health...")
    try:
        check_service_health("localhost:50056")
        print("OK: Controller service is healthy")
        return True
    except Exception as e:
        pytest.fail(f"ERROR: Controller service not available: {e}")


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
        for file in outputs_dir.glob("controller_*"):
            try:
                file.unlink()
                print(f"CLEANUP: Cleaned up previous output: {file.name}")
            except Exception as e:
                print(f"WARNING: Could not clean up {file.name}: {e}")


def test_controller_client_comprehensive(source_language, target_language):
    """Comprehensive Controller client test covering basic and complex functionality."""
    print("\nStarting comprehensive Controller client test...")

    # Clean up previous outputs
    cleanup_previous_outputs()

    # Create outputs directory
    outputs_dir = Path(__file__).parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    output_file = outputs_dir / "controller_comprehensive_output.mp4"

    project_root = Path(__file__).parent.parent
    defaults = argsfactory().parse_args([])
    resolved_source_language = source_language or defaults.source_language
    resolved_target_language = target_language or defaults.target_language
    audio_file = project_root / defaults.input_audio
    video_file = project_root / defaults.input_mp4

    # Build command with optimized chunk sizes for better streaming
    cmd = [
        sys.executable,
        "client/controller/app.py",
        "--controller-server",
        "localhost:50056",
        "--input-audio",
        str(audio_file),
        "--input-mp4",
        str(video_file),
        "--output-mp4",
        str(output_file),
        "--source-language",
        resolved_source_language,
        "--target-language",
        resolved_target_language,
        "--chunk-size-audio-secs",
        "2.0",  # Optimized chunk size for better streaming
        "--chunk-size-video-bytes",
        "1048576",  # 1MB chunk size for optimal streaming
    ]

    print(f"Running comprehensive Controller test: {' '.join(cmd)}")

    # Run the controller client
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
                f"ERROR: Controller client failed with return code: {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )

        print(f"OK: Controller client completed successfully in {processing_time:.2f} seconds")
        print(f"STDOUT: {result.stdout}")

        # Validate output
        if not output_file.exists():
            pytest.fail(f"ERROR: Output file not created: {output_file}")

        if output_file.stat().st_size == 0:
            pytest.fail(f"ERROR: Output file is empty: {output_file}")

        # Validate MP4 format and content
        with open(output_file, "rb") as f:
            header = f.read(12)
            if len(header) < 8:
                pytest.fail(f"ERROR: Output file too small: {len(header)} bytes")

            # Check for MP4 signature patterns
            is_valid_mp4 = (
                header[4:8] == b"ftyp"  # ftyp atom
                or header[4:8] == b"moov"  # moov atom
                or header[4:8] == b"mdat"  # mdat atom
            )

            if not is_valid_mp4:
                pytest.fail(f"ERROR: Output file is not valid MP4, header: {header}")

        # Check file size is reasonable.
        # LipSync re-encodes at a low bitrate (default 3 Mbps) so the output
        # will be much smaller than the original high-bitrate input (~20 Mbps).
        # Use a 1 MB absolute minimum instead of an input-relative threshold.
        output_size = output_file.stat().st_size
        min_output_bytes = 1 * 1024 * 1024  # 1 MB

        if output_size < min_output_bytes:
            pytest.fail(
                f"ERROR: Output file seems too small: {output_size} bytes "
                f"(minimum: {min_output_bytes} bytes)"
            )

        print(f"OK: Output file created successfully: {output_file}")
        print(f"OK: Output file size: {output_size} bytes")
        print("OK: Output file is valid MP4 format")
        print(f"OK: File size validation passed (output: {output_size} bytes)")
        print("OK: Comprehensive Controller test completed successfully")

    except subprocess.TimeoutExpired:
        pytest.fail("ERROR: Controller client timed out after 5 minutes")
    except Exception as e:
        pytest.fail(f"ERROR: Controller client failed with exception: {e}")


# Pytest will automatically discover and run all test_* functions
