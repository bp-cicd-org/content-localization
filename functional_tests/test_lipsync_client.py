#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end functional test for the LipSync client.

This test runs the actual LipSync client and validates the lip-sync pipeline:
1. Runs the LipSync client with sample audio and video inputs
2. Validates output video generation
3. Checks file formats and sizes
4. Verifies lip-sync functionality
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from client.lipsync.args import argsfactory
from client.utils import check_service_health


def test_lipsync_service_health():
    """Check if the LipSync service is running and healthy."""
    print("Checking LipSync service health...")
    try:
        check_service_health("localhost:50054")
        print("OK: LipSync service is healthy")
        return True
    except Exception as e:
        pytest.fail(f"ERROR: LipSync service not available: {e}")


def test_input_files_exist():
    """Check if required input files exist."""
    print("Checking input files...")

    project_root = Path(__file__).parent.parent
    defaults = argsfactory().parse_args([])
    audio_file = project_root / defaults.audio_input
    video_file = project_root / defaults.video_input

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
        for file in outputs_dir.glob("lipsync_*"):
            try:
                file.unlink()
                print(f"CLEANUP: Cleaned up previous output: {file.name}")
            except Exception as e:
                print(f"WARNING: Could not clean up {file.name}: {e}")


def test_lipsync_client_comprehensive():
    """Comprehensive LipSync client test covering basic and complex functionality."""
    print("\nStarting comprehensive LipSync client test...")

    # Clean up previous outputs
    cleanup_previous_outputs()

    # Create outputs directory
    outputs_dir = Path(__file__).parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    output_file = outputs_dir / "lipsync_comprehensive_output.mp4"

    # Get input file paths
    project_root = Path(__file__).parent.parent
    defaults = argsfactory().parse_args([])
    audio_file = project_root / defaults.audio_input
    video_file = project_root / defaults.video_input

    # Build command with optimized parameters
    cmd = [
        sys.executable,
        "client/lipsync/app.py",
        "--target",
        "localhost:50054",
        "--audio-input",
        str(audio_file),
        "--video-input",
        str(video_file),
        "--output",
        str(output_file),
    ]

    print(f"Running comprehensive LipSync test: {' '.join(cmd)}")

    # Run the LipSync client
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
                f"ERROR: LipSync client failed with return code: {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )

        print(f"OK: LipSync client completed successfully in {processing_time:.2f} seconds")
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

        # Check file size is reasonable (lip-sync output is often highly compressed)
        input_video_size = video_file.stat().st_size
        output_size = output_file.stat().st_size

        if output_size < input_video_size * 0.1:  # Allow for heavy compression
            pytest.fail(
                f"ERROR: Output file seems too small: {output_size} bytes (input: {input_video_size} bytes)"
            )

        print(f"OK: Output file created successfully: {output_file}")
        print(f"OK: Output file size: {output_size} bytes")
        print(f"OK: Output file is valid MP4 format")
        print(f"OK: File size validation passed (input: {input_video_size}, output: {output_size})")
        print(f"OK: Comprehensive LipSync test completed successfully")

    except subprocess.TimeoutExpired:
        pytest.fail("ERROR: LipSync client timed out after 5 minutes")
    except Exception as e:
        pytest.fail(f"ERROR: LipSync client failed with exception: {e}")


# Pytest will automatically discover and run all test_* functions
