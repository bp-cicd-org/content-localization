#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end functional test for the ASD client.

This test runs the actual ASD client and validates the speaker detection pipeline:
1. Runs the ASD client with sample video input
2. Validates output generation
3. Checks file formats and sizes
4. Verifies speaker detection functionality
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

from client.asd.args import argsfactory
from client.utils import check_service_health


def test_asd_service_health():
    """Check if the ASD service is running and healthy."""
    print("Checking ASD service health...")
    try:
        check_service_health("localhost:50055")
        print("OK: ASD service is healthy")
        return True
    except Exception as e:
        pytest.fail(f"ERROR: ASD service not available: {e}")


def test_input_files_exist():
    """Check if required input files exist."""
    print("Checking input files...")

    project_root = Path(__file__).parent.parent
    defaults = argsfactory().parse_args([])
    video_file = project_root / defaults.input_mp4

    if not video_file.exists():
        pytest.fail(f"ERROR: Video file not found: {video_file}")

    print(f"OK: Video file found: {video_file}")
    return True


def cleanup_previous_outputs():
    """Clean up any previous test outputs."""
    outputs_dir = Path(__file__).parent / "outputs"
    if outputs_dir.exists():
        for file in outputs_dir.glob("asd_*"):
            try:
                file.unlink()
                print(f"CLEANUP: Cleaned up previous output: {file.name}")
            except Exception as e:
                print(f"WARNING: Could not clean up {file.name}: {e}")


def test_asd_client_comprehensive():
    """Comprehensive ASD client test covering basic and complex functionality."""
    print("\nStarting comprehensive ASD client test...")

    # Clean up previous outputs
    cleanup_previous_outputs()

    # Create outputs directory
    outputs_dir = Path(__file__).parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    output_file = outputs_dir / "asd_comprehensive_output.csv"

    project_root = Path(__file__).parent.parent
    defaults = argsfactory().parse_args([])
    video_file = project_root / defaults.input_mp4

    # Build command with optimized chunk sizes for better streaming
    cmd = [
        sys.executable,
        "client/asd/app.py",
        "--asd-server",
        "localhost:50055",
        "--input-mp4",
        str(video_file),
        "--output-speaker-info",
        str(output_file),
        "--chunk-size-video-bytes",
        "1048576",  # 1MB chunk size for optimal streaming
    ]

    print(f"Running comprehensive ASD test: {' '.join(cmd)}")

    # Run the ASD client
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
                f"ERROR: ASD client failed with return code: {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )

        print(f"OK: ASD client completed successfully in {processing_time:.2f} seconds")
        print(f"STDOUT: {result.stdout}")

        # Validate output
        if not output_file.exists():
            pytest.fail(f"ERROR: Output file not created: {output_file}")

        if output_file.stat().st_size == 0:
            pytest.fail(f"ERROR: Output file is empty: {output_file}")

        # Validate CSV format and content
        try:
            with open(output_file, "r") as f:
                lines = f.readlines()

            if not lines:
                pytest.fail(f"ERROR: Output file is empty")

            # Check header structure
            header = lines[0].strip()
            if not header or "," not in header:
                pytest.fail(f"ERROR: Output file is not a valid CSV file")

            # Check for data content
            if len(lines) < 2:
                pytest.fail(f"ERROR: Output file has no data rows")

            # Validate data structure (should have speaker detection results)
            data_line = lines[1].strip()
            if not data_line or "," not in data_line:
                pytest.fail(f"ERROR: Output file has invalid data format")

            print(f"OK: Output file is valid CSV format with {len(lines)} lines")
            print(f"OK: Header: {header}")
            print(f"OK: Sample data: {data_line}")

        except Exception as e:
            pytest.fail(f"ERROR: Output file is not valid CSV: {e}")

        print(f"OK: Output file created successfully: {output_file}")
        print(f"OK: Output file size: {output_file.stat().st_size} bytes")
        print(f"OK: Comprehensive ASD test completed successfully")

    except subprocess.TimeoutExpired:
        pytest.fail("ERROR: ASD client timed out after 5 minutes")
    except Exception as e:
        pytest.fail(f"ERROR: ASD client failed with exception: {e}")


# Pytest will automatically discover and run all test_* functions
