#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end functional test for the S2S client.

This test runs the actual S2S client and validates the audio translation pipeline:
1. Runs the S2S client with sample audio input
2. Validates output audio generation
3. Checks file formats and sizes
4. Verifies audio translation functionality
"""

import subprocess
import sys
import time
from pathlib import Path

import pytest

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from client.s2s.args import argsfactory
from client.utils import check_service_health


def test_s2s_service_health():
    """Check if the S2S service is running and healthy."""
    print("Checking S2S service health...")
    try:
        check_service_health("localhost:50050")
        print("OK: S2S service is healthy")
        return True
    except Exception as e:
        pytest.fail(f"ERROR: S2S service not available: {e}")


def test_input_files_exist():
    """Check if required input files exist."""
    print("Checking input files...")

    project_root = Path(__file__).parent.parent
    defaults = argsfactory().parse_args([])
    audio_file = project_root / defaults.input_audio

    if not audio_file.exists():
        pytest.fail(f"ERROR: Audio file not found: {audio_file}")

    print(f"OK: Audio file found: {audio_file}")
    return True


def cleanup_previous_outputs():
    """Clean up any previous test outputs."""
    outputs_dir = Path(__file__).parent / "outputs"
    if outputs_dir.exists():
        for file in outputs_dir.glob("s2s_*"):
            try:
                file.unlink()
                print(f"CLEANUP: Cleaned up previous output: {file.name}")
            except Exception as e:
                print(f"WARNING: Could not clean up {file.name}: {e}")


def test_s2s_client_comprehensive(source_language, target_language, audio_format):
    """Comprehensive S2S client test covering basic and complex functionality."""
    print("\nStarting comprehensive S2S client test...")

    # Clean up previous outputs
    cleanup_previous_outputs()

    # Create outputs directory
    outputs_dir = Path(__file__).parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    project_root = Path(__file__).parent.parent
    defaults = argsfactory().parse_args([])
    resolved_source_language = source_language or defaults.source_language
    resolved_target_language = target_language or defaults.target_language
    resolved_audio_format = audio_format or Path(defaults.output_audio).suffix.lstrip(".")
    if resolved_audio_format not in {"mp3", "wav"}:
        pytest.fail(f"ERROR: Unsupported audio format: {resolved_audio_format}")

    output_file = outputs_dir / f"s2s_comprehensive_output.{resolved_audio_format}"
    latency_plot = outputs_dir / "s2s_comprehensive_latency_plot.png"
    audio_file = project_root / defaults.input_audio

    # Build command with latency analysis and optimized chunk sizes
    cmd = [
        sys.executable,
        "client/s2s/app.py",
        "--s2s-server",
        "localhost:50050",
        "--input-audio",
        str(audio_file),
        "--output-audio",
        str(output_file),
        "--latency-plot",
        str(latency_plot),
        "--source-language",
        resolved_source_language,
        "--target-language",
        resolved_target_language,
        "--chunk-size-audio-secs",
        "1.0",  # Optimized chunk size for better streaming
    ]

    print(f"Running comprehensive S2S test: {' '.join(cmd)}")

    # Run the S2S client
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
                f"ERROR: S2S client failed with return code: {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )

        print(f"OK: S2S client completed successfully in {processing_time:.2f} seconds")
        print(f"STDOUT: {result.stdout}")

        # Validate audio output
        if not output_file.exists():
            pytest.fail(f"ERROR: Output audio file not created: {output_file}")

        if output_file.stat().st_size == 0:
            pytest.fail(f"ERROR: Output audio file is empty: {output_file}")

        # Validate format by header based on requested audio_format
        if resolved_audio_format == "mp3":
            with open(output_file, "rb") as f:
                header = f.read(10)
                if len(header) < 3:
                    pytest.fail(f"ERROR: Output file too small: {len(header)} bytes")
                is_valid_mp3 = header.startswith(b"ID3") or (
                    header[0] == 0xFF and (header[1] & 0xE0) == 0xE0
                )
                if not is_valid_mp3:
                    pytest.fail(f"ERROR: Output file is not valid MP3, header: {header[:10]}")
        else:
            with open(output_file, "rb") as f:
                header = f.read(12)
                if len(header) < 12:
                    pytest.fail(f"ERROR: Output file too small: {len(header)} bytes")
                is_valid_wav = header[0:4] == b"RIFF" and header[8:12] == b"WAVE"
                if not is_valid_wav:
                    pytest.fail(f"ERROR: Output file is not valid WAV, header: {header[:12]}")

        # Validate latency plot
        if not latency_plot.exists():
            pytest.fail(f"ERROR: Latency plot not created: {latency_plot}")

        if latency_plot.stat().st_size == 0:
            pytest.fail(f"ERROR: Latency plot is empty: {latency_plot}")

        # Validate PNG format for latency plot
        with open(latency_plot, "rb") as f:
            png_header = f.read(8)
            if png_header != b"\x89PNG\r\n\x1a\n":
                pytest.fail("ERROR: Latency plot is not valid PNG format")

        print(f"OK: Output audio file created successfully: {output_file}")
        print(f"OK: Output audio file size: {output_file.stat().st_size} bytes")
        print(f"OK: Output audio file is valid {resolved_audio_format.upper()} format")
        print(f"OK: Latency plot created successfully: {latency_plot}")
        print(f"OK: Latency plot size: {latency_plot.stat().st_size} bytes")
        print("OK: Comprehensive S2S test completed successfully")

    except subprocess.TimeoutExpired:
        pytest.fail("ERROR: S2S client timed out after 5 minutes")
    except Exception as e:
        pytest.fail(f"ERROR: S2S client failed with exception: {e}")


# Pytest will automatically discover and run all test_* functions
