#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Functional test for Magpie TTS Zeroshot.

This test validates that the Magpie TTS Zeroshot model is properly deployed and functional:
1. Checks if the TTS service is running and healthy
2. Queries available models and voices
3. Tests zero-shot voice cloning with audio prompt
4. Validates output audio generation
"""

import os
import sys
import wave
from pathlib import Path

import grpc
import pytest
import riva.client
import riva.client.proto.riva_tts_pb2 as rtts
import riva.client.proto.riva_tts_pb2_grpc as rtts_srv

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
TTS_SERVER = os.environ.get("MAGPIE_TTS_SERVER", "localhost:50053")
TTS_HTTP_SERVER = "http://localhost:9003"
TEST_TEXT = "Hello, this is a test of zero shot voice cloning from Magpie TTS."
AUDIO_PROMPT_FILE = os.environ.get("MAGPIE_AUDIO_PROMPT_FILE", "assets/sample_audio.wav")
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "magpie_zeroshot_test_output.wav"
LANGUAGE_CODE = "en-US"
SAMPLE_RATE = 16000


@pytest.fixture(scope="module", autouse=True)
def setup_output_dir():
    """Ensure output directory exists."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _fetch_tts_config_or_skip(timeout: float):
    try:
        channel = grpc.insecure_channel(TTS_SERVER)
        stub = rtts_srv.RivaSpeechSynthesisStub(channel)
        request = rtts.RivaSynthesisConfigRequest(model_name="")
        return stub.GetRivaSynthesisConfig(request, timeout=timeout)
    except Exception as e:
        pytest.skip(f"TTS service not available: {type(e).__name__}: {e}")


def _get_audio_prompt_or_skip() -> Path:
    audio_file = project_root / AUDIO_PROMPT_FILE
    if not audio_file.exists():
        pytest.skip(f"Audio prompt file not found: {audio_file}")
    return audio_file


@pytest.mark.functional
def test_tts_service_health():
    """Check if the TTS service is running and responding."""
    print(f"\nChecking TTS service health at {TTS_SERVER}...")

    response = _fetch_tts_config_or_skip(timeout=5.0)
    print("✓ TTS service is healthy")
    print(f"✓ Found {len(response.model_config)} model configuration(s)")


@pytest.mark.functional
def test_model_availability():
    """Verify that Magpie TTS Zeroshot model is loaded."""
    print(f"\nQuerying model availability...")

    response = _fetch_tts_config_or_skip(timeout=10.0)
    if len(response.model_config) == 0:
        pytest.fail("✗ No models loaded on TTS server! Check model deployment.")

    # Check for Magpie-ZeroShot model
    model_found = False
    for config in response.model_config:
        print(f"\n  Model: {config.model_name}")
        if "Magpie-ZeroShot" in config.model_name or "magpie" in config.model_name.lower():
            model_found = True

            # Print important parameters
            if config.parameters:
                important_params = [
                    "voice_name",
                    "subvoices",
                    "zero_shot_sample_rate",
                    "is_magpie_tts",
                    "language_code",
                ]
                for key in important_params:
                    if key in config.parameters:
                        print(f"    - {key}: {config.parameters[key]}")

    if not model_found:
        pytest.fail("✗ Magpie TTS Zeroshot model not found on server!")

    print("\n✓ Magpie TTS Zeroshot model is loaded and available")


@pytest.mark.functional
def test_input_file_exists():
    """Check if the audio prompt file exists."""
    print(f"\nChecking input files...")

    audio_file = _get_audio_prompt_or_skip()
    print(f"✓ Audio prompt file found: {audio_file}")


@pytest.mark.functional
@pytest.mark.slow
def test_zero_shot_synthesis():
    """Test zero-shot voice cloning with audio prompt."""
    print(f"\n{'=' * 60}")
    print("Testing Zero-Shot Voice Cloning")
    print(f"{'=' * 60}")

    try:
        _fetch_tts_config_or_skip(timeout=5.0)

        # Setup Riva TTS client
        auth = riva.client.Auth(uri=TTS_SERVER)
        tts_service = riva.client.SpeechSynthesisService(auth=auth)

        # Read audio prompt
        audio_file = _get_audio_prompt_or_skip()
        with wave.open(str(audio_file), "rb") as wav:
            sample_rate = wav.getframerate()
            total_frames = wav.getnframes()
            audio_data = wav.readframes(total_frames)
            duration = total_frames / sample_rate

        print(f"\nAudio prompt: {len(audio_data)} bytes, {duration:.2f}s at {sample_rate} Hz")

        if duration < 3.0 or duration > 10.0:
            print(f"⚠ Warning: Audio prompt duration ({duration:.2f}s) should be 3-10 seconds")

        # Build proto request directly (required for zero_shot_data.sample_rate_hz)
        req = rtts.SynthesizeSpeechRequest(
            text=TEST_TEXT,
            language_code=LANGUAGE_CODE,
            sample_rate_hz=SAMPLE_RATE,
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
        )
        # Do NOT set voice_name for pure zero-shot

        # Set zero-shot data with sample_rate_hz (CRITICAL)
        req.zero_shot_data.audio_prompt = audio_data
        req.zero_shot_data.sample_rate_hz = sample_rate  # Must match audio prompt
        req.zero_shot_data.encoding = riva.client.AudioEncoding.LINEAR_PCM
        req.zero_shot_data.quality = 20

        print(f"\nSynthesizing text: '{TEST_TEXT}'")
        print(f"Parameters:")
        print(f"  - voice_name: None (pure zero-shot)")
        print(f"  - language_code: {LANGUAGE_CODE}")
        print(f"  - zero_shot_data.sample_rate_hz: {sample_rate}")
        print(f"  - zero_shot_quality: 20")

        # Call TTS
        response = tts_service.stub.SynthesizeOnline(
            req, metadata=tts_service.auth.get_auth_metadata()
        )

        # Collect audio chunks
        audio_chunks = []
        print("\nGenerating audio", end="", flush=True)
        for chunk in response:
            audio_chunks.append(chunk.audio)
            print(".", end="", flush=True)

        print(f"\n✓ Generated {len(audio_chunks)} audio chunks")

        if len(audio_chunks) == 0:
            pytest.fail("✗ No audio chunks generated!")

        # Write output
        with wave.open(str(OUTPUT_FILE), "wb") as wav_out:
            wav_out.setnchannels(1)
            wav_out.setsampwidth(2)  # 16-bit
            wav_out.setframerate(SAMPLE_RATE)
            for chunk in audio_chunks:
                wav_out.writeframes(chunk)

        # Validate output
        if not OUTPUT_FILE.exists():
            pytest.fail(f"✗ Output file was not created: {OUTPUT_FILE}")

        file_size = OUTPUT_FILE.stat().st_size
        if file_size == 0:
            pytest.fail(f"✗ Output file is empty: {OUTPUT_FILE}")

        print(f"✓ Output written to: {OUTPUT_FILE}")
        print(f"✓ Output size: {file_size:,} bytes")

        # Validate WAV format
        with wave.open(str(OUTPUT_FILE), "rb") as wav_verify:
            output_frames = wav_verify.getnframes()
            output_duration = output_frames / wav_verify.getframerate()
            print(f"✓ Output duration: {output_duration:.2f}s ({output_frames} frames)")

        print(f"\n{'=' * 60}")
        print("✓ Zero-Shot Voice Cloning Test PASSED")
        print(f"{'=' * 60}")
        return True

    except Exception as e:
        import traceback

        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        traceback.print_exc()
        pytest.fail(f"Zero-shot synthesis failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    """Allow running tests directly for debugging."""
    print("Running Magpie TTS Zeroshot Functional Tests")
    print("=" * 60)

    # Run tests individually for better visibility
    try:
        test_tts_service_health()
        test_input_file_exists()
        test_model_availability()
        test_zero_shot_synthesis()

        print("\n" + "=" * 60)
        print("ALL FUNCTIONAL TESTS PASSED!")
        print("=" * 60)
        sys.exit(0)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"FUNCTIONAL TEST FAILED: {e}")
        print("=" * 60)
        sys.exit(1)
