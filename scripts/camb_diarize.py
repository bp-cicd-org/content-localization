#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Generate diarization data using Camb AI Transcription API.
# Outputs the native Camb AI transcription JSON response for direct reuse.
#
# Usage:
#   CAMB_API_KEY=<key> .venv/bin/python scripts/camb_diarize.py \
#       --input-file <path>.wav \
#       --output-file diarization.json

import argparse
import json
import os
import time
import wave

import requests

CAMB_API_BASE_URL = "https://client.camb.ai/apis"
DEFAULT_POLL_INTERVAL = 10
DEFAULT_MAX_ATTEMPTS = 120


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Camb AI diarization.

    Returns:
        argparse.Namespace: Parsed CLI values with these defaults:
            - ``input_file``: required
            - ``output_file``: ``"diarization.json"``
            - ``language_id``: ``1`` (English)

    Examples:
        >>> # python scripts/camb_diarize.py --input-file audio.wav
    """
    parser = argparse.ArgumentParser(
        description="Generate diarization data using Camb AI Transcription API. "
        "Outputs native Camb AI transcription JSON with word-level timestamps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to audio file (WAV, MP3, etc.).",
    )
    parser.add_argument(
        "--output-file",
        default="diarization.json",
        help="Path to output JSON diarization file.",
    )
    parser.add_argument(
        "--language-id",
        type=int,
        default=1,
        help="Camb AI numeric language ID (1=English, 54=Spanish, etc.).",
    )
    return parser.parse_args()


def print_wav_info(wav_path: str) -> None:
    """Print WAV file metadata if the input is a WAV file.

    Silently skips non-WAV files.

    Args:
        wav_path (str): Path to the audio file.

    Returns:
        None.

    Examples:
        >>> print_wav_info("audio.wav")
        WAV: 16000 Hz, 1 ch, 160000 frames (10.00 s)
    """
    try:
        with wave.open(wav_path, "rb") as wav:
            nch = wav.getnchannels()
            framerate = wav.getframerate()
            nframes = wav.getnframes()
            total_sec = nframes / float(framerate)
            print(f"WAV: {framerate} Hz, {nch} ch, {nframes} frames ({total_sec:.2f} s)")
    except wave.Error:
        pass


def submit_transcription(
    file_path: str,
    language_id: int,
    headers: dict[str, str],
) -> str:
    """Submit a transcription request to Camb AI.

    Args:
        file_path (str): Path to the audio file to transcribe.
        language_id (int): Camb AI numeric language ID.
        headers (dict[str, str]): HTTP headers including ``x-api-key``.

    Returns:
        str: Task ID for polling.

    Raises:
        requests.HTTPError: If the endpoint returns a non-2xx response.
        RuntimeError: If the response does not contain a task_id.

    Examples:
        >>> submit_transcription("audio.wav", 1, {"x-api-key": "k"})
        'task_123'
    """
    with open(file_path, "rb") as f:
        response = requests.post(
            f"{CAMB_API_BASE_URL}/transcribe",
            headers=headers,
            files={"media_file": (os.path.basename(file_path), f)},
            data={"language": str(language_id)},
            timeout=120,
        )
    response.raise_for_status()
    data = response.json()
    task_id = data.get("task_id")
    if not task_id:
        raise RuntimeError(f"Camb AI /transcribe response missing task_id: {data}")
    return str(task_id)


def wait_for_transcription(
    task_id: str,
    headers: dict[str, str],
) -> int:
    """Poll Camb AI transcription status until SUCCESS.

    Args:
        task_id (str): Task ID from ``submit_transcription``.
        headers (dict[str, str]): HTTP headers including ``x-api-key``.

    Returns:
        int: Run ID when task reaches SUCCESS.

    Raises:
        requests.HTTPError: If the status endpoint returns a non-2xx response.
        RuntimeError: If Camb AI returns a terminal error status.
        TimeoutError: If polling exceeds max attempts.

    Examples:
        >>> wait_for_transcription("task_123", {"x-api-key": "k"})
        42
    """
    for attempt in range(DEFAULT_MAX_ATTEMPTS):
        response = requests.get(
            f"{CAMB_API_BASE_URL}/transcribe/{task_id}",
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        status = str(data.get("status", "")).upper()
        print(f"  Poll attempt {attempt + 1}: status={status}")

        if status == "SUCCESS":
            run_id = data.get("run_id")
            if not isinstance(run_id, int):
                raise RuntimeError(f"Camb AI status missing run_id on SUCCESS: {data}")
            return run_id
        if status in {"ERROR", "TIMEOUT", "PAYMENT_REQUIRED"}:
            message = data.get("message")
            raise RuntimeError(f"Camb AI transcription failed: status={status}, message={message}")

        time.sleep(DEFAULT_POLL_INTERVAL)

    raise TimeoutError(
        f"Camb AI transcription timed out after {DEFAULT_MAX_ATTEMPTS} attempts "
        f"(interval={DEFAULT_POLL_INTERVAL}s)."
    )


def get_transcription_result(
    run_id: int,
    headers: dict[str, str],
) -> list[dict]:
    """Fetch the transcription result with word-level timestamps.

    The Camb AI API wraps the segment list inside
    ``{"transcript": [...]}``.  This function unwraps it and returns
    the inner list directly.

    Args:
        run_id (int): Run ID from ``wait_for_transcription``.
        headers (dict[str, str]): HTTP headers including ``x-api-key``.

    Returns:
        list[dict]: List of transcription segments with start, end, text, speaker.

    Raises:
        requests.HTTPError: If the endpoint returns a non-2xx response.
        RuntimeError: If the response does not contain a ``transcript`` key.

    Examples:
        >>> get_transcription_result(42, {"x-api-key": "k"})
        [{'start': 0.0, 'end': 1.5, 'text': 'hello', 'speaker': 'SPEAKER_0'}]
    """
    response = requests.get(
        f"{CAMB_API_BASE_URL}/transcription-result/{run_id}",
        headers=headers,
        params={"word_level_timestamps": "true"},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    # Camb AI wraps segments inside {"transcript": [...]}
    if isinstance(data, dict) and "transcript" in data:
        return data["transcript"]
    return data


def extract_diarization_stats(data: list[dict]) -> tuple[int, int]:
    """Extract segment and speaker counts from Camb AI transcription result.

    Args:
        data (list[dict]): List of transcription segments.

    Returns:
        tuple[int, int]: Tuple of (segment_count, unique_speaker_count).

    Examples:
        >>> extract_diarization_stats([])
        (0, 0)
        >>> extract_diarization_stats(
        ...     [
        ...         {"start": 0, "end": 1, "text": "hi", "speaker": "Speaker 1"},
        ...         {"start": 1, "end": 2, "text": "bye", "speaker": "Speaker 2"},
        ...     ]
        ... )
        (2, 2)
    """
    speakers: set[str] = set()
    for seg in data:
        speaker = seg.get("speaker")
        if speaker is not None:
            speakers.add(str(speaker))
    return len(data), len(speakers)


def main() -> None:
    """Run Camb AI diarization and write native JSON output.

    Reads ``CAMB_API_KEY`` from the environment, sends the input
    audio file through the Camb AI Transcription API, and writes
    the native JSON response to disk.

    Returns:
        None.

    Raises:
        ValueError: If ``CAMB_API_KEY`` is not set.
        FileNotFoundError: If the input file does not exist.

    Examples:
        >>> # python scripts/camb_diarize.py --input-file audio.wav -o out.json
    """
    args = parse_args()

    api_key = os.getenv("CAMB_API_KEY")
    if not api_key:
        raise ValueError(
            "CAMB_API_KEY environment variable not set. "
            "Export it or pass via: CAMB_API_KEY=<key> python scripts/camb_diarize.py ..."
        )

    if not os.path.isfile(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    headers = {"x-api-key": api_key}

    print_wav_info(args.input_file)

    file_size = os.path.getsize(args.input_file)
    print(f"Input file: {args.input_file} ({file_size:,} bytes)")
    print(f"Config: language_id={args.language_id}")

    print(f"Submitting {args.input_file} to Camb AI Transcription API...")
    task_id = submit_transcription(
        file_path=args.input_file,
        language_id=args.language_id,
        headers=headers,
    )
    print(f"Task submitted: task_id={task_id}")

    print("Polling for transcription completion...")
    run_id = wait_for_transcription(task_id=task_id, headers=headers)
    print(f"Transcription completed: run_id={run_id}")

    print("Fetching transcription result with word-level timestamps...")
    result = get_transcription_result(run_id=run_id, headers=headers)

    segment_count, speaker_count = extract_diarization_stats(result)
    if segment_count == 0:
        print("WARNING: No segments found in Camb AI response.")
    else:
        print(f"Generated {segment_count} segments across {speaker_count} speakers")

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Native Camb AI diarization JSON written to {args.output_file}")


if __name__ == "__main__":
    main()
