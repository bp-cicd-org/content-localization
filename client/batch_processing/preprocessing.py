# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video preprocessing: audio extraction and duration probing."""

import os
import subprocess


def get_video_duration(video_path: str) -> float:
    """Get duration of a video file in seconds using ffprobe.

    Args:
        video_path (str): Path to the video file.

    Returns:
        float: Duration in seconds.

    Raises:
        RuntimeError: If ffprobe fails or returns invalid output.

    Examples:
        >>> dur = get_video_duration("sample.mp4")
        >>> dur > 0
        True
    """
    result = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {result.stderr.strip()}")
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid duration from ffprobe for {video_path}: {result.stdout.strip()}"
        ) from exc


def extract_audio(
    video_path: str,
    output_wav_path: str,
) -> None:
    """Extract audio from a video file as 16 kHz mono WAV.

    Uses ffmpeg to demux the audio track and re-encode it as
    16-bit PCM WAV at 16 kHz with a single channel.

    Args:
        video_path (str): Path to the input video file.
        output_wav_path (str): Path for the output WAV file.

    Raises:
        RuntimeError: If ffmpeg extraction fails.

    Examples:
        >>> extract_audio("input.mp4", "output.wav")
    """
    os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
    result = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            output_wav_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed for {video_path}: {result.stderr.strip()}")


def preprocess_video(
    video_path: str,
    output_dir: str,
) -> tuple[str, float]:
    """Extract audio from a video and probe its duration.

    Creates a WAV file in ``{output_dir}/preprocessed/`` with
    the same stem as the input video.

    Args:
        video_path (str): Path to the input MP4 video.
        output_dir (str): Base directory for preprocessed files.

    Returns:
        tuple[str, float]: ``(audio_wav_path, duration_secs)``.

    Raises:
        RuntimeError: If any preprocessing step fails.

    Examples:
        >>> wav, dur = preprocess_video("in.mp4", "out/")
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]
    prep_dir = os.path.join(output_dir, "preprocessed")
    os.makedirs(prep_dir, exist_ok=True)

    duration = get_video_duration(video_path)

    wav_path = os.path.join(prep_dir, f"{stem}.wav")
    extract_audio(
        video_path=video_path,
        output_wav_path=wav_path,
    )

    return wav_path, duration
