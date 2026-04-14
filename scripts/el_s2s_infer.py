#!/bin/python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invoke ElevenLabs end-to-end dubbing for local media and save translated audio output."""

import argparse
import os
import subprocess
import tempfile
import time
from pathlib import Path

from elevenlabs import ElevenLabs


def convert_video_to_audio_ffmpeg(
    video_file: Path, audio_file: Path, sample_rate_hz: int = 16000
) -> None:
    """Convert a video file to WAV audio using ``ffmpeg``.

    Extracts mono PCM audio at the given sample rate via the ``ffmpeg``
    command-line tool.

    Args:
        video_file (Path): Input video file path.
        audio_file (Path): Output WAV audio file path.
        sample_rate_hz (int): Sampling rate for output audio. Defaults to ``16000``.

    Returns:
        None.

    Raises:
        FileNotFoundError: If ``video_file`` does not exist.
        subprocess.CalledProcessError: If ``ffmpeg`` exits with a non-zero status.

    Examples:
        >>> convert_video_to_audio_ffmpeg(Path("input.mp4"), Path("output.wav"))
    """
    if not os.path.isfile(video_file):
        raise FileNotFoundError(f"Input video file {video_file} not found.")
    print(f"Converting {video_file} to {audio_file}.")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        f"{video_file}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        f"{sample_rate_hz}",
        "-acodec",
        "pcm_s16le",
        f"{audio_file}",
    ]
    subprocess.run(cmd, check=True)


def download_dubbed_file(
    client: ElevenLabs, dubbing_id: str, language_code: str, output_path: Path
) -> None:
    """Download the dubbed file for a given dubbing ID and language code.

    Args:
        client (ElevenLabs): Authenticated ElevenLabs client instance.
        dubbing_id (str): The ID of the dubbing project.
        language_code (str): The language code for the dubbing.
        output_path (Path): Destination file path for the downloaded audio.

    Returns:
        None.

    Examples:
        >>> download_dubbed_file(client, "dub-123", "es", Path("output.wav"))
    """
    with open(output_path, "wb") as file:
        for chunk in client.dubbing.audio.get(dubbing_id, language_code):
            file.write(chunk)


def wait_for_dubbing_completion(client: ElevenLabs, dubbing_id: str) -> bool:
    """Wait for the dubbing process to complete by polling status.

    Args:
        client (ElevenLabs): Authenticated ElevenLabs client instance.
        dubbing_id (str): The dubbing project ID.

    Returns:
        bool: ``True`` if the dubbing succeeded, ``False`` if it failed or timed out.

    Examples:
        >>> wait_for_dubbing_completion(client, "dub-123")
        True
    """
    MAX_ATTEMPTS = 120
    CHECK_INTERVAL = 10  # In seconds

    for _ in range(MAX_ATTEMPTS):
        metadata = client.dubbing.get(dubbing_id)
        status = (metadata.status or "").lower()

        if status == "dubbed":
            return True
        elif status in {"dubbing", "queued", "in_progress", "processing", "pending", "created"}:
            print(
                "Dubbing status:",
                metadata.status,
                "- will check again in",
                CHECK_INTERVAL,
                "seconds.",
            )
            time.sleep(CHECK_INTERVAL)
        elif status in {"failed", "error", "cancelled"}:
            print(f"Dubbing failed with status={metadata.status}: {metadata.error}")
            return False
        else:
            print(f"Unknown dubbing status={metadata.status}; continuing to poll for completion.")
            time.sleep(CHECK_INTERVAL)

    print("Dubbing timed out")
    return False


def create_dub_from_file(
    client: ElevenLabs,
    input_file_path: Path,
    file_format: str,
    source_language: str,
    target_language: str,
    output_file_path: Path,
) -> Path | None:
    """Dub an audio or video file from one language to another and save the output.

    Args:
        client (ElevenLabs): Authenticated ElevenLabs client instance.
        input_file_path (Path): The file path of the audio or video to dub.
        file_format (str): The MIME type of the input file (e.g. ``"audio/wav"``).
        source_language (str): The language of the input file.
        target_language (str): The target language to dub into.
        output_file_path (Path): The file path of the output file.

    Returns:
        Path | None: The file path of the dubbed file, or ``None`` if the operation failed.

    Examples:
        >>> create_dub_from_file(
        ...     client, Path("input.wav"), "audio/wav", "en", "es", Path("output.wav")
        ... )
        PosixPath('output.wav')
    """
    if not os.path.isfile(input_file_path):
        raise FileNotFoundError(f"{input_file_path} does not exist.")

    with open(input_file_path, "rb") as audio_file:
        response = client.dubbing.create(
            file=(os.path.basename(input_file_path), audio_file, file_format),
            target_lang=target_language,
            mode="automatic",
            source_lang=source_language,
        )

    dubbing_id = response.dubbing_id
    if wait_for_dubbing_completion(client, dubbing_id):
        download_dubbed_file(
            client=client,
            dubbing_id=dubbing_id,
            language_code=target_language,
            output_path=output_file_path,
        )
        return output_file_path
    else:
        return None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for ElevenLabs dubbing pipeline.

    Returns:
        argparse.Namespace: Parsed CLI values with these defaults:
            - ``source_language_code``: ``"en"``
            - ``target_language_code``: ``"es"``
            - ``output_file``: ``"output.wav"``

    Examples:
        >>> args = parse_args()  # with appropriate sys.argv
    """
    parser = argparse.ArgumentParser(
        description="Speech-to-Speech translation service using ElevenLabs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-file",
        required=True,
        type=Path,
        help="A path to a local file (mp4) to transcribe.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=Path,
        default="output.wav",
        required=True,
        help="Output file .wav file to write audio.",
    )
    parser.add_argument(
        "--source-language-code",
        default="en",
        help="Language code of the source input.",
    )
    parser.add_argument(
        "--target-language-code",
        default="es",
        help="Language code of the target language.",
    )
    args = parser.parse_args()
    args.input_file = args.input_file.expanduser()
    if args.output_file is not None:
        args.output_file = args.output_file.expanduser()
    return args


def main() -> None:
    """Run ElevenLabs end-to-end dubbing flow and save translated audio.

    Reads ``ELEVENLABS_API_KEY`` from the environment, extracts audio from
    the input video, submits a dubbing request, and writes the result.

    Returns:
        None.

    Raises:
        ValueError: If ``ELEVENLABS_API_KEY`` is not set.

    Examples:
        >>> # CLI usage:
        >>> # python scripts/el_s2s_infer.py --input-file input.mp4 -o output.wav
    """
    start_time = time.time()
    args = parse_args()

    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
    if not elevenlabs_api_key:
        raise ValueError("ELEVENLABS_API_KEY environment variable not set")
    client = ElevenLabs(api_key=elevenlabs_api_key)

    _, extracted_audio = tempfile.mkstemp(prefix="audio_", suffix=".wav")
    extracted_audio_path = Path(extracted_audio)
    convert_video_to_audio_ffmpeg(args.input_file, extracted_audio_path)

    result = create_dub_from_file(
        client=client,
        input_file_path=extracted_audio_path,
        file_format="audio/wav",
        source_language=args.source_language_code,
        target_language=args.target_language_code,
        output_file_path=args.output_file,
    )
    if result:
        print(f"Dubbing was successful! File saved at: {result}.")
        print(f"Time taken for invocation: {time.time() - start_time}")
    else:
        print("Dubbing failed or timed out.")


if __name__ == "__main__":
    main()
