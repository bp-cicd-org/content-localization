#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Generate diarization data using RIVA Parakeet ASR NIM.
# Outputs the native RIVA offline_recognize JSON response for direct reuse.
#
# Usage:
#   .venv/bin/python scripts/riva_parakeet_diarize.py \
#       --input-file <path>.wav \
#       --output-file diarization.json \
#       --server localhost:50053

import argparse
import json
import wave

import riva.client
from google.protobuf import json_format


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for RIVA Parakeet diarization.

    Returns:
        argparse.Namespace: Parsed CLI values with these defaults:
            - ``input_file``: required
            - ``output_file``: ``"diarization.json"``
            - ``server``: ``"localhost:50053"``
            - ``language_code``: ``"en-US"``
            - ``max_speakers``: ``4``

    Examples:
        >>> # python scripts/riva_parakeet_diarize.py --input-file audio.wav
    """
    parser = argparse.ArgumentParser(
        description="Generate diarization data using RIVA Parakeet ASR NIM. "
        "Outputs native RIVA JSON from offline_recognize.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to WAV file (16 kHz mono 16-bit PCM recommended).",
    )
    parser.add_argument(
        "--output-file",
        default="diarization.json",
        help="Path to output JSON diarization file.",
    )
    parser.add_argument(
        "--server",
        default="localhost:50053",
        help="RIVA ASR gRPC address (host:port).",
    )
    parser.add_argument(
        "--language-code",
        default="en-US",
        help="Language code for ASR.",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=4,
        help="Maximum number of speakers for diarization.",
    )
    return parser.parse_args()


def read_wav_pcm(wav_path: str) -> tuple[bytes, int, int]:
    """Read raw PCM bytes from a WAV file.

    Args:
        wav_path: Path to the WAV file.

    Returns:
        Tuple of (pcm_bytes, sample_rate_hz, n_channels).
    """
    with wave.open(wav_path, "rb") as wav:
        nch = wav.getnchannels()
        framerate = wav.getframerate()
        nframes = wav.getnframes()
        total_sec = nframes / float(framerate)
        print(f"WAV: {framerate} Hz, {nch} ch, {nframes} frames ({total_sec:.2f} s)")
        pcm = wav.readframes(nframes)
    return pcm, framerate, nch


def response_to_native_json(response: object) -> dict:
    """Convert a RIVA ASR protobuf response to JSON-ready dict.

    Args:
        response (object): RIVA ``offline_recognize`` response protobuf message.

    Returns:
        dict: Native RIVA response as a JSON-serializable dictionary.

    Examples:
        >>> native = response_to_native_json(riva_response)
        >>> "results" in native
        True
    """
    return json_format.MessageToDict(response)


def extract_diarization_stats(native_response: dict) -> tuple[int, int]:
    """Extract word and speaker counts from native RIVA JSON response.

    Args:
        native_response (dict): Native RIVA ``offline_recognize`` JSON dictionary.

    Returns:
        tuple[int, int]: Tuple of (word_count, unique_speaker_count).

    Examples:
        >>> stats = extract_diarization_stats({"results": []})
        >>> stats
        (0, 0)
    """
    words_count = 0
    speaker_ids: set[int] = set()

    for result in native_response.get("results", []):
        alternatives = result.get("alternatives", [])
        if not alternatives:
            continue
        words = alternatives[0].get("words", [])
        words_count += len(words)
        for word in words:
            speaker_tag = word.get("speakerTag")
            if speaker_tag is not None:
                speaker_ids.add(int(speaker_tag))

    return words_count, len(speaker_ids)


def main() -> None:
    """Run RIVA Parakeet diarization and write native JSON output.

    Connects to a RIVA ASR server, sends the input WAV file for
    offline recognition with diarization enabled, and writes the
    native RIVA JSON response to disk.

    Returns:
        None.

    Raises:
        Exception: Propagated from RIVA client on connection or recognition failure.

    Examples:
        >>> # python scripts/riva_parakeet_diarize.py --input-file audio.wav -o out.json
    """
    args = parse_args()

    auth = riva.client.Auth(uri=args.server)
    asr_service = riva.client.ASRService(auth)

    # Configure recognition with diarization and word-level timestamps
    config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        max_alternatives=1,
        enable_automatic_punctuation=True,
        verbatim_transcripts=False,
        sample_rate_hertz=16000,
        language_code=args.language_code,
        enable_word_time_offsets=True,
    )

    # Enable speaker diarization
    config.diarization_config.enable_speaker_diarization = True
    config.diarization_config.max_speaker_count = args.max_speakers

    print(
        f"Config: language_code={args.language_code}, "
        f"max_speakers={args.max_speakers}, diarization=enabled"
    )

    pcm_bytes, sample_rate, nch = read_wav_pcm(args.input_file)
    bytes_per_sec = sample_rate * nch * 2  # 16-bit
    duration_sec = len(pcm_bytes) / bytes_per_sec
    print(f"Sending {len(pcm_bytes)} PCM bytes ({duration_sec:.2f} s) to RIVA...")

    response = asr_service.offline_recognize(audio_bytes=pcm_bytes, config=config)

    native_response = response_to_native_json(response)
    words_count, speaker_count = extract_diarization_stats(native_response)
    if words_count == 0:
        print("WARNING: No diarized words found in RIVA response.")
    else:
        print(f"Generated {words_count} diarized words across {speaker_count} speakers")

    # Write native RIVA JSON output
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(native_response, f, indent=2, ensure_ascii=False)

    print(f"Native RIVA diarization JSON written to {args.output_file}")


if __name__ == "__main__":
    main()
