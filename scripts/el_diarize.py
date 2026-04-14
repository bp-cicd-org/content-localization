#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Generate diarization data using ElevenLabs Speech-to-Text (Scribe) API.
# Outputs the native ElevenLabs STT JSON response for direct reuse.
#
# Usage:
#   ELEVENLABS_API_KEY=<key> .venv/bin/python scripts/el_diarize.py \
#       --input-file <path>.wav \
#       --output-file diarization.json

import argparse
import json
import os
import wave

from elevenlabs import ElevenLabs


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for ElevenLabs diarization.

    Returns:
        argparse.Namespace: Parsed CLI values with these defaults:
            - ``input_file``: required
            - ``output_file``: ``"diarization.json"``
            - ``language_code``: ``None`` (auto-detect)
            - ``max_speakers``: ``None`` (model default)
            - ``model_id``: ``"scribe_v2"``
            - ``tag_audio_events``: ``False``

    Examples:
        >>> # python scripts/el_diarize.py --input-file audio.wav
    """
    parser = argparse.ArgumentParser(
        description="Generate diarization data using ElevenLabs Speech-to-Text (Scribe) API. "
        "Outputs native ElevenLabs JSON from speech_to_text.convert.",
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
        "--language-code",
        default=None,
        help="ISO-639-1 or ISO-639-3 language code (e.g. 'eng', 'en'). None for auto-detection.",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum number of speakers for diarization (up to 32). None lets the model decide.",
    )
    parser.add_argument(
        "--model-id",
        default="scribe_v2",
        help="ElevenLabs STT model ID.",
    )
    parser.add_argument(
        "--tag-audio-events",
        action="store_true",
        default=False,
        help="Tag audio events like (laughter), (footsteps) in the transcription.",
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


def response_to_native_json(response: object) -> dict:
    """Convert an ElevenLabs STT response object to a JSON-serializable dict.

    Uses the Pydantic ``model_dump`` method available on the SDK response
    objects.  Falls back to ``dict()`` / ``__dict__`` for older SDK versions.

    Args:
        response (object): ElevenLabs ``SpeechToTextConvertResponse`` object.

    Returns:
        dict: Native ElevenLabs response as a JSON-serializable dictionary.

    Examples:
        >>> native = response_to_native_json(el_response)
        >>> "words" in native
        True
    """
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    return response.__dict__


def extract_diarization_stats(native_response: dict) -> tuple[int, int]:
    """Extract word and speaker counts from native ElevenLabs JSON response.

    Args:
        native_response (dict): Native ElevenLabs STT JSON dictionary
            containing a ``words`` list.

    Returns:
        tuple[int, int]: Tuple of (word_count, unique_speaker_count).

    Examples:
        >>> extract_diarization_stats({"words": []})
        (0, 0)
        >>> extract_diarization_stats(
        ...     {
        ...         "words": [
        ...             {"text": "hello", "speaker_id": "speaker_0"},
        ...             {"text": "world", "speaker_id": "speaker_1"},
        ...         ]
        ...     }
        ... )
        (2, 2)
    """
    words = native_response.get("words", [])
    speaker_ids: set[str] = set()

    for word in words:
        speaker_id = word.get("speaker_id")
        if speaker_id is not None:
            speaker_ids.add(speaker_id)

    return len(words), len(speaker_ids)


def main() -> None:
    """Run ElevenLabs diarization and write native JSON output.

    Reads ``ELEVENLABS_API_KEY`` from the environment, sends the input
    audio file through the ElevenLabs Speech-to-Text API with diarization
    enabled, and writes the native JSON response to disk.

    Returns:
        None.

    Raises:
        ValueError: If ``ELEVENLABS_API_KEY`` is not set.
        FileNotFoundError: If the input file does not exist.

    Examples:
        >>> # python scripts/el_diarize.py --input-file audio.wav -o out.json
    """
    args = parse_args()

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError(
            "ELEVENLABS_API_KEY environment variable not set. "
            "Export it or pass via: ELEVENLABS_API_KEY=<key> python scripts/el_diarize.py ..."
        )

    if not os.path.isfile(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    client = ElevenLabs(api_key=api_key)

    print_wav_info(args.input_file)

    file_size = os.path.getsize(args.input_file)
    print(f"Input file: {args.input_file} ({file_size:,} bytes)")

    stt_kwargs: dict = {
        "model_id": args.model_id,
        "diarize": True,
        "timestamps_granularity": "word",
        "tag_audio_events": args.tag_audio_events,
    }
    if args.language_code is not None:
        stt_kwargs["language_code"] = args.language_code
    if args.max_speakers is not None:
        stt_kwargs["num_speakers"] = args.max_speakers

    print(
        f"Config: model={args.model_id}, language_code={args.language_code}, "
        f"max_speakers={args.max_speakers}, diarization=enabled"
    )

    print(f"Sending {args.input_file} to ElevenLabs STT API...")
    with open(args.input_file, "rb") as audio_file:
        response = client.speech_to_text.convert(file=audio_file, **stt_kwargs)

    native_response = response_to_native_json(response)
    words_count, speaker_count = extract_diarization_stats(native_response)
    if words_count == 0:
        print("WARNING: No diarized words found in ElevenLabs response.")
    else:
        print(f"Generated {words_count} diarized words across {speaker_count} speakers")

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(native_response, f, indent=2, ensure_ascii=False)

    print(f"Native ElevenLabs diarization JSON written to {args.output_file}")


if __name__ == "__main__":
    main()
