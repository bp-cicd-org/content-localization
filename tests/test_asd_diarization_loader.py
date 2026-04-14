# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ASD diarization JSON loading compatibility."""

import json
from pathlib import Path

import pytest

from client.asd.diarization import load_diarization_info

pytestmark = pytest.mark.unit


def test_load_diarization_info_with_flat_format(tmp_path: Path) -> None:
    """Flat ASD diarization JSON parses into AudioDiarizationInfo."""
    diarization_path = tmp_path / "flat_diarization.json"
    diarization_path.write_text(
        json.dumps(
            [
                {
                    "start_time": 10,
                    "end_time": 50,
                    "speaker_id": 2,
                    "word": "hello",
                    "language_code": "en-US",
                    "transcript": "hello world",
                },
                {
                    "start_time": 60,
                    "end_time": 100,
                    "speaker_id": 3,
                    "word": "world",
                    "language_code": "en-US",
                },
            ]
        ),
        encoding="utf-8",
    )

    diarization_info = load_diarization_info(str(diarization_path), diarization_format="flat")

    assert diarization_info is not None
    assert len(diarization_info.segments) == 2
    assert diarization_info.segments[0].start_time == 10
    assert diarization_info.segments[0].end_time == 50
    assert diarization_info.segments[0].speaker_id == 2
    assert diarization_info.segments[0].word == "hello"
    assert diarization_info.segments[0].language_code == "en-US"
    assert diarization_info.transcript == "hello world"


def test_load_diarization_info_with_riva_native_format(tmp_path: Path) -> None:
    """Riva-native diarization JSON parses into AudioDiarizationInfo."""
    diarization_path = tmp_path / "riva_diarization.json"
    diarization_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "alternatives": [
                            {
                                "transcript": "hello world",
                                "words": [
                                    {
                                        "startTime": 320,
                                        "endTime": 400,
                                        "word": "hello",
                                        "languageCode": "en-US",
                                        "speakerTag": 0,
                                    },
                                    {
                                        "startTime": 640,
                                        "endTime": 800,
                                        "word": "world",
                                        "speakerTag": 1,
                                    },
                                ],
                                "languageCode": ["en-US"],
                            }
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    diarization_info = load_diarization_info(str(diarization_path), diarization_format="riva")

    assert diarization_info is not None
    assert len(diarization_info.segments) == 2
    assert diarization_info.segments[0].start_time == 320
    assert diarization_info.segments[0].end_time == 400
    assert diarization_info.segments[0].speaker_id == 0
    assert diarization_info.segments[0].language_code == "en-US"
    assert diarization_info.segments[1].speaker_id == 1
    assert diarization_info.segments[1].language_code == "en-US"
    assert diarization_info.transcript == "hello world"


def test_load_diarization_info_with_elevenlabs_format(tmp_path: Path) -> None:
    """ElevenLabs STT diarization JSON parsed correctly with explicit format."""
    diarization_path = tmp_path / "el_diarization.json"
    diarization_path.write_text(
        json.dumps(
            {
                "language_code": "eng",
                "language_probability": 1.0,
                "text": "hello world",
                "words": [
                    {
                        "text": "hello",
                        "start": 0.5,
                        "end": 0.8,
                        "type": "word",
                        "speaker_id": "speaker_0",
                        "logprob": -0.01,
                        "characters": None,
                    },
                    {
                        "text": " ",
                        "start": 0.8,
                        "end": 0.85,
                        "type": "spacing",
                        "speaker_id": "speaker_0",
                        "logprob": 0.0,
                        "characters": None,
                    },
                    {
                        "text": "world",
                        "start": 0.9,
                        "end": 1.2,
                        "type": "word",
                        "speaker_id": "speaker_1",
                        "logprob": -0.02,
                        "characters": None,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    diarization_info = load_diarization_info(str(diarization_path), diarization_format="elevenlabs")

    assert diarization_info is not None
    # Only "word" type entries are included (spacing is filtered out)
    assert len(diarization_info.segments) == 2
    assert diarization_info.segments[0].start_time == 500
    assert diarization_info.segments[0].end_time == 800
    assert diarization_info.segments[0].speaker_id == 0
    assert diarization_info.segments[0].word == "hello"
    assert diarization_info.segments[0].language_code == "eng"
    assert diarization_info.segments[1].start_time == 900
    assert diarization_info.segments[1].end_time == 1200
    assert diarization_info.segments[1].speaker_id == 1
    assert diarization_info.segments[1].word == "world"
    assert diarization_info.transcript == "hello world"


def test_load_diarization_info_with_elevenlabs_format_explicit(tmp_path: Path) -> None:
    """Explicit diarization_format='elevenlabs' forces the ElevenLabs parser."""
    diarization_path = tmp_path / "el_diarization.json"
    diarization_path.write_text(
        json.dumps(
            {
                "text": "hi",
                "words": [
                    {
                        "text": "hi",
                        "start": 0.1,
                        "end": 0.3,
                        "type": "word",
                        "speaker_id": "speaker_5",
                        "logprob": -0.1,
                        "characters": None,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    diarization_info = load_diarization_info(str(diarization_path), diarization_format="elevenlabs")

    assert diarization_info is not None
    assert len(diarization_info.segments) == 1
    assert diarization_info.segments[0].speaker_id == 5
    assert diarization_info.segments[0].start_time == 100
    assert diarization_info.segments[0].end_time == 300


def test_load_diarization_info_with_explicit_riva_format(tmp_path: Path) -> None:
    """Explicit diarization_format='riva' forces the RIVA parser."""
    diarization_path = tmp_path / "riva_diarization.json"
    diarization_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "alternatives": [
                            {
                                "words": [
                                    {
                                        "startTime": 100,
                                        "endTime": 200,
                                        "word": "ok",
                                        "speakerTag": 0,
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    diarization_info = load_diarization_info(str(diarization_path), diarization_format="riva")

    assert diarization_info is not None
    assert len(diarization_info.segments) == 1
    assert diarization_info.segments[0].word == "ok"


def test_load_diarization_info_invalid_format_value(tmp_path: Path) -> None:
    """Unknown diarization_format value raises ValueError."""
    diarization_path = tmp_path / "any.json"
    diarization_path.write_text(json.dumps([]), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown diarization_format"):
        load_diarization_info(str(diarization_path), diarization_format="nope")


def test_load_diarization_info_auto_is_rejected(tmp_path: Path) -> None:
    """'auto' is no longer a valid format value."""
    diarization_path = tmp_path / "any.json"
    diarization_path.write_text(json.dumps([]), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown diarization_format"):
        load_diarization_info(str(diarization_path), diarization_format="auto")


def test_load_diarization_info_with_camb_format_explicit(tmp_path: Path) -> None:
    """Explicit diarization_format='camb' forces the Camb AI parser."""
    diarization_path = tmp_path / "camb_diarization.json"
    diarization_path.write_text(
        json.dumps(
            [
                {
                    "start": 0.1,
                    "end": 0.5,
                    "text": "hi",
                    "speaker": "Speaker 3",
                },
            ]
        ),
        encoding="utf-8",
    )

    diarization_info = load_diarization_info(str(diarization_path), diarization_format="camb")

    assert diarization_info is not None
    assert len(diarization_info.segments) == 1
    assert diarization_info.segments[0].start_time == 100
    assert diarization_info.segments[0].end_time == 500
    # "Speaker 3" → 2 (zero-based)
    assert diarization_info.segments[0].speaker_id == 2
    assert diarization_info.segments[0].word == "hi"


def test_load_diarization_info_with_camb_format_multi_speaker(tmp_path: Path) -> None:
    """Camb AI format with multiple speakers parses correctly."""
    diarization_path = tmp_path / "camb_diarization.json"
    diarization_path.write_text(
        json.dumps(
            [
                {
                    "start": 0.5,
                    "end": 1.2,
                    "text": "hello world",
                    "speaker": "Speaker 1",
                },
                {
                    "start": 1.5,
                    "end": 2.3,
                    "text": "how are you",
                    "speaker": "Speaker 2",
                },
            ]
        ),
        encoding="utf-8",
    )

    diarization_info = load_diarization_info(str(diarization_path), diarization_format="camb")

    assert diarization_info is not None
    assert len(diarization_info.segments) == 2
    # Seconds → milliseconds
    assert diarization_info.segments[0].start_time == 500
    assert diarization_info.segments[0].end_time == 1200
    # "Speaker 1" → 0 (zero-based)
    assert diarization_info.segments[0].speaker_id == 0
    assert diarization_info.segments[0].word == "hello world"
    assert diarization_info.segments[1].start_time == 1500
    assert diarization_info.segments[1].end_time == 2300
    # "Speaker 2" → 1 (zero-based)
    assert diarization_info.segments[1].speaker_id == 1
    assert diarization_info.segments[1].word == "how are you"
    assert diarization_info.transcript == "hello world how are you"


def test_wrong_format_flat_with_dict_raises_error(tmp_path: Path) -> None:
    """Passing format='flat' with dict data raises ValueError."""
    diarization_path = tmp_path / "el_data.json"
    diarization_path.write_text(
        json.dumps({"words": [{"text": "hi", "start": 0.1, "end": 0.3, "type": "word"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid flat diarization JSON"):
        load_diarization_info(str(diarization_path), diarization_format="flat")


def test_wrong_format_elevenlabs_with_list_raises_error(tmp_path: Path) -> None:
    """Passing format='elevenlabs' with list data raises ValueError."""
    diarization_path = tmp_path / "flat_data.json"
    diarization_path.write_text(
        json.dumps([{"start_time": 0, "end_time": 100, "speaker_id": 0}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid ElevenLabs diarization JSON"):
        load_diarization_info(str(diarization_path), diarization_format="elevenlabs")


def test_wrong_format_camb_with_dict_raises_error(tmp_path: Path) -> None:
    """Passing format='camb' with dict data raises ValueError."""
    diarization_path = tmp_path / "riva_data.json"
    diarization_path.write_text(
        json.dumps({"results": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid Camb AI diarization JSON"):
        load_diarization_info(str(diarization_path), diarization_format="camb")


def test_wrong_format_riva_with_list_raises_error(tmp_path: Path) -> None:
    """Passing format='riva' with list data raises ValueError."""
    diarization_path = tmp_path / "flat_data.json"
    diarization_path.write_text(
        json.dumps([{"start_time": 0, "end_time": 100, "speaker_id": 0}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid Riva diarization JSON"):
        load_diarization_info(str(diarization_path), diarization_format="riva")
