# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for client LipSync app helpers."""

import pytest

from client.lipsync.request_generators import _speaker_info_from_row
from client.lipsync.request_generators import group_rows_into_per_frame_infos

pytestmark = pytest.mark.unit


def test_speaker_info_from_row_basic_bbox_only() -> None:
    """CSV rows with only frame and bbox still parse correctly."""
    row = ["12", "1.0", "2.0", "3.0", "4.0"]
    frame_id, speaker = _speaker_info_from_row(row)

    assert frame_id == 12
    assert speaker.speaker_bbox.x == 1.0
    assert speaker.speaker_bbox.y == 2.0
    assert speaker.speaker_bbox.width == 3.0
    assert speaker.speaker_bbox.height == 4.0
    assert speaker.HasField("speaker_id") is False
    assert speaker.HasField("is_speaking") is False


def test_speaker_info_from_row_preserves_metadata() -> None:
    """CSV rows with ASD metadata populate speaker_id and is_speaking."""
    row = ["42", "10", "20", "30", "40", "0", "7", "True", "0.95"]
    frame_id, speaker = _speaker_info_from_row(row)

    assert frame_id == 42
    assert speaker.speaker_id == 7
    assert speaker.is_speaking is True


def test_speaker_info_from_row_false_speaking_value() -> None:
    """String false-like values are interpreted as not speaking."""
    row = ["99", "1", "2", "3", "4", "0", "11", "false", "0.1"]
    _, speaker = _speaker_info_from_row(row)

    assert speaker.speaker_id == 11
    assert speaker.is_speaking is False


def test_group_rows_into_per_frame_infos_groups_same_frame() -> None:
    """Multiple speakers in the same frame are grouped into one message."""
    rows = [
        ["5", "10", "20", "30", "40", "0", "1", "True", "0.9"],
        ["5", "50", "60", "70", "80", "0", "2", "False", "0.8"],
        ["6", "11", "21", "31", "41", "0", "3", "True", "0.7"],
    ]
    per_frame = group_rows_into_per_frame_infos(rows)

    # Frame 5 has two speakers, frame 6 has one
    assert len(per_frame) == 2
    assert per_frame[0].frame_id == 5
    assert len(per_frame[0].speaker_infos) == 2
    assert per_frame[0].speaker_infos[0].speaker_id == 1
    assert per_frame[0].speaker_infos[1].speaker_id == 2
    assert per_frame[1].frame_id == 6
    assert len(per_frame[1].speaker_infos) == 1
