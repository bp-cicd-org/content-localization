# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diarization parsers for ASD (flat, RIVA, ElevenLabs, ElevenLabs Studio, and Camb AI formats)."""

import csv
import json

from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import AudioDiarizationInfo
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import AudioSegmentInfo


def _build_segment_from_flat_entry(seg: dict, index: int) -> AudioSegmentInfo:
    """Build ``AudioSegmentInfo`` from a single flat ASD-compatible JSON entry.

    Args:
        seg (dict): Flat JSON object with required keys ``start_time``,
            ``end_time``, ``speaker_id`` and optional ``word``, ``language_code``.
        index (int): Position of this entry in the source list (for error messages).

    Returns:
        AudioSegmentInfo: Populated protobuf message.

    Raises:
        ValueError: If required fields are missing from *seg*.

    Examples:
        >>> seg = {"start_time": 10, "end_time": 50, "speaker_id": 1, "word": "hi"}
        >>> info = _build_segment_from_flat_entry(seg=seg, index=0)
        >>> info.speaker_id
        1
    """
    required_fields = ("start_time", "end_time", "speaker_id")
    missing = [field for field in required_fields if field not in seg]
    if missing:
        raise ValueError(
            f"Invalid flat diarization entry at index {index}: missing fields {missing}."
        )

    segment_kwargs = {
        "start_time": int(seg["start_time"]),
        "end_time": int(seg["end_time"]),
        "speaker_id": int(seg["speaker_id"]),
    }
    if "word" in seg:
        segment_kwargs["word"] = str(seg["word"])
    if "language_code" in seg:
        segment_kwargs["language_code"] = str(seg["language_code"])
    return AudioSegmentInfo(**segment_kwargs)


def _build_segments_from_flat_json(data: list[dict]) -> tuple[list[AudioSegmentInfo], str | None]:
    """Build diarization segments from a flat ASD-compatible JSON list.

    Args:
        data (list[dict]): List of flat segment dictionaries. Each must contain
            ``start_time``, ``end_time``, ``speaker_id``. An optional
            ``transcript`` key on the first entry is used as the overall transcript.

    Returns:
        tuple[list[AudioSegmentInfo], str | None]: Tuple of segment protos and
            transcript string (``None`` if absent).

    Raises:
        ValueError: If *data* is not a list.

    Examples:
        >>> data = [{"start_time": 0, "end_time": 100, "speaker_id": 0}]
        >>> segments, transcript = _build_segments_from_flat_json(data=data)
        >>> len(segments)
        1
    """
    if not isinstance(data, list):
        raise ValueError(
            f"Invalid flat diarization JSON: expected a top-level list, got {type(data).__name__}."
        )
    segments = [_build_segment_from_flat_entry(seg=seg, index=idx) for idx, seg in enumerate(data)]
    transcript = None
    if data and isinstance(data[0], dict):
        raw_transcript = data[0].get("transcript")
        if raw_transcript is not None:
            transcript = str(raw_transcript)
    return segments, transcript


def _parse_elevenlabs_speaker_id(speaker_id: str | None) -> int:
    """Convert an ElevenLabs speaker ID string to an integer.

    Handles formats like ``"speaker_0"``, ``"speaker_12"``, or plain
    numeric strings like ``"0"``.  Returns ``0`` when the input is
    ``None`` or cannot be parsed.

    Args:
        speaker_id (str | None): The raw ``speaker_id`` value from the
            ElevenLabs STT response.

    Returns:
        int: Extracted integer speaker ID.

    Examples:
        >>> _parse_elevenlabs_speaker_id(speaker_id="speaker_0")
        0
        >>> _parse_elevenlabs_speaker_id(speaker_id="speaker_12")
        12
        >>> _parse_elevenlabs_speaker_id(speaker_id=None)
        0
    """
    if speaker_id is None:
        return 0
    # "speaker_0" → "0"
    stripped = speaker_id.rsplit("_", 1)[-1]
    try:
        return int(stripped)
    except ValueError:
        return 0


def _build_segments_from_elevenlabs_json(
    data: dict,
) -> tuple[list[AudioSegmentInfo], str | None]:
    """Build diarization segments from a native ElevenLabs STT JSON response.

    Filters to ``type == "word"`` entries only (skipping spacing and
    punctuation tokens).  Merges consecutive words from the same speaker
    into a single segment (matching demo app behavior).  Converts
    floating-point seconds to integer milliseconds and string speaker
    IDs (``"speaker_0"``) to integers.

    Args:
        data (dict): ElevenLabs ``speech_to_text.convert`` JSON response
            containing a top-level ``words`` list and optional ``text``
            transcript.

    Returns:
        tuple[list[AudioSegmentInfo], str | None]: Tuple of segment protos
            and transcript string (``None`` if absent).

    Raises:
        ValueError: If *data* is not a dict or ``words`` is not a list.

    Examples:
        >>> data = {
        ...     "text": "hello world",
        ...     "language_code": "eng",
        ...     "words": [
        ...         {
        ...             "text": "hello",
        ...             "start": 0.5,
        ...             "end": 0.8,
        ...             "type": "word",
        ...             "speaker_id": "speaker_0",
        ...         },
        ...         {
        ...             "text": " ",
        ...             "start": 0.8,
        ...             "end": 0.9,
        ...             "type": "spacing",
        ...             "speaker_id": "speaker_0",
        ...         },
        ...         {
        ...             "text": "world",
        ...             "start": 0.9,
        ...             "end": 1.2,
        ...             "type": "word",
        ...             "speaker_id": "speaker_1",
        ...         },
        ...     ],
        ... }
        >>> segments, transcript = _build_segments_from_elevenlabs_json(data=data)
        >>> len(segments)
        2
        >>> segments[0].word
        'hello'
    """
    if not isinstance(data, dict):
        raise ValueError(
            "Invalid ElevenLabs diarization JSON: expected a top-level dict, "
            f"got {type(data).__name__}."
        )
    words = data.get("words")
    if not isinstance(words, list):
        raise ValueError("Invalid ElevenLabs diarization JSON: expected 'words' to be a list.")

    language_code = data.get("language_code")
    transcript: str | None = data.get("text")

    # Merge consecutive words with the same speaker into one segment
    segments: list[AudioSegmentInfo] = []
    cur_speaker: int | None = None
    cur_words: list[str] = []
    cur_start: int | None = None
    cur_end: int | None = None

    def _flush() -> None:
        if cur_start is None or cur_end is None or cur_speaker is None:
            return
        segment_kwargs: dict = {
            "start_time": cur_start,
            "end_time": cur_end,
            "speaker_id": cur_speaker,
        }
        if cur_words:
            segment_kwargs["word"] = " ".join(cur_words).strip()
        if language_code:
            segment_kwargs["language_code"] = str(language_code)
        segments.append(AudioSegmentInfo(**segment_kwargs))

    for word in words:
        if word.get("type") != "word":
            continue

        speaker = _parse_elevenlabs_speaker_id(word.get("speaker_id"))
        start_ms = int(word.get("start", 0) * 1000)
        end_ms = int(word.get("end", 0) * 1000)

        # Speaker changed — flush the accumulated segment
        if cur_speaker is not None and speaker != cur_speaker:
            _flush()
            cur_words = []
            cur_start = None
            cur_end = None

        cur_speaker = speaker
        if cur_start is None:
            cur_start = start_ms
        cur_end = end_ms
        if word.get("text") is not None:
            cur_words.append(str(word["text"]))

    _flush()

    return segments, transcript


def _build_segments_from_riva_json(data: dict) -> tuple[list[AudioSegmentInfo], str | None]:
    """Build diarization segments from a native RIVA JSON response.

    Args:
        data (dict): RIVA ``offline_recognize`` JSON response containing a
            top-level ``results`` list with ``alternatives[].words[]`` entries.

    Returns:
        tuple[list[AudioSegmentInfo], str | None]: Tuple of segment protos and
            transcript string (``None`` if absent).

    Raises:
        ValueError: If *data* is not a dict or ``results`` is not a list.

    Examples:
        >>> data = {
        ...     "results": [
        ...         {
        ...             "alternatives": [
        ...                 {
        ...                     "words": [
        ...                         {"startTime": 0, "endTime": 100, "speakerTag": 0, "word": "hi"}
        ...                     ]
        ...                 }
        ...             ]
        ...         }
        ...     ]
        ... }
        >>> segments, transcript = _build_segments_from_riva_json(data=data)
        >>> segments[0].word
        'hi'
    """
    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid Riva diarization JSON: expected a top-level dict, got {type(data).__name__}."
        )
    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("Invalid Riva diarization JSON: expected 'results' to be a list.")

    segments: list[AudioSegmentInfo] = []
    transcript: str | None = None
    for result in results:
        alternatives = result.get("alternatives", [])
        if not alternatives:
            continue

        alt = alternatives[0]
        if transcript is None and alt.get("transcript") is not None:
            transcript = str(alt["transcript"])

        words = alt.get("words", [])
        for word in words:
            segment_kwargs = {
                "start_time": int(word.get("startTime", 0)),
                "end_time": int(word.get("endTime", 0)),
                "speaker_id": int(word.get("speakerTag", 0)),
            }
            if word.get("word") is not None:
                segment_kwargs["word"] = str(word["word"])

            language_code = word.get("languageCode")
            if language_code is None:
                language_code = alt.get("languageCode")
                if isinstance(language_code, list):
                    language_code = language_code[0] if language_code else None
            if language_code:
                segment_kwargs["language_code"] = str(language_code)

            segments.append(AudioSegmentInfo(**segment_kwargs))

    return segments, transcript


def _parse_studio_timestamp(ts: str) -> int:
    """Convert an ElevenLabs Studio ``"HH:MM:SS,ms"`` timestamp to milliseconds.

    Args:
        ts (str): Timestamp string in ``"HH:MM:SS,mmm"`` format
            (e.g. ``"00:01:23,456"``).

    Returns:
        int: Total milliseconds.

    Examples:
        >>> _parse_studio_timestamp(ts="00:01:23,456")
        83456
        >>> _parse_studio_timestamp(ts="00:00:00,000")
        0
    """
    # "HH:MM:SS,ms" → split on comma first for the milliseconds part
    time_part, ms_part = ts.split(",")
    hours, minutes, seconds = time_part.split(":")
    return int(hours) * 3_600_000 + int(minutes) * 60_000 + int(seconds) * 1_000 + int(ms_part)


def _parse_studio_speaker_id(speaker: str) -> int:
    """Convert an ElevenLabs Studio ``"Speaker N"`` label to a zero-based integer.

    Args:
        speaker (str): Speaker label like ``"Speaker 1"`` or ``"Speaker 2"``.

    Returns:
        int: Zero-based speaker ID (``Speaker 1`` → ``0``).

    Examples:
        >>> _parse_studio_speaker_id(speaker="Speaker 1")
        0
        >>> _parse_studio_speaker_id(speaker="Speaker 3")
        2
    """
    # "Speaker 1" → 1 → 0 (zero-based)
    return int(speaker.split()[-1]) - 1


def _build_segments_from_elevenlabs_studio_csv(
    file_path: str,
) -> tuple[list[AudioSegmentInfo], str | None]:
    """Build diarization segments from an ElevenLabs Studio CSV export.

    The CSV has columns: ``speaker``, ``start_time``, ``end_time``,
    ``transcription``, ``translation``.  Each row represents a spoken
    segment with timestamps in ``HH:MM:SS,mmm`` format and speaker
    labels like ``Speaker 1``.

    Args:
        file_path (str): Path to the ElevenLabs Studio CSV file.

    Returns:
        tuple[list[AudioSegmentInfo], str | None]: Tuple of segment protos
            and the concatenated transcript.

    Examples:
        >>> segments, transcript = _build_segments_from_elevenlabs_studio_csv(
        ...     file_path="diarization.csv",
        ... )
    """
    segments: list[AudioSegmentInfo] = []
    transcript_parts: list[str] = []

    with open(file_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            segment = AudioSegmentInfo(
                start_time=_parse_studio_timestamp(ts=row["start_time"]),
                end_time=_parse_studio_timestamp(ts=row["end_time"]),
                speaker_id=_parse_studio_speaker_id(speaker=row["speaker"]),
                word=row.get("transcription", ""),
            )
            segments.append(segment)
            text = row.get("transcription", "")
            if text:
                transcript_parts.append(text)

    transcript = " ".join(transcript_parts) if transcript_parts else None
    return segments, transcript


def _parse_camb_speaker_id(speaker: str) -> int:
    """Convert a Camb AI speaker label to a zero-based integer.

    Handles two formats returned by different Camb AI API versions:
    - ``"SPEAKER_0"`` (underscore-separated, already zero-based)
    - ``"Speaker 1"`` (space-separated, one-based)

    Args:
        speaker (str): Speaker label like ``"SPEAKER_0"`` or ``"Speaker 1"``.

    Returns:
        int: Zero-based speaker ID.

    Examples:
        >>> _parse_camb_speaker_id(speaker="SPEAKER_0")
        0
        >>> _parse_camb_speaker_id(speaker="SPEAKER_2")
        2
        >>> _parse_camb_speaker_id(speaker="Speaker 1")
        0
        >>> _parse_camb_speaker_id(speaker="Speaker 3")
        2
    """
    # "SPEAKER_0" format: underscore-separated, already zero-based
    if "_" in speaker:
        suffix = speaker.rsplit("_", 1)[-1]
        try:
            return int(suffix)
        except ValueError:
            return 0
    # "Speaker 1" format: space-separated, one-based
    return _parse_studio_speaker_id(speaker=speaker)


def _build_segments_from_camb_json(
    data: list[dict] | dict,
) -> tuple[list[AudioSegmentInfo], str | None]:
    """Build diarization segments from Camb AI transcription JSON.

    Accepts either a raw list of segment objects or the API wrapper
    format ``{"transcript": [...]}``.  Each segment has ``start``,
    ``end`` (seconds as float), ``text``, and ``speaker``
    (``"SPEAKER_0"`` or ``"Speaker N"``).
    Converts seconds to integer milliseconds.

    Args:
        data (list[dict] | dict): Camb AI transcription segments — either
            a raw list or a dict with a ``transcript`` key containing the list.

    Returns:
        tuple[list[AudioSegmentInfo], str | None]: Tuple of segment protos
            and the concatenated transcript.

    Raises:
        ValueError: If *data* cannot be resolved to a list of segments.

    Examples:
        >>> data = [
        ...     {"start": 0.5, "end": 1.2, "text": "hello", "speaker": "SPEAKER_0"},
        ...     {"start": 1.5, "end": 2.0, "text": "world", "speaker": "SPEAKER_1"},
        ... ]
        >>> segments, transcript = _build_segments_from_camb_json(data=data)
        >>> len(segments)
        2
        >>> segments[0].start_time
        500
    """
    # Unwrap {"transcript": [...]} wrapper if present
    if isinstance(data, dict):
        if "transcript" in data and isinstance(data["transcript"], list):
            data = data["transcript"]
        else:
            raise ValueError(
                "Invalid Camb AI diarization JSON: expected a list or "
                f"a dict with 'transcript' key, got keys {list(data.keys())}."
            )
    if not isinstance(data, list):
        raise ValueError(
            "Invalid Camb AI diarization JSON: expected a top-level list, "
            f"got {type(data).__name__}."
        )
    segments: list[AudioSegmentInfo] = []
    transcript_parts: list[str] = []

    for seg in data:
        segment_kwargs: dict = {
            "start_time": int(float(seg.get("start", 0)) * 1000),
            "end_time": int(float(seg.get("end", 0)) * 1000),
            "speaker_id": _parse_camb_speaker_id(speaker=str(seg.get("speaker", "Speaker 1"))),
        }
        text = seg.get("text")
        if text is not None:
            segment_kwargs["word"] = str(text)
            if text:
                transcript_parts.append(str(text))

        segments.append(AudioSegmentInfo(**segment_kwargs))

    transcript = " ".join(transcript_parts) if transcript_parts else None
    return segments, transcript


VALID_DIARIZATION_FORMATS = ("flat", "riva", "elevenlabs", "elevenlabs-studio", "camb")


def load_diarization_info(
    diarization_file: str,
    diarization_format: str,
) -> AudioDiarizationInfo | None:
    """Load diarization from flat ASD, RIVA, ElevenLabs, Studio CSV, or Camb AI format.

    Supported schemas:
        1) Flat ASD format (``"flat"``):
            ``[{"start_time": 0, "end_time": 320, "speaker_id": 1, ...}]``
        2) Native RIVA ASR format (``"riva"``):
            ``{"results": [{"alternatives": [{"words": [...]}]}]}``
        3) Native ElevenLabs STT format (``"elevenlabs"``):
            ``{"text": "...", "words": [{"text": "hello", "start": 0.5,
            "end": 0.8, "type": "word", "speaker_id": "speaker_0"}]}``
        4) ElevenLabs Studio CSV (``"elevenlabs-studio"``):
            CSV with ``Speaker``, ``Start``, ``End``, ``Text`` columns.
        5) Camb AI transcription format (``"camb"``):
            ``[{"start": 0.5, "end": 1.2, "text": "hello",
            "speaker": "Speaker 1"}]``

    Args:
        diarization_file (str): Path to the diarization file (JSON or CSV).
        diarization_format (str): Explicit format — one of ``"flat"``,
            ``"riva"``, ``"elevenlabs"``, ``"elevenlabs-studio"``,
            ``"camb"``.

    Returns:
        AudioDiarizationInfo | None: Parsed diarization message, or ``None``
            if *diarization_file* is falsy.

    Raises:
        ValueError: If *diarization_format* is not a recognised value.

    Examples:
        >>> info = load_diarization_info(
        ...     diarization_file="diarization.json",
        ...     diarization_format="flat",
        ... )
        >>> info = load_diarization_info(
        ...     diarization_file="studio.csv",
        ...     diarization_format="elevenlabs-studio",
        ... )
        >>> info = load_diarization_info(
        ...     diarization_file="camb.json",
        ...     diarization_format="camb",
        ... )
    """
    if not diarization_file:
        return None

    if diarization_format not in VALID_DIARIZATION_FORMATS:
        raise ValueError(
            f"Unknown diarization_format={diarization_format!r}. "
            f"Expected one of {VALID_DIARIZATION_FORMATS}."
        )

    # ElevenLabs Studio uses CSV, not JSON — handle it before json.load
    if diarization_format == "elevenlabs-studio":
        segments, transcript = _build_segments_from_elevenlabs_studio_csv(
            file_path=diarization_file,
        )
        kwargs: dict[str, list[AudioSegmentInfo] | str] = {"segments": segments}
        if transcript:
            kwargs["transcript"] = transcript
        return AudioDiarizationInfo(**kwargs)

    with open(diarization_file, encoding="utf-8") as f:
        data = json.load(f)

    if diarization_format == "flat":
        segments, transcript = _build_segments_from_flat_json(data=data)
    elif diarization_format == "riva":
        segments, transcript = _build_segments_from_riva_json(data=data)
    elif diarization_format == "elevenlabs":
        segments, transcript = _build_segments_from_elevenlabs_json(data=data)
    elif diarization_format == "camb":
        segments, transcript = _build_segments_from_camb_json(data=data)

    kwargs = {"segments": segments}
    if transcript:
        kwargs["transcript"] = transcript
    return AudioDiarizationInfo(**kwargs)
