# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CambAI REST API helpers for the S2S dubbing service.

Provides functions for the three-step file upload flow, dubbing task
submission, status polling, and output audio retrieval via the CambAI
public API.
"""

import mimetypes
import os
import time
from pathlib import Path
from typing import Any

import requests

from base_utils import logger

CAMB_API_BASE_URL = "https://client.camb.ai/apis"

# Polling defaults read from environment at call time
_DEFAULT_MAX_ATTEMPTS = 120
_DEFAULT_POLL_INTERVAL = 10


def _detect_content_type(file_path: Path) -> str:
    """Detect MIME content type for a local file.

    Args:
        file_path (Path): Path to the file whose content type is needed.

    Returns:
        str: MIME type string, falling back to ``"application/octet-stream"``
            when the type cannot be determined.

    Examples:
        >>> _detect_content_type(Path("audio.wav"))
        'audio/x-wav'
    """
    guessed_type, _ = mimetypes.guess_type(str(file_path))
    return guessed_type or "application/octet-stream"


def _request_upload_url(
    filename: str,
    content_type: str,
    headers: dict[str, str],
) -> tuple[str, str, dict[str, str]]:
    """Request a presigned upload URL from the CambAI Files API.

    Args:
        filename (str): Name of the file to upload.
        content_type (str): MIME type of the file.
        headers (dict[str, str]): HTTP headers including ``x-api-key``.

    Returns:
        tuple[str, str, dict[str, str]]: ``(file_id, upload_url, upload_headers)``.

    Raises:
        requests.HTTPError: If the endpoint returns a non-2xx response.

    Examples:
        >>> _request_upload_url("clip.wav", "audio/x-wav", {"x-api-key": "k"})
        ('file-abc', 'https://storage.../clip.wav', {})
    """
    response = requests.post(
        f"{CAMB_API_BASE_URL}/files/upload-url",
        headers=headers,
        json={"filename": filename, "content_type": content_type},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    file_id = data["file"]["file_id"]
    upload_url = data["upload"]["url"]
    upload_headers = data["upload"].get("headers", {})
    return file_id, upload_url, upload_headers


def _upload_file_to_presigned_url(
    file_path: Path,
    upload_url: str,
    upload_headers: dict[str, str],
) -> None:
    """Upload a local file to a presigned cloud storage URL.

    Args:
        file_path (Path): Path to the local file to upload.
        upload_url (str): Presigned URL returned by ``_request_upload_url``.
        upload_headers (dict[str, str]): Headers required by the presigned URL.

    Raises:
        RuntimeError: If the upload response status is not in {200, 201, 204}.

    Examples:
        >>> _upload_file_to_presigned_url(Path("clip.wav"), url, {})
    """
    with file_path.open("rb") as f:
        response = requests.put(
            upload_url,
            data=f,
            headers=upload_headers,
            timeout=300,
        )
    if response.status_code not in {200, 201, 204}:
        raise RuntimeError(
            f"File upload failed with status {response.status_code}: {response.text}"
        )


def _confirm_upload(file_id: str, headers: dict[str, str]) -> None:
    """Notify CambAI that a file upload is complete.

    Args:
        file_id (str): CambAI file identifier from ``_request_upload_url``.
        headers (dict[str, str]): HTTP headers including ``x-api-key``.

    Raises:
        requests.HTTPError: If the endpoint returns a non-2xx response.

    Examples:
        >>> _confirm_upload("file-abc", {"x-api-key": "k"})
    """
    response = requests.post(
        f"{CAMB_API_BASE_URL}/files/{file_id}/complete",
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()


def upload_local_file(file_path: Path, headers: dict[str, str]) -> str:
    """Upload a local media file to CambAI via the three-step flow.

    Coordinates: request presigned URL → upload file content → confirm.

    Args:
        file_path (Path): Path to the local file to upload.
        headers (dict[str, str]): HTTP headers including ``x-api-key``.

    Returns:
        str: CambAI file ID ready for use in dubbing tasks.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
        requests.HTTPError: If any CambAI API call returns a non-2xx response.
        RuntimeError: If the presigned-URL upload fails.

    Examples:
        >>> upload_local_file(Path("clip.wav"), {"x-api-key": "k"})
        'file-abc'
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    content_type = _detect_content_type(file_path)
    logger.info(f"Uploading {file_path} (content_type={content_type})")

    file_id, upload_url, upload_headers = _request_upload_url(
        filename=file_path.name,
        content_type=content_type,
        headers=headers,
    )
    logger.debug(f"Received upload URL for file_id={file_id}")

    _upload_file_to_presigned_url(
        file_path=file_path,
        upload_url=upload_url,
        upload_headers=upload_headers,
    )
    logger.debug("File uploaded to presigned URL")

    _confirm_upload(file_id=file_id, headers=headers)
    logger.info(f"Upload confirmed for file_id={file_id}")

    return file_id


def submit_dub_task(
    source_language_id: int,
    target_language_id: int,
    headers: dict[str, str],
    file_id: str,
    *,
    chosen_dictionaries: list[int] | None = None,
    ai_optimization: bool = True,
) -> str:
    """Submit a CambAI direct-dubbing task and return its task ID.

    Args:
        source_language_id (int): CambAI source language ID.
        target_language_id (int): CambAI target language ID.
        headers (dict[str, str]): HTTP headers including ``x-api-key``.
        file_id (str): CambAI file ID from a prior upload.
        chosen_dictionaries (list[int] | None): Unique CambAI dictionary
            IDs for custom terminology. Defaults to ``None`` (omitted).
        ai_optimization (bool): Enable CambAI AI optimization.
            Defaults to ``True``.

    Returns:
        str: CambAI task ID for status polling.

    Raises:
        requests.HTTPError: If CambAI returns a non-2xx response.
        RuntimeError: If CambAI response does not contain a valid task ID.

    Examples:
        >>> submit_dub_task(1, 54, {"x-api-key": "k"}, file_id="file-abc")
        'task_123'
    """
    payload: dict[str, Any] = {
        "source_language": source_language_id,
        "target_languages": [target_language_id],
        "file_id": file_id,
        "ai_optimization": ai_optimization,
    }
    if chosen_dictionaries:
        payload["chosen_dictionaries"] = chosen_dictionaries

    response = requests.post(
        f"{CAMB_API_BASE_URL}/dub",
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()

    data = response.json()
    task_id = data.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise RuntimeError(f"CambAI /dub response missing task_id: {data}")
    return task_id


def wait_for_completion(
    task_id: str,
    headers: dict[str, str],
    max_attempts: int | None = None,
    poll_interval_seconds: int | None = None,
) -> int:
    """Poll CambAI dubbing status until terminal state and return run ID.

    Args:
        task_id (str): CambAI task ID returned by ``/dub``.
        headers (dict[str, str]): HTTP headers including ``x-api-key``.
        max_attempts (int | None): Maximum polling iterations.
            Reads ``S2S_CAMB_DUBBING_MAX_ATTEMPTS`` env if ``None``.
        poll_interval_seconds (int | None): Sleep duration between polls.
            Reads ``S2S_CAMB_DUBBING_POLL_INTERVAL`` env if ``None``.

    Returns:
        int: CambAI run ID when task reaches ``SUCCESS``.

    Raises:
        requests.HTTPError: If status endpoint returns a non-2xx response.
        RuntimeError: If CambAI returns a terminal error status.
        TimeoutError: If polling exceeds ``max_attempts``.

    Examples:
        >>> wait_for_completion("task_123", {"x-api-key": "k"})
        42
    """
    if max_attempts is None:
        max_attempts = int(
            os.environ.get("S2S_CAMB_DUBBING_MAX_ATTEMPTS", str(_DEFAULT_MAX_ATTEMPTS))
        )
    if poll_interval_seconds is None:
        poll_interval_seconds = int(
            os.environ.get("S2S_CAMB_DUBBING_POLL_INTERVAL", str(_DEFAULT_POLL_INTERVAL))
        )

    for attempt in range(max_attempts):
        response = requests.get(
            f"{CAMB_API_BASE_URL}/dub/{task_id}",
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        status_payload = response.json()
        status = str(status_payload.get("status", "")).upper()
        logger.debug(f"CambAI dubbing status (attempt {attempt + 1}): {status}")

        if status == "SUCCESS":
            run_id = status_payload.get("run_id")
            if not isinstance(run_id, int):
                raise RuntimeError(f"CambAI status missing run_id on SUCCESS: {status_payload}")
            return run_id
        if status in {"ERROR", "TIMEOUT", "PAYMENT_REQUIRED"}:
            message = status_payload.get("message")
            raise RuntimeError(f"CambAI dubbing failed with status={status}, message={message}")

        time.sleep(poll_interval_seconds)

    raise TimeoutError(
        f"CambAI dubbing timed out after {max_attempts} attempts "
        f"(interval={poll_interval_seconds}s)."
    )


def get_output_audio_url(run_id: int, headers: dict[str, str]) -> str:
    """Fetch dubbed run metadata and extract the output audio URL.

    Args:
        run_id (int): CambAI run ID returned by status polling.
        headers (dict[str, str]): HTTP headers including ``x-api-key``.

    Returns:
        str: Output audio URL from the CambAI dub-result endpoint.

    Raises:
        requests.HTTPError: If the endpoint returns a non-2xx response.
        RuntimeError: If the response does not contain a valid audio URL.

    Examples:
        >>> get_output_audio_url(42, {"x-api-key": "k"})
        'https://.../dubbed_audio.mp3'
    """
    response = requests.get(
        f"{CAMB_API_BASE_URL}/dub-result/{run_id}",
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    # CambAI may return either output_audio_url or audio_url
    audio_url = payload.get("output_audio_url") or payload.get("audio_url")
    if not isinstance(audio_url, str) or not audio_url:
        raise RuntimeError(f"CambAI dub-result missing audio URL: {payload}")
    return audio_url


def download_output_audio_to_file(audio_url: str, output_file: Path) -> Path:
    """Download dubbed audio from URL and save to disk.

    Args:
        audio_url (str): Public URL pointing to translated audio output.
        output_file (Path): Destination file path for downloaded audio.

    Returns:
        Path: The ``output_file`` path after writing.

    Raises:
        requests.HTTPError: If audio download returns a non-2xx response.

    Examples:
        >>> download_output_audio_to_file("https://.../out.mp3", Path("/tmp/out.mp3"))
        PosixPath('/tmp/out.mp3')
    """
    response = requests.get(audio_url, stream=True, timeout=120)
    response.raise_for_status()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    with output_file.open("wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                total_bytes += len(chunk)
    logger.info(f"Downloaded dubbed audio: {total_bytes} bytes to {output_file}")
    return output_file
