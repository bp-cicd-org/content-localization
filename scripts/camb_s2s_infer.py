#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invoke CAMB end-to-end dubbing for local or URL media and save translated audio output."""

import argparse
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

import requests

CAMB_API_BASE_URL = "https://client.camb.ai/apis"
CAMB_API_KEY_ENV = "CAMB_API_KEY"


def detect_content_type(file_path: Path) -> str:
    """Detect MIME content type for a local file.

    Uses Python's ``mimetypes`` module to map the file extension to a MIME
    type. Common mappings relevant to CAMB:

    - ``.mp4``  → ``video/mp4``
    - ``.mov``  → ``video/quicktime``
    - ``.mp3``  → ``audio/mpeg``
    - ``.wav``  → ``audio/x-wav``

    Args:
        file_path (Path): Path to the file whose content type is needed.

    Returns:
        str: MIME type string, falling back to ``"application/octet-stream"``
        when the type cannot be determined.

    Example:
        ``detect_content_type(Path("video.mp4"))`` returns ``"video/mp4"``.
    """
    guessed_type, _ = mimetypes.guess_type(str(file_path))
    return guessed_type or "application/octet-stream"


def request_upload_url(
    filename: str,
    content_type: str,
    headers: dict[str, str],
) -> tuple[str, str, dict[str, str]]:
    """Request a presigned upload URL from CAMB Files API.

    Args:
        filename (str): Name of the file to upload (used by CAMB for metadata).
        content_type (str): MIME type of the file.
        headers (dict[str, str]): HTTP headers including ``x-api-key``.

    Returns:
        tuple[str, str, dict[str, str]]: A 3-tuple of
        ``(file_id, upload_url, upload_headers)``.

    Raises:
        requests.HTTPError: If the CAMB ``/files/upload-url`` endpoint
            returns a non-2xx response.

    Example:
        ``request_upload_url("clip.mp4", "video/mp4", headers)``
        returns ``("file-abc", "https://storage.../clip.mp4", {})``.
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


def upload_file_to_presigned_url(
    file_path: Path,
    upload_url: str,
    upload_headers: dict[str, str],
) -> None:
    """Upload a local file to a presigned cloud storage URL.

    Args:
        file_path (Path): Path to the local file to upload.
        upload_url (str): Presigned URL returned by ``request_upload_url``.
        upload_headers (dict[str, str]): Headers required by the presigned URL.

    Returns:
        None.

    Raises:
        RuntimeError: If the upload response status is not in {200, 201, 204}.

    Example:
        ``upload_file_to_presigned_url(Path("clip.mp4"), url, headers)``.
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


def confirm_upload(file_id: str, headers: dict[str, str]) -> None:
    """Notify CAMB that a file upload is complete.

    Args:
        file_id (str): CAMB file identifier returned by ``request_upload_url``.
        headers (dict[str, str]): HTTP headers including ``x-api-key``.

    Returns:
        None.

    Raises:
        requests.HTTPError: If the ``/files/{file_id}/complete`` endpoint
            returns a non-2xx response.

    Example:
        ``confirm_upload("file-abc", headers)``.
    """
    response = requests.post(
        f"{CAMB_API_BASE_URL}/files/{file_id}/complete",
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()


def upload_local_file(file_path: Path, headers: dict[str, str]) -> str:
    """Upload a local media file to CAMB and return its file ID.

    Coordinates the three-step upload flow: request presigned URL,
    upload file content, and confirm completion. Supports MP4, MOV,
    MP3, WAV, and other common audio/video formats (content type is
    auto-detected from the file extension).

    Args:
        file_path (Path): Path to the local file to upload.
        headers (dict[str, str]): HTTP headers including ``x-api-key``.

    Returns:
        str: CAMB file ID ready for use in dubbing tasks.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
        requests.HTTPError: If any CAMB API call returns a non-2xx response.
        RuntimeError: If the presigned-URL upload fails.

    Example:
        ``upload_local_file(Path("clip.mp4"), headers)`` returns
        ``"file-abc"``.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    content_type = detect_content_type(file_path)
    print(f"Uploading {file_path} (content_type={content_type})")

    file_id, upload_url, upload_headers = request_upload_url(
        filename=file_path.name,
        content_type=content_type,
        headers=headers,
    )
    print(f"Received upload URL for file_id={file_id}")

    upload_file_to_presigned_url(
        file_path=file_path,
        upload_url=upload_url,
        upload_headers=upload_headers,
    )
    print("File uploaded to presigned URL")

    confirm_upload(file_id=file_id, headers=headers)
    print(f"Upload confirmed for file_id={file_id}")

    return file_id


def parse_task_id(response_payload: dict[str, Any]) -> str:
    """Extract task ID from CAMB ``/dub`` response payload.

    Args:
        response_payload (dict[str, Any]): JSON-decoded body from CAMB create-dub API.

    Returns:
        str: Non-empty CAMB task ID.

    Raises:
        RuntimeError: If ``task_id`` is missing or not a non-empty string.

    Example:
        ``parse_task_id({"task_id": "abc"})`` returns ``"abc"``.
    """
    task_id = response_payload.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise RuntimeError(f"CAMB /dub response missing task_id: {response_payload}")
    return task_id


def submit_dub_task(
    source_language_id: int,
    target_language_id: int,
    headers: dict[str, str],
    input_url: str | None = None,
    file_id: str | None = None,
) -> str:
    """Submit a CAMB direct-dubbing task and return its task ID.

    Exactly one of ``input_url`` or ``file_id`` must be provided.

    Args:
        source_language_id (int): CAMB source language ID.
        target_language_id (int): CAMB target language ID.
        headers (dict[str, str]): HTTP headers including ``x-api-key``.
        input_url (str | None): Public media URL passed as ``file_url``.
            Default: ``None``.
        file_id (str | None): CAMB file ID from a prior upload.
            Default: ``None``.

    Returns:
        str: CAMB task ID for status polling.

    Raises:
        ValueError: If neither or both of ``input_url`` and ``file_id``
            are provided.
        requests.HTTPError: If CAMB returns a non-2xx response.
        RuntimeError: If CAMB response does not contain a valid task ID.

    Example:
        ``submit_dub_task(1, 54, headers, input_url="https://example.com/a.mp3")``
        returns a task ID like ``"task_123"``.
    """
    if bool(input_url) == bool(file_id):
        raise ValueError(
            "Exactly one of input_url or file_id must be provided, "
            f"got input_url={input_url!r}, file_id={file_id!r}"
        )

    payload: dict[str, Any] = {
        "source_language": source_language_id,
        "target_languages": [target_language_id],
    }
    if input_url:
        payload["file_url"] = input_url
    else:
        payload["file_id"] = file_id

    response = requests.post(
        f"{CAMB_API_BASE_URL}/dub",
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return parse_task_id(response.json())


def wait_for_completion(
    task_id: str,
    headers: dict[str, str],
    max_attempts: int,
    poll_interval_seconds: int,
) -> int:
    """Poll CAMB dubbing status until terminal state and return run ID.

    Args:
        task_id (str): CAMB task ID returned by ``/dub``.
        headers (dict[str, str]): HTTP headers, including ``x-api-key``.
        max_attempts (int): Maximum number of polling iterations.
        poll_interval_seconds (int): Sleep duration between polls.

    Returns:
        int: CAMB run ID when task reaches ``SUCCESS``.

    Raises:
        requests.HTTPError: If status endpoint returns a non-2xx response.
        RuntimeError: If CAMB returns terminal error states or missing ``run_id``.
        TimeoutError: If polling exceeds ``max_attempts`` before success.

    Example:
        ``wait_for_completion("task_123", headers, max_attempts=120, poll_interval_seconds=10)``
        returns an integer run ID on success.
    """
    for _ in range(max_attempts):
        response = requests.get(
            f"{CAMB_API_BASE_URL}/dub/{task_id}",
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        status_payload = response.json()
        status = str(status_payload.get("status", "")).upper()
        print(f"Dubbing status: {status}")

        if status == "SUCCESS":
            run_id = status_payload.get("run_id")
            if not isinstance(run_id, int):
                raise RuntimeError(f"CAMB status missing run_id on SUCCESS: {status_payload}")
            return run_id
        if status in {"ERROR", "TIMEOUT", "PAYMENT_REQUIRED"}:
            message = status_payload.get("message")
            raise RuntimeError(f"CAMB dubbing failed with status={status}, message={message}")

        time.sleep(poll_interval_seconds)

    raise TimeoutError(
        f"CAMB dubbing timed out after {max_attempts} attempts (interval={poll_interval_seconds}s)."
    )


def get_output_audio_url(run_id: int, headers: dict[str, str]) -> str:
    """Fetch dubbed run metadata and extract the output audio URL.

    Args:
        run_id (int): CAMB run ID returned by status polling.
        headers (dict[str, str]): HTTP headers, including ``x-api-key``.

    Returns:
        str: Output audio URL from ``output_audio_url`` (or fallback ``audio_url``).

    Raises:
        requests.HTTPError: If run-info endpoint returns a non-2xx response.
        RuntimeError: If the response does not contain a valid audio URL.

    Example:
        ``get_output_audio_url(42, headers)`` returns
        ``"https://.../dubbed_audio.mp3"``.
    """
    response = requests.get(
        f"{CAMB_API_BASE_URL}/dub-result/{run_id}",
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    # CAMB docs/examples may return either output_audio_url or audio_url.
    audio_url = payload.get("output_audio_url") or payload.get("audio_url")
    if not isinstance(audio_url, str) or not audio_url:
        raise RuntimeError(f"CAMB dub-result missing audio URL: {payload}")
    return audio_url


def download_output_audio(audio_url: str, output_file: Path) -> None:
    """Download dubbed audio bytes from URL and save them to disk.

    Args:
        audio_url (str): Public URL pointing to translated audio output.
        output_file (Path): Destination file path for downloaded audio.

    Returns:
        None: Writes content to ``output_file``.

    Raises:
        requests.HTTPError: If audio download returns a non-2xx response.
        OSError: If destination path cannot be created or written.

    Example:
        ``download_output_audio("https://.../out.mp3", Path("outputs/out.mp3"))``.
    """
    response = requests.get(audio_url, stream=True, timeout=120)
    response.raise_for_status()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for CAMB direct dubbing.

    Args:
        None.

    Returns:
        argparse.Namespace: Parsed CLI values with these defaults:
            - ``source_language``: ``1`` (English)
            - ``target_language``: ``54`` (Spanish)
            - ``max_attempts``: ``120``
            - ``poll_interval_seconds``: ``10``
            - ``input_url``: ``None`` (mutually exclusive with ``input_file``)
            - ``input_file``: ``None`` (mutually exclusive with ``input_url``)

    Raises:
        SystemExit: Raised by ``argparse`` for invalid/missing CLI inputs.

    Example:
        ``python scripts/camb_s2s_infer.py --input-url https://example.com/a.mp3``
        ``-o outputs/a_es.mp3``

        ``python scripts/camb_s2s_infer.py --input-file ./clip.mp4``
        ``--source-language 1 --target-language 54 -o outputs/clip_es.mp3``
    """
    parser = argparse.ArgumentParser(
        description="Speech-to-Speech translation service using CAMB direct dubbing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-url",
        type=str,
        default=None,
        help=(
            "Publicly accessible input media URL. "
            "Supported formats: MP4, MOV, MP3, WAV, and other common audio/video types."
        ),
    )
    input_group.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help=(
            "Path to a local media file to upload to CAMB. "
            "Supported formats: MP4 (video/mp4), MOV (video/quicktime), "
            "MP3 (audio/mpeg), WAV (audio/x-wav), and other common audio/video types. "
            "Content type is auto-detected from the file extension."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-file",
        required=True,
        type=Path,
        default=Path("output.mp3"),
        help="Output file path to write translated audio.",
    )
    parser.add_argument(
        "--source-language",
        type=int,
        default=1,
        help=(
            "CAMB source language ID (integer). See "
            "https://docs.camb.ai/api-reference/endpoint/get-source-languages"
        ),
    )
    parser.add_argument(
        "--target-language",
        type=int,
        default=54,
        help=(
            "CAMB target language ID (integer). See "
            "https://docs.camb.ai/api-reference/endpoint/get-target-languages"
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=120,
        help="Maximum polling attempts while waiting for dubbing completion.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=10,
        help="Polling interval in seconds.",
    )
    args = parser.parse_args()
    args.output_file = args.output_file.expanduser()
    if args.input_file is not None:
        args.input_file = args.input_file.expanduser()
    return args


def main() -> None:
    """Run CAMB end-to-end dubbing flow and save translated audio.

    Supports two input modes: ``--input-url`` for publicly accessible media
    URLs, or ``--input-file`` for local files (uploaded via CAMB Files API).

    Args:
        None.

    Returns:
        None: Submits CAMB dubbing, polls completion, downloads output audio,
        and prints progress to stdout.

    Raises:
        ValueError: If ``CAMB_API_KEY`` is missing.
        FileNotFoundError: If ``--input-file`` path does not exist.
        requests.HTTPError: If any CAMB API call or audio download call fails.
        RuntimeError: If CAMB returns malformed payloads or terminal failure state.
        TimeoutError: If task polling exceeds configured attempts.

    Example:
        ``python scripts/camb_s2s_infer.py --input-url https://example.com/input.mp3``
        ``--source-language 1 --target-language 54 -o outputs/output_es.mp3``

        ``python scripts/camb_s2s_infer.py --input-file ./clip.mp4``
        ``-o outputs/clip_es.mp3``
    """
    start_time = time.time()
    args = parse_args()

    camb_api_key = os.getenv(CAMB_API_KEY_ENV)
    if not camb_api_key:
        raise ValueError(f"{CAMB_API_KEY_ENV} environment variable not set")

    source_language_id = args.source_language
    target_language_id = args.target_language

    # Auth-only headers — Content-Type is auto-set by requests when using json=
    headers = {"x-api-key": camb_api_key}

    # Upload local file or use URL directly
    file_id = None
    input_url = args.input_url
    if args.input_file:
        file_id = upload_local_file(
            file_path=args.input_file,
            headers=headers,
        )

    task_id = submit_dub_task(
        source_language_id=source_language_id,
        target_language_id=target_language_id,
        headers=headers,
        input_url=input_url,
        file_id=file_id,
    )
    print(f"Created CAMB dubbing task_id={task_id}")

    run_id = wait_for_completion(
        task_id=task_id,
        headers=headers,
        max_attempts=args.max_attempts,
        poll_interval_seconds=args.poll_interval_seconds,
    )
    print(f"CAMB dubbing completed with run_id={run_id}")

    output_audio_url = get_output_audio_url(run_id=run_id, headers=headers)
    download_output_audio(audio_url=output_audio_url, output_file=args.output_file)
    print(f"Dubbing successful! File saved at: {args.output_file}")
    print(f"Time taken for invocation: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
