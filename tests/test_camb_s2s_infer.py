# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: S101,PLR2004,PLR0913

"""Unit tests for CAMB standalone dubbing script."""

from argparse import Namespace
from pathlib import Path
from unittest.mock import ANY
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import requests

from scripts.camb_s2s_infer import confirm_upload
from scripts.camb_s2s_infer import detect_content_type
from scripts.camb_s2s_infer import main
from scripts.camb_s2s_infer import request_upload_url
from scripts.camb_s2s_infer import submit_dub_task
from scripts.camb_s2s_infer import upload_file_to_presigned_url
from scripts.camb_s2s_infer import upload_local_file
from scripts.camb_s2s_infer import wait_for_completion

pytestmark = pytest.mark.unit


def _mock_response(
    payload: dict,
    http_error: Exception | None = None,
    status_code: int = 200,
) -> MagicMock:
    response = MagicMock()
    response.json.return_value = payload
    response.status_code = status_code
    response.text = ""
    if http_error is None:
        response.raise_for_status.return_value = None
    else:
        response.raise_for_status.side_effect = http_error
    return response


# --- detect_content_type ---


def test_detect_content_type_known() -> None:
    """Known file extensions return correct MIME types."""
    assert detect_content_type(Path("video.mp4")) == "video/mp4"
    assert detect_content_type(Path("audio.wav")) == "audio/x-wav"


def test_detect_content_type_fallback() -> None:
    """Unknown extensions fall back to application/octet-stream."""
    assert detect_content_type(Path("data.xyz123")) == "application/octet-stream"


# --- request_upload_url ---


@patch("scripts.camb_s2s_infer.requests.post")
def test_request_upload_url_happy_path(mock_post: MagicMock) -> None:
    """Request upload URL returns file_id, URL, and headers."""
    mock_post.return_value = _mock_response(
        {
            "file": {"file_id": "file-abc"},
            "upload": {"url": "https://storage.example.com/upload", "headers": {"x-tok": "v"}},
        }
    )
    file_id, url, hdrs = request_upload_url(
        filename="clip.mp4",
        content_type="video/mp4",
        headers={"x-api-key": "test"},
    )
    assert file_id == "file-abc"
    assert url == "https://storage.example.com/upload"
    assert hdrs == {"x-tok": "v"}


@patch("scripts.camb_s2s_infer.requests.post")
def test_request_upload_url_api_error(mock_post: MagicMock) -> None:
    """Request upload URL surfaces HTTP errors."""
    mock_post.return_value = _mock_response(
        {},
        http_error=requests.HTTPError("forbidden"),
    )
    with pytest.raises(requests.HTTPError, match="forbidden"):
        request_upload_url(
            filename="clip.mp4",
            content_type="video/mp4",
            headers={"x-api-key": "test"},
        )


# --- upload_file_to_presigned_url ---


@patch("scripts.camb_s2s_infer.requests.put")
def test_upload_file_to_presigned_url_success(
    mock_put: MagicMock,
    tmp_path: Path,
) -> None:
    """Successful upload does not raise."""
    test_file = tmp_path / "clip.mp4"
    test_file.write_bytes(b"fake video data")
    mock_put.return_value = _mock_response({}, status_code=200)
    # Should not raise
    upload_file_to_presigned_url(
        file_path=test_file,
        upload_url="https://storage.example.com/upload",
        upload_headers={"x-tok": "v"},
    )


@patch("scripts.camb_s2s_infer.requests.put")
def test_upload_file_to_presigned_url_failure(
    mock_put: MagicMock,
    tmp_path: Path,
) -> None:
    """Failed upload raises RuntimeError."""
    test_file = tmp_path / "clip.mp4"
    test_file.write_bytes(b"fake video data")
    mock_put.return_value = _mock_response({}, status_code=403)
    with pytest.raises(RuntimeError, match="File upload failed with status 403"):
        upload_file_to_presigned_url(
            file_path=test_file,
            upload_url="https://storage.example.com/upload",
            upload_headers={},
        )


# --- confirm_upload ---


@patch("scripts.camb_s2s_infer.requests.post")
def test_confirm_upload_happy_path(mock_post: MagicMock) -> None:
    """Confirm upload calls complete endpoint without error."""
    mock_post.return_value = _mock_response({"status": "complete"})
    confirm_upload(file_id="file-abc", headers={"x-api-key": "test"})
    mock_post.assert_called_once()
    assert "file-abc" in mock_post.call_args[0][0]


# --- upload_local_file ---


def test_upload_local_file_file_not_found() -> None:
    """Upload raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError, match="Input file not found"):
        upload_local_file(
            file_path=Path("/nonexistent/file.mp4"),
            headers={"x-api-key": "test"},
        )


@patch("scripts.camb_s2s_infer.confirm_upload")
@patch("scripts.camb_s2s_infer.upload_file_to_presigned_url")
@patch("scripts.camb_s2s_infer.request_upload_url")
def test_upload_local_file_happy_path(
    mock_request_url: MagicMock,
    mock_upload: MagicMock,
    mock_confirm: MagicMock,
    tmp_path: Path,
) -> None:
    """Upload local file chains request, upload, confirm and returns file_id."""
    test_file = tmp_path / "clip.mp4"
    test_file.write_bytes(b"fake video data")
    mock_request_url.return_value = ("file-xyz", "https://storage/upload", {"h": "v"})

    file_id = upload_local_file(
        file_path=test_file,
        headers={"x-api-key": "test"},
    )
    assert file_id == "file-xyz"
    mock_request_url.assert_called_once()
    mock_upload.assert_called_once_with(
        file_path=test_file,
        upload_url="https://storage/upload",
        upload_headers={"h": "v"},
    )
    mock_confirm.assert_called_once_with(
        file_id="file-xyz",
        headers={"x-api-key": "test"},
    )


# --- submit_dub_task ---


@patch("scripts.camb_s2s_infer.requests.post")
def test_submit_dub_task_happy_path(mock_post: MagicMock) -> None:
    """Submit task parses returned task id."""
    mock_post.return_value = _mock_response({"task_id": "task-123"})
    task_id = submit_dub_task(
        source_language_id=1,
        target_language_id=54,
        headers={"x-api-key": "test"},
        input_url="https://example.com/input.mp3",
    )
    assert task_id == "task-123"


@patch("scripts.camb_s2s_infer.requests.post")
def test_submit_dub_task_with_file_id(mock_post: MagicMock) -> None:
    """Submit task sends file_id instead of file_url."""
    mock_post.return_value = _mock_response({"task_id": "task-456"})
    task_id = submit_dub_task(
        source_language_id=1,
        target_language_id=54,
        headers={"x-api-key": "test"},
        file_id="file-abc",
    )
    assert task_id == "task-456"
    sent_payload = mock_post.call_args[1]["json"]
    assert "file_id" in sent_payload
    assert "file_url" not in sent_payload


def test_submit_dub_task_no_input_raises() -> None:
    """Submit task raises ValueError when neither input is provided."""
    with pytest.raises(ValueError, match="Exactly one of input_url or file_id"):
        submit_dub_task(
            source_language_id=1,
            target_language_id=54,
            headers={"x-api-key": "test"},
        )


def test_submit_dub_task_both_inputs_raises() -> None:
    """Submit task raises ValueError when both inputs are provided."""
    with pytest.raises(ValueError, match="Exactly one of input_url or file_id"):
        submit_dub_task(
            source_language_id=1,
            target_language_id=54,
            headers={"x-api-key": "test"},
            input_url="https://example.com/input.mp3",
            file_id="file-abc",
        )


@patch("scripts.camb_s2s_infer.requests.post")
def test_submit_dub_task_api_error(mock_post: MagicMock) -> None:
    """Submit task surfaces HTTP errors."""
    mock_post.return_value = _mock_response(
        {},
        http_error=requests.HTTPError("bad request"),
    )
    with pytest.raises(requests.HTTPError, match="bad request"):
        submit_dub_task(
            source_language_id=1,
            target_language_id=54,
            headers={"x-api-key": "test"},
            input_url="https://example.com/input.mp3",
        )


# --- wait_for_completion ---


@patch("scripts.camb_s2s_infer.time.sleep")
@patch("scripts.camb_s2s_infer.requests.get")
def test_wait_for_completion_happy_path(mock_get: MagicMock, mock_sleep: MagicMock) -> None:
    """Polling returns run id once status reaches SUCCESS."""
    mock_get.side_effect = [
        _mock_response({"status": "PENDING"}),
        _mock_response({"status": "SUCCESS", "run_id": 42}),
    ]
    run_id = wait_for_completion(
        task_id="task-1",
        headers={"x-api-key": "test"},
        max_attempts=5,
        poll_interval_seconds=1,
    )
    assert run_id == 42
    assert mock_sleep.call_count == 1


@patch("scripts.camb_s2s_infer.time.sleep")
@patch("scripts.camb_s2s_infer.requests.get")
def test_wait_for_completion_timeout(mock_get: MagicMock, mock_sleep: MagicMock) -> None:
    """Polling raises timeout when task never reaches terminal state."""
    mock_get.return_value = _mock_response({"status": "PENDING"})
    with pytest.raises(TimeoutError, match="timed out"):
        wait_for_completion(
            task_id="task-1",
            headers={"x-api-key": "test"},
            max_attempts=2,
            poll_interval_seconds=1,
        )
    assert mock_sleep.call_count == 2


# --- main (end-to-end) ---


@patch("scripts.camb_s2s_infer.download_output_audio")
@patch("scripts.camb_s2s_infer.get_output_audio_url")
@patch("scripts.camb_s2s_infer.wait_for_completion")
@patch("scripts.camb_s2s_infer.submit_dub_task")
@patch("scripts.camb_s2s_infer.parse_args")
@patch("scripts.camb_s2s_infer.os.getenv")
def test_main_happy_path(
    mock_getenv: MagicMock,
    mock_parse_args: MagicMock,
    mock_submit: MagicMock,
    mock_wait: MagicMock,
    mock_get_audio_url: MagicMock,
    mock_download: MagicMock,
) -> None:
    """Main flow submits, polls, gets audio URL, and downloads output."""
    mock_getenv.return_value = "test-key"
    mock_parse_args.return_value = Namespace(
        input_url="https://example.com/input.mp3",
        input_file=None,
        output_file=Path("outputs/test.mp3"),
        source_language=1,
        target_language=54,
        max_attempts=10,
        poll_interval_seconds=1,
    )
    mock_submit.return_value = "task-1"
    mock_wait.return_value = 99
    mock_get_audio_url.return_value = "https://example.com/output.mp3"

    main()

    mock_submit.assert_called_once_with(
        source_language_id=1,
        target_language_id=54,
        headers=ANY,
        input_url="https://example.com/input.mp3",
        file_id=None,
    )
    mock_wait.assert_called_once()
    mock_get_audio_url.assert_called_once_with(run_id=99, headers=ANY)
    mock_download.assert_called_once_with(
        audio_url="https://example.com/output.mp3",
        output_file=Path("outputs/test.mp3"),
    )


@patch("scripts.camb_s2s_infer.download_output_audio")
@patch("scripts.camb_s2s_infer.get_output_audio_url")
@patch("scripts.camb_s2s_infer.wait_for_completion")
@patch("scripts.camb_s2s_infer.submit_dub_task")
@patch("scripts.camb_s2s_infer.upload_local_file")
@patch("scripts.camb_s2s_infer.parse_args")
@patch("scripts.camb_s2s_infer.os.getenv")
def test_main_happy_path_file_upload(
    mock_getenv: MagicMock,
    mock_parse_args: MagicMock,
    mock_upload: MagicMock,
    mock_submit: MagicMock,
    mock_wait: MagicMock,
    mock_get_audio_url: MagicMock,
    mock_download: MagicMock,
) -> None:
    """Main flow with --input-file uploads then submits with file_id."""
    mock_getenv.return_value = "test-key"
    mock_parse_args.return_value = Namespace(
        input_url=None,
        input_file=Path("/data/media/clip.mp4"),
        output_file=Path("outputs/test.mp3"),
        source_language=1,
        target_language=54,
        max_attempts=10,
        poll_interval_seconds=1,
    )
    mock_upload.return_value = "file-xyz"
    mock_submit.return_value = "task-1"
    mock_wait.return_value = 99
    mock_get_audio_url.return_value = "https://example.com/output.mp3"

    main()

    mock_upload.assert_called_once_with(
        file_path=Path("/data/media/clip.mp4"),
        headers=ANY,
    )
    mock_submit.assert_called_once_with(
        source_language_id=1,
        target_language_id=54,
        headers=ANY,
        input_url=None,
        file_id="file-xyz",
    )
    mock_download.assert_called_once()


@patch("scripts.camb_s2s_infer.parse_args")
@patch("scripts.camb_s2s_infer.os.getenv")
def test_main_missing_api_key_raises(
    mock_getenv: MagicMock,
    mock_parse_args: MagicMock,
) -> None:
    """Main fails early when CAMB_API_KEY is missing."""
    mock_getenv.return_value = None
    mock_parse_args.return_value = Namespace(
        input_url="https://example.com/input.mp3",
        input_file=None,
        output_file=Path("outputs/test.mp3"),
        source_language=1,
        target_language=54,
        max_attempts=10,
        poll_interval_seconds=1,
    )
    with pytest.raises(ValueError, match="CAMB_API_KEY environment variable not set"):
        main()
