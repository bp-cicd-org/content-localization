#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for NIM client classes."""

import unittest
from collections.abc import Iterator
from unittest.mock import MagicMock

import grpc
import pytest
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    ActiveSpeakerDetectionResult,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    DetectActiveSpeakerResponse,
)
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncResponse
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechResponse

from common.buffers import Buffer
from common.buffers import RequestIteratorFromBuffer
from common.nims import ActiveSpeakerDetectionClient
from common.nims import LipsyncClient
from common.nims import SpeechToSpeechClient

pytestmark = pytest.mark.unit


class _FakeServer:
    def __init__(self, responses: list[object]) -> None:
        self.stub = None
        self._responses = responses
        self.create_called = 0

    def is_healthy(self) -> bool:
        return True

    def create_server(self) -> None:
        self.stub = object()
        self.create_called += 1

    def get_response_iterator(self, request_iterator: Iterator[object]) -> Iterator[object]:
        _ = list(request_iterator)
        return iter(self._responses)


class TestNimClients(unittest.TestCase):
    """Unit tests for NIM client buffer behavior."""

    def _run_client(self, client_cls: type, responses: list[object]) -> list[object]:
        server = _FakeServer(responses=responses)
        client = client_cls(server)
        context = MagicMock(spec=grpc.ServicerContext)
        output_buffer: Buffer[object] = Buffer()

        def request_iter() -> Iterator[object]:
            yield object()
            yield object()

        client(
            request_iterator=request_iter(),
            output_buffer=output_buffer,
            context=context,
            request_id="r1",
        )
        self.assertEqual(server.create_called, 1)
        self.assertTrue(output_buffer.done)
        return list(RequestIteratorFromBuffer(output_buffer, poll_timeout=0.01))

    def _run_client_with_stub(self, client_cls: type, responses: list[object]) -> list[object]:
        server = _FakeServer(responses=responses)
        server.stub = object()
        client = client_cls(server)
        context = MagicMock(spec=grpc.ServicerContext)
        output_buffer: Buffer[object] = Buffer()

        def request_iter() -> Iterator[object]:
            yield object()

        client(
            request_iterator=request_iter(),
            output_buffer=output_buffer,
            context=context,
            request_id="r2",
        )
        self.assertEqual(server.create_called, 0)
        self.assertTrue(output_buffer.done)
        return list(RequestIteratorFromBuffer(output_buffer, poll_timeout=0.01))

    def test_s2s_client_streams_responses(self) -> None:
        """Speech-to-Speech client streams all responses into buffer."""
        responses = [
            SpeechToSpeechResponse(audio_data=b"chunk1"),
            SpeechToSpeechResponse(audio_data=b"chunk2"),
            SpeechToSpeechResponse(audio_data=b"chunk3"),
        ]
        buffered = self._run_client(SpeechToSpeechClient, responses)
        self.assertEqual(len(buffered), 3)

    def test_asd_client_streams_responses(self) -> None:
        """Active speaker detection client streams all responses into buffer."""
        responses = [
            DetectActiveSpeakerResponse(
                active_speaker_detection_result=ActiveSpeakerDetectionResult(frame_id=0)
            ),
            DetectActiveSpeakerResponse(
                active_speaker_detection_result=ActiveSpeakerDetectionResult(frame_id=1)
            ),
        ]
        buffered = self._run_client(ActiveSpeakerDetectionClient, responses)
        self.assertEqual(len(buffered), 2)

    def test_lipsync_client_streams_responses(self) -> None:
        """LipSync client streams all responses into buffer."""
        responses = [LipsyncResponse(video_file_data=b"frame")]
        buffered = self._run_client(LipsyncClient, responses)
        self.assertEqual(len(buffered), 1)

    def test_clients_skip_create_server_when_stub_exists(self) -> None:
        """Clients do not recreate server when stub already exists."""
        responses = [
            SpeechToSpeechResponse(audio_data=b"chunk1"),
            SpeechToSpeechResponse(audio_data=b"chunk2"),
        ]
        buffered = self._run_client_with_stub(SpeechToSpeechClient, responses)
        self.assertEqual(len(buffered), 2)

    def test_clients_filter_keepalive_responses(self) -> None:
        """Clients drop keep-alive responses before writing to output buffers."""
        s2s_keepalive = SpeechToSpeechResponse()
        s2s_keepalive.keepalive.SetInParent()
        s2s_audio = SpeechToSpeechResponse(audio_data=b"audio")
        s2s_buffered = self._run_client(SpeechToSpeechClient, [s2s_keepalive, s2s_audio])
        self.assertEqual(len(s2s_buffered), 1)
        self.assertTrue(s2s_buffered[0].HasField("audio_data"))

        asd_keepalive = DetectActiveSpeakerResponse()
        asd_keepalive.keepalive.SetInParent()
        asd_result = DetectActiveSpeakerResponse(
            active_speaker_detection_result=ActiveSpeakerDetectionResult(frame_id=0)
        )
        asd_buffered = self._run_client(ActiveSpeakerDetectionClient, [asd_keepalive, asd_result])
        self.assertEqual(len(asd_buffered), 1)
        self.assertEqual(asd_buffered[0].active_speaker_detection_result.frame_id, 0)

        lipsync_keepalive = LipsyncResponse()
        lipsync_keepalive.keepalive.SetInParent()
        lipsync_video = LipsyncResponse(video_file_data=b"video")
        lipsync_buffered = self._run_client(LipsyncClient, [lipsync_keepalive, lipsync_video])
        self.assertEqual(len(lipsync_buffered), 1)
        self.assertEqual(lipsync_buffered[0].video_file_data, b"video")


if __name__ == "__main__":
    unittest.main()
