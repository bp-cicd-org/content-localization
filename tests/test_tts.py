# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from common.buffers import Buffer
from common.buffers import RequestIteratorFromBuffer
from s2s_service.riva_utils.servers import RivaTTSServer
from s2s_service.riva_utils.tts import GRPCRIVATTSClient


class DummyContext:
    def __init__(self):
        self.aborted = False
        self.abort_args = None

    def abort(self, code, msg):
        self.aborted = True
        self.abort_args = (code, msg)
        raise Exception(f"Aborted: {code}, {msg}")


class TestGRPCRIVATTSClient(unittest.TestCase):
    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50053")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    @patch("s2s_service.riva_utils.servers.riva.client.SpeechSynthesisService")
    @patch("s2s_service.riva_utils.servers.riva.client.Auth")
    def test_init(self, mock_auth, mock_tts, mock_healthy, mock_call):
        server = RivaTTSServer("localhost", 50053)
        client = GRPCRIVATTSClient(server)
        self.assertEqual(client.server, server)
        mock_auth.assert_called_once()
        mock_tts.assert_called_once()

    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50053")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    def test_is_healthy_success(self, mock_healthy, mock_call):
        server = RivaTTSServer("localhost", 50053)
        client = GRPCRIVATTSClient(server)
        self.assertTrue(client.is_healthy())
        self.assertGreaterEqual(mock_healthy.call_count, 1)

    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50053")
    @patch("common.servers.GRPCServer.is_healthy", side_effect=ConnectionError("fail"))
    def test_is_healthy_failure(self, mock_healthy, mock_call):
        server = RivaTTSServer("localhost", 50053)
        client = GRPCRIVATTSClient(server)
        with self.assertRaises(ConnectionError):
            client.is_healthy()

    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50053")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    @patch("s2s_service.riva_utils.servers.RivaTTSServer.from_string")
    @patch("s2s_service.riva_utils.servers.riva.client.SpeechSynthesisService")
    @patch("s2s_service.riva_utils.servers.riva.client.Auth")
    def test_from_string(self, mock_auth, mock_tts, mock_from_string, mock_healthy, mock_call):
        mock_server = RivaTTSServer("localhost", 50053)
        mock_from_string.return_value = mock_server
        client = GRPCRIVATTSClient.from_string("localhost:50053")
        self.assertIsInstance(client, GRPCRIVATTSClient)
        self.assertEqual(client.server, mock_server)

    @patch("s2s_service.riva_utils.tts.GRPCRIVATTSClient.is_healthy", return_value=True)
    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50053")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    @patch("s2s_service.riva_utils.servers.riva.client.SpeechSynthesisService")
    @patch("s2s_service.riva_utils.servers.riva.client.Auth")
    def test_call_success(self, mock_auth, mock_tts, mock_healthy, mock_call, mock_client_healthy):
        server = RivaTTSServer("localhost", 50053)
        client = GRPCRIVATTSClient(server)
        context = DummyContext()
        request_id = "req-1"
        # Patch the correct attribute
        mock_tts_instance = mock_tts.return_value
        client._tts_service = mock_tts_instance
        mock_response = [MagicMock(audio=b"audio")]
        mock_tts_instance.stub.SynthesizeOnline.return_value = mock_response

        def text_iter():
            yield "hello"

        output_buffer: Buffer[object] = Buffer()
        client(text_iter(), output_buffer, context, request_id)
        output_buffer.done = True
        result = list(RequestIteratorFromBuffer(output_buffer, poll_timeout=0.01))
        self.assertEqual(result, mock_response)

    @patch("s2s_service.riva_utils.tts.GRPCRIVATTSClient.is_healthy", return_value=True)
    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50053")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    @patch("s2s_service.riva_utils.servers.riva.client.SpeechSynthesisService")
    @patch("s2s_service.riva_utils.servers.riva.client.Auth")
    def test_call_exception(
        self, mock_auth, mock_tts, mock_healthy, mock_call, mock_client_healthy
    ):
        server = RivaTTSServer("localhost", 50053)
        client = GRPCRIVATTSClient(server)
        context = DummyContext()
        request_id = "req-1"
        # Patch the correct attribute
        mock_tts_instance = mock_tts.return_value
        client._tts_service = mock_tts_instance
        mock_tts_instance.stub.SynthesizeOnline.side_effect = Exception("fail")

        def text_iter():
            yield "hello"

        output_buffer: Buffer[object] = Buffer()
        with self.assertRaises(Exception):
            client(text_iter(), output_buffer, context, request_id)
        self.assertTrue(context.aborted)

    @patch("s2s_service.riva_utils.tts.GRPCRIVATTSClient.is_healthy", return_value=True)
    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50053")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    @patch("s2s_service.riva_utils.servers.riva.client.SpeechSynthesisService")
    @patch("s2s_service.riva_utils.servers.riva.client.Auth")
    def test_call_multiple_chunks(
        self, mock_auth, mock_tts, mock_healthy, mock_call, mock_client_healthy
    ):
        server = RivaTTSServer("localhost", 50053)
        client = GRPCRIVATTSClient(server)
        context = DummyContext()
        request_id = "req-1"
        # Patch the correct attribute
        mock_tts_instance = mock_tts.return_value
        client._tts_service = mock_tts_instance
        responses = [MagicMock(audio=f"audio-{i}".encode()) for i in range(5)]
        mock_tts_instance.stub.SynthesizeOnline.side_effect = [[resp] for resp in responses]

        def text_iter():
            for i in range(5):
                yield f"chunk-{i}"

        output_buffer: Buffer[object] = Buffer()
        client(text_iter(), output_buffer, context, request_id)
        output_buffer.done = True
        result = list(RequestIteratorFromBuffer(output_buffer, poll_timeout=0.01))
        self.assertEqual(result, responses)


if __name__ == "__main__":
    unittest.main()
