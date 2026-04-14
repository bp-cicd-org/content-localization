# ruff: noqa
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from typing import Iterator, NoReturn
from unittest.mock import MagicMock
from unittest.mock import patch

from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechRequest  # type: ignore

from common.buffers import Buffer
from common.buffers import RequestIteratorFromBuffer
from s2s_service.riva_utils.asr import GRPCRIVAStreamingASTClient
from s2s_service.riva_utils.servers import RivaASRServer


class DummyContext:
    def __init__(self) -> None:
        self.aborted = False
        self.abort_args = None

    def abort(self, code, msg) -> NoReturn:
        self.aborted = True
        self.abort_args = (code, msg)
        raise Exception(f"Aborted: {code}, {msg}")


class TestGRPCRIVAStreamingASTClient(unittest.TestCase):
    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50051")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    @patch("s2s_service.riva_utils.servers.riva.client.ASRService")
    @patch("s2s_service.riva_utils.servers.riva.client.Auth")
    def test_init(self, mock_auth, mock_asr, mock_healthy, mock_call):
        server = RivaASRServer("localhost", 50051)
        client = GRPCRIVAStreamingASTClient(server)
        self.assertEqual(client.server, server)
        mock_auth.assert_called_once()
        mock_asr.assert_called_once()

    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50051")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    def test_is_healthy_success(self, mock_healthy, mock_call):
        server = RivaASRServer("localhost", 50051)
        client = GRPCRIVAStreamingASTClient(server)
        self.assertTrue(client.is_healthy())
        # is_healthy may be called more than once due to constructor logic
        self.assertGreaterEqual(mock_healthy.call_count, 1)

    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50051")
    @patch("common.servers.GRPCServer.is_healthy", side_effect=ConnectionError("fail"))
    def test_is_healthy_failure(self, mock_healthy, mock_call):
        server = RivaASRServer("localhost", 50051)
        client = GRPCRIVAStreamingASTClient(server)
        with self.assertRaises(ConnectionError):
            client.is_healthy()

    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50051")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    @patch("s2s_service.riva_utils.servers.RivaASRServer.from_string")
    @patch("s2s_service.riva_utils.servers.riva.client.ASRService")
    @patch("s2s_service.riva_utils.servers.riva.client.Auth")
    def test_from_string(self, mock_auth, mock_asr, mock_from_string, mock_healthy, mock_call):
        mock_server = RivaASRServer("localhost", 50051)
        mock_from_string.return_value = mock_server
        client = GRPCRIVAStreamingASTClient.from_string("localhost:50051")
        self.assertIsInstance(client, GRPCRIVAStreamingASTClient)
        self.assertEqual(client.server, mock_server)

    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50051")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    @patch("s2s_service.riva_utils.servers.riva.client.ASRService")
    @patch("s2s_service.riva_utils.servers.riva.client.Auth")
    def test_call_success(self, mock_auth, mock_asr, mock_healthy, mock_call):
        # Setup
        server = RivaASRServer("localhost", 50051)
        client = GRPCRIVAStreamingASTClient(server)
        context = DummyContext()
        request_id = "req-1"
        # Mock streaming_response_generator
        mock_asr_instance = mock_asr.return_value
        mock_response = MagicMock()
        mock_result = MagicMock()
        mock_result.is_final = True
        mock_result.alternatives = [MagicMock(transcript="hello world")]
        mock_response.results = [mock_result]
        mock_asr_instance.streaming_response_generator.return_value = [mock_response]

        # Call
        def req_iter():
            yield SpeechToSpeechRequest(audio_data=b"audio")

        output_buffer: Buffer[str] = Buffer()
        client(
            req_iter(),
            output_buffer,
            context,
            request_id,
        )
        transcripts = list(RequestIteratorFromBuffer(output_buffer, poll_timeout=0.01))
        self.assertIn("hello world", transcripts)

    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50051")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    @patch("s2s_service.riva_utils.servers.riva.client.ASRService")
    @patch("s2s_service.riva_utils.servers.riva.client.Auth")
    def test_call_exception(self, mock_auth, mock_asr, mock_healthy, mock_call):
        server = RivaASRServer("localhost", 50051)
        client = GRPCRIVAStreamingASTClient(server)
        context = DummyContext()
        request_id = "req-1"
        # Mock streaming_response_generator to raise
        mock_asr_instance = mock_asr.return_value
        mock_asr_instance.streaming_response_generator.side_effect = Exception("fail")

        def req_iter():
            yield SpeechToSpeechRequest(audio_data=b"audio")

        output_buffer: Buffer[str] = Buffer()
        with self.assertRaises(Exception):
            client(req_iter(), output_buffer, context, request_id)
        self.assertTrue(context.aborted)

    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50051")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    @patch("s2s_service.riva_utils.servers.riva.client.ASRService")
    @patch("s2s_service.riva_utils.servers.riva.client.Auth")
    def test_call_multiple_chunks(self, mock_auth, mock_asr, mock_healthy, mock_call):
        server = RivaASRServer("localhost", 50051)
        client = GRPCRIVAStreamingASTClient(server)
        context = DummyContext()
        request_id = "req-1"
        # Mock streaming_response_generator to yield 5 responses
        mock_asr_instance = mock_asr.return_value
        responses = []
        for i in range(5):
            mock_result = MagicMock()
            mock_result.is_final = True
            mock_result.alternatives = [MagicMock(transcript=f"chunk-{i}")]
            mock_response = MagicMock()
            mock_response.results = [mock_result]
            responses.append(mock_response)
        mock_asr_instance.streaming_response_generator.return_value = responses

        def req_iter():
            yield SpeechToSpeechRequest(audio_data=b"audio")

        output_buffer: Buffer[str] = Buffer()
        client(req_iter(), output_buffer, context, request_id)
        transcripts = list(RequestIteratorFromBuffer(output_buffer, poll_timeout=0.01))
        self.assertEqual(transcripts, [f"chunk-{i}" for i in range(5)])

    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50051")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    @patch("s2s_service.riva_utils.asr.riva.client.StreamingRecognitionConfig")
    @patch("s2s_service.riva_utils.asr.riva.client.RecognitionConfig")
    @patch("s2s_service.riva_utils.asr.riva.client.AudioEncoding")
    @patch("s2s_service.riva_utils.asr.riva.client.add_custom_configuration_to_config")
    def test_config_ast_stream_exception(
        self,
        mock_add_custom,
        mock_audio_encoding,
        mock_recog,
        mock_streaming,
        mock_healthy,
        mock_call,
    ):
        server = RivaASRServer("localhost", 50051)
        client = GRPCRIVAStreamingASTClient(server)
        context = DummyContext()
        request_id = "req-1"
        # Simulate exception in StreamingRecognitionConfig
        mock_streaming.side_effect = Exception("fail")
        with self.assertRaises(Exception):
            client._config_ast_stream(context, request_id)
        self.assertTrue(context.aborted)

    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50051")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    @patch("s2s_service.riva_utils.servers.riva.client.ASRService")
    @patch("s2s_service.riva_utils.servers.riva.client.Auth")
    def test_impl_exception(self, mock_auth, mock_asr, mock_healthy, mock_call):
        server = RivaASRServer("localhost", 50051)
        client = GRPCRIVAStreamingASTClient(server)
        context = DummyContext()
        request_id = "req-1"
        streaming_config = MagicMock()
        client._config_ast_stream = MagicMock(return_value=streaming_config)
        # Simulate exception in streaming_response_generator
        mock_asr_instance = mock_asr.return_value
        mock_asr_instance.streaming_response_generator.side_effect = Exception("fail")

        def req_iter():
            yield SpeechToSpeechRequest(audio_data=b"audio")

        output_buffer: Buffer[str] = Buffer()
        with self.assertRaises(Exception):
            client._impl(req_iter(), output_buffer, context, request_id)
        self.assertTrue(context.aborted)

    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50051")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    @patch("s2s_service.riva_utils.servers.riva.client.ASRService")
    @patch("s2s_service.riva_utils.servers.riva.client.Auth")
    def test_impl_empty_results(self, mock_auth, mock_asr, mock_healthy, mock_call):
        server = RivaASRServer("localhost", 50051)
        client = GRPCRIVAStreamingASTClient(server)
        context = DummyContext()
        request_id = "req-1"
        streaming_config = MagicMock()
        client._config_ast_stream = MagicMock(return_value=streaming_config)
        mock_asr_instance = mock_asr.return_value
        # Simulate asr_response.results is empty
        mock_response = MagicMock()
        mock_response.results = []
        mock_asr_instance.streaming_response_generator.return_value = [mock_response]

        def req_iter():
            yield SpeechToSpeechRequest(audio_data=b"audio")

        output_buffer: Buffer[str] = Buffer()
        client._impl(req_iter(), output_buffer, context, request_id)
        output_buffer.done = True
        transcripts = list(RequestIteratorFromBuffer(output_buffer, poll_timeout=0.01))
        self.assertEqual(transcripts, [])

    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50051")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    @patch("s2s_service.riva_utils.servers.riva.client.ASRService")
    @patch("s2s_service.riva_utils.servers.riva.client.Auth")
    def test_impl_empty_alternatives(
        self,
        mock_auth: MagicMock,
        mock_asr: MagicMock,
        mock_healthy: MagicMock,
        mock_call: MagicMock,
    ) -> None:
        server = RivaASRServer("localhost", 50051)
        client = GRPCRIVAStreamingASTClient(server)
        context = DummyContext()
        request_id = "req-1"
        streaming_config = MagicMock()
        client._config_ast_stream = MagicMock(return_value=streaming_config)
        mock_asr_instance = mock_asr.return_value
        # Simulate result.alternatives is empty
        mock_result = MagicMock()
        mock_result.alternatives = []
        mock_result.is_final = True
        mock_response = MagicMock()
        mock_response.results = [mock_result]
        mock_asr_instance.streaming_response_generator.return_value = [mock_response]

        def req_iter() -> Iterator[SpeechToSpeechRequest]:
            yield SpeechToSpeechRequest(audio_data=b"audio")

        output_buffer: Buffer[str] = Buffer()
        # Empty alternatives are now gracefully skipped (no IndexError).
        client._impl(req_iter(), output_buffer, context, request_id)
        output_buffer.done = True
        transcripts = list(RequestIteratorFromBuffer(output_buffer, poll_timeout=0.01))
        self.assertEqual(transcripts, [])

    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50051")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    @patch("s2s_service.riva_utils.servers.riva.client.ASRService")
    @patch("s2s_service.riva_utils.servers.riva.client.Auth")
    def test_impl_not_final(
        self,
        mock_auth: MagicMock,
        mock_asr: MagicMock,
        mock_healthy: MagicMock,
        mock_call: MagicMock,
    ) -> None:
        server = RivaASRServer("localhost", 50051)
        client = GRPCRIVAStreamingASTClient(server)
        context = DummyContext()
        request_id = "req-1"
        streaming_config = MagicMock()
        client._config_ast_stream = MagicMock(return_value=streaming_config)
        mock_asr_instance = mock_asr.return_value
        # Simulate result.is_final is False
        mock_result = MagicMock()
        mock_result.alternatives = [MagicMock(transcript="should not yield")]
        mock_result.is_final = False
        mock_response = MagicMock()
        mock_response.results = [mock_result]
        mock_asr_instance.streaming_response_generator.return_value = [mock_response]

        def req_iter() -> Iterator[SpeechToSpeechRequest]:
            yield SpeechToSpeechRequest(audio_data=b"audio")

        output_buffer: Buffer[str] = Buffer()
        client._impl(req_iter(), output_buffer, context, request_id)
        output_buffer.done = True
        transcripts = list(RequestIteratorFromBuffer(output_buffer, poll_timeout=0.01))
        self.assertEqual(transcripts, [])

    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50051")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    @patch("s2s_service.riva_utils.asr.riva.client.StreamingRecognitionConfig")
    @patch("s2s_service.riva_utils.asr.riva.client.RecognitionConfig")
    @patch("s2s_service.riva_utils.asr.riva.client.AudioEncoding")
    @patch("s2s_service.riva_utils.asr.riva.client.add_custom_configuration_to_config")
    def test_config_ast_stream_normal(
        self,
        mock_add_custom: MagicMock,
        mock_audio_encoding: MagicMock,
        mock_recog: MagicMock,
        mock_streaming: MagicMock,
        mock_healthy: MagicMock,
        mock_call: MagicMock,
    ) -> None:
        server = RivaASRServer("localhost", 50051)
        client = GRPCRIVAStreamingASTClient(server)
        context = DummyContext()
        request_id = "req-1"
        mock_streaming_instance = MagicMock()
        mock_streaming.return_value = mock_streaming_instance
        result = client._config_ast_stream(context, request_id)
        self.assertEqual(result, mock_streaming_instance)

    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50051")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    @patch("s2s_service.riva_utils.servers.riva.client.ASRService")
    @patch("s2s_service.riva_utils.servers.riva.client.Auth")
    def test_impl_exception_return(self, mock_auth, mock_asr, mock_healthy, mock_call):
        server = RivaASRServer("localhost", 50051)
        client = GRPCRIVAStreamingASTClient(server)
        context = DummyContext()
        request_id = "req-1"
        streaming_config = MagicMock()
        client._config_ast_stream = MagicMock(return_value=streaming_config)
        # Simulate exception in streaming_response_generator
        mock_asr_instance = mock_asr.return_value
        mock_asr_instance.streaming_response_generator.side_effect = Exception("fail")

        def req_iter():
            yield SpeechToSpeechRequest(audio_data=b"audio")

        # The function should abort and return (yield nothing)
        output_buffer: Buffer[str] = Buffer()
        with self.assertRaises(Exception):
            client._impl(req_iter(), output_buffer, context, request_id)
        self.assertTrue(context.aborted)

    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50051")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    @patch("s2s_service.riva_utils.servers.riva.client.ASRService")
    @patch("s2s_service.riva_utils.servers.riva.client.Auth")
    def test_impl_exception_return_no_raise(self, mock_auth, mock_asr, mock_healthy, mock_call):
        server = RivaASRServer("localhost", 50051)
        client = GRPCRIVAStreamingASTClient(server)

        # Patch context.abort to NOT raise
        class NoRaiseContext(DummyContext):
            def abort(self, code, msg):
                self.aborted = True
                self.abort_args = (code, msg)
                # Do not raise

        context = NoRaiseContext()
        request_id = "req-1"
        streaming_config = MagicMock()
        client._config_ast_stream = MagicMock(return_value=streaming_config)
        # Simulate exception in streaming_response_generator
        mock_asr_instance = mock_asr.return_value
        mock_asr_instance.streaming_response_generator.side_effect = Exception("fail")

        def req_iter():
            yield SpeechToSpeechRequest(audio_data=b"audio")

        output_buffer: Buffer[str] = Buffer()
        # Should not raise, should just return (yield nothing)
        client._impl(req_iter(), output_buffer, context, request_id)
        self.assertTrue(context.aborted)
        output_buffer.done = True
        transcripts = list(RequestIteratorFromBuffer(output_buffer, poll_timeout=0.01))
        self.assertEqual(transcripts, [])


if __name__ == "__main__":
    unittest.main()
