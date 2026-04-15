# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for S2S services - testing full end-to-end flows."""

import tempfile
import unittest
import wave
from typing import NoReturn
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechConfig
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechRequest
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechResponse

from s2s_service.riva_utils.servers import RivaASRServer
from s2s_service.riva_utils.servers import RivaTTSServer


class DummyContext:
    """Dummy gRPC context for testing."""

    def __init__(self) -> None:
        self.aborted = False
        self.abort_args = None

    def abort(self, code, msg) -> NoReturn:
        self.aborted = True
        self.abort_args = (code, msg)
        raise Exception(f"Aborted: {code}, {msg}")

    def peer(self) -> str:
        return "test-peer"


def create_test_wav_file(duration_seconds: float = 1.0, sample_rate: int = 16000) -> str:
    """Create a test WAV file.

    Args:
        duration_seconds: Duration of the audio file in seconds.
        sample_rate: Sample rate in Hz.

    Returns:
        Path to the temporary WAV file.
    """
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file.close()

    # Generate test audio (simple sine wave)
    frequency = 440  # Hz
    duration = duration_seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * frequency * t) * 32767
    audio_data = audio_data.astype(np.int16)

    # Write WAV file
    with wave.open(temp_file.name, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_data.tobytes())

    return temp_file.name


@patch("s2s_service.riva_utils.s2s.GRPCRIVATTSClient")
@patch("s2s_service.riva_utils.s2s.GRPCRIVAStreamingASTClient")
@patch("common.servers.GRPCServer.__call__", return_value="localhost:50051")
@patch("common.servers.GRPCServer.is_healthy", return_value=True)
class TestS2SStreamingIntegration(unittest.TestCase):
    """Integration tests for RIVA Streaming S2S service."""

    def test_full_streaming_flow(
        self, mock_healthy, mock_call, mock_ast_client_class, mock_tts_client_class
    ):
        """Test full streaming S2S flow from audio input to audio output."""
        from s2s_service.riva_utils.s2s import S2SRIVAStreamingService

        # Create service
        ast_server = RivaASRServer("localhost", 50051)
        tts_server = RivaTTSServer("localhost", 50052)

        # Mock ASR and TTS clients to pass health checks
        mock_ast_client = MagicMock()
        mock_ast_client_class.return_value = mock_ast_client
        mock_ast_client.is_healthy.return_value = True
        mock_ast_client.return_value = mock_ast_client

        mock_tts_client = MagicMock()
        mock_tts_client_class.return_value = mock_tts_client
        mock_tts_client.is_healthy.return_value = True
        mock_tts_client.sample_rate_hz = 16000

        def _mock_tts(request_iterator, output_buffer, **_):
            output_buffer.put(MagicMock(audio=b"audio_chunk_0"))
            output_buffer.done = True
            return output_buffer

        mock_tts_client.__call__ = MagicMock(side_effect=_mock_tts)

        def _mock_tts_call(request_iterator, output_buffer, **_):
            output_buffer.put(MagicMock(audio=b"audio_chunk_0"))
            output_buffer.done = True
            return output_buffer

        mock_tts_client.side_effect = _mock_tts_call

        service = S2SRIVAStreamingService(
            ast_server=ast_server,
            tts_server=tts_server,
            default_voice_name="Magpie-Multilingual.ES-US.Isabela",
        )

        context = DummyContext()

        # Create request iterator with config and audio
        def request_iterator():
            config = SpeechToSpeechConfig(
                source_language="en-US",
                target_language="es-US",
                voice_name="Magpie-Multilingual.ES-US.Isabela",
            )
            yield SpeechToSpeechRequest(config=config)
            for i in range(3):
                yield SpeechToSpeechRequest(audio_data=f"audio_data_{i}".encode())

        # Mock the _s2s_impl method to return responses
        with patch.object(service, "_s2s_impl") as mock_impl:
            mock_impl.return_value = iter(
                [
                    SpeechToSpeechResponse(audio_data=b"audio_chunk_0", audio_sample_rate=16000),
                    SpeechToSpeechResponse(audio_data=b"audio_chunk_1", audio_sample_rate=16000),
                    SpeechToSpeechResponse(audio_data=b"audio_chunk_2", audio_sample_rate=16000),
                ]
            )

            # Run inference
            responses = list(service.infer(request_iterator(), context, "test-request"))

            # Verify responses
            self.assertGreater(len(responses), 0)

            # Verify audio responses
            audio_responses = [r for r in responses if r.audio_data]
            self.assertEqual(len(audio_responses), 3)
            self.assertEqual(audio_responses[0].audio_data, b"audio_chunk_0")

            # Verify _s2s_impl was called
            mock_impl.assert_called_once()

    def test_streaming_with_segmentizer(
        self, mock_healthy, mock_call, mock_ast_client_class, mock_tts_client_class
    ):
        """Test streaming flow with segmentizer processing."""
        from s2s_service.riva_utils.s2s import S2SRIVAStreamingService
        from s2s_service.segmentizer import sentence_segmentizer

        # Create service with sentence segmentizer
        ast_server = RivaASRServer("localhost", 50051)
        tts_server = RivaTTSServer("localhost", 50052)

        # Mock ASR and TTS clients to pass health checks
        mock_ast_client = MagicMock()
        mock_ast_client_class.return_value = mock_ast_client
        mock_ast_client.is_healthy.return_value = True

        def _mock_ast(request_iterator, output_buffer, **_):
            output_buffer.put("auto-text")
            output_buffer.done = True
            return output_buffer

        mock_ast_client.__call__ = MagicMock(side_effect=_mock_ast)

        mock_tts_client = MagicMock()
        mock_tts_client_class.return_value = mock_tts_client
        mock_tts_client.is_healthy.return_value = True
        mock_tts_client.sample_rate_hz = 16000

        def _mock_tts(request_iterator, output_buffer, **_):
            output_buffer.put(MagicMock(audio=b"audio_chunk_0"))
            output_buffer.done = True
            return output_buffer

        mock_tts_client.side_effect = _mock_tts

        service = S2SRIVAStreamingService(
            ast_server=ast_server,
            tts_server=tts_server,
            segmentizer=sentence_segmentizer,
        )

        context = DummyContext()

        # Create request iterator
        def request_iterator():
            config = SpeechToSpeechConfig(source_language="en-US", target_language="es-US")
            yield SpeechToSpeechRequest(config=config)
            yield SpeechToSpeechRequest(audio_data=b"audio_data")

        # Mock the _s2s_impl method to return responses
        with patch.object(service, "_s2s_impl") as mock_impl:
            mock_impl.return_value = iter(
                [
                    SpeechToSpeechResponse(audio_data=b"segmented_audio", audio_sample_rate=16000),
                ]
            )

            # Run inference
            responses = list(service.infer(request_iterator(), context, "test-request"))

            # Verify responses
            self.assertGreater(len(responses), 0)
            self.assertEqual(responses[0].audio_data, b"segmented_audio")

            # Verify _s2s_impl was called
            mock_impl.assert_called_once()


@patch("s2s_service.riva_utils.s2s.GRPCRIVATTSClient")
@patch("s2s_service.riva_utils.s2s.GRPCRIVATransactionalASTClient")
@patch("common.servers.GRPCServer.__call__", return_value="localhost:50051")
@patch("common.servers.GRPCServer.is_healthy", return_value=True)
class TestS2STransactionalIntegration(unittest.TestCase):
    """Integration tests for RIVA Transactional S2S service."""

    @patch("s2s_service.riva_utils.s2s.S2SRIVATransactionalService.download_input_audio")
    @patch(
        "s2s_service.riva_utils.s2s.S2SRIVATransactionalService.extract_zeroshot_reference_audio"
    )
    def test_full_transactional_flow_with_zero_shot(
        self,
        mock_extract_ref,
        mock_receive_audio,
        mock_healthy,
        mock_call,
        mock_ast_client_class,
        mock_tts_client_class,
    ):
        """Test full transactional S2S flow with zero-shot TTS."""
        from s2s_service.riva_utils.s2s import S2SRIVATransactionalService

        # Create service
        ast_server = RivaASRServer("localhost", 50051)
        tts_server = RivaTTSServer("localhost", 50052)

        # Mock ASR and TTS clients to pass health checks
        mock_ast_client = MagicMock()
        mock_ast_client_class.return_value = mock_ast_client
        mock_ast_client.is_healthy.return_value = True

        def _mock_ast(request_iterator, output_buffer, **_):
            output_buffer.put("auto-text")
            output_buffer.done = True
            return output_buffer

        mock_ast_client.__call__ = MagicMock(side_effect=_mock_ast)

        mock_tts_client = MagicMock()
        mock_tts_client_class.return_value = mock_tts_client
        mock_tts_client.is_healthy.return_value = True
        mock_tts_client.sample_rate_hz = 16000

        def _mock_tts(request_iterator, output_buffer, **_):
            output_buffer.put(MagicMock(audio=b"audio_chunk_0"))
            output_buffer.done = True
            return output_buffer

        mock_tts_client.__call__ = MagicMock(side_effect=_mock_tts)

        service = S2SRIVATransactionalService(
            ast_server=ast_server,
            tts_server=tts_server,
        )

        context = DummyContext()

        # Create request iterator
        def request_iterator():
            config = SpeechToSpeechConfig(source_language="es-ES", target_language="en-US")
            yield SpeechToSpeechRequest(config=config)
            yield SpeechToSpeechRequest(audio_data=b"audio_data")

        # Mock the _s2s_impl method to return responses
        with patch.object(service, "_s2s_impl") as mock_impl:
            mock_impl.return_value = iter(
                [
                    SpeechToSpeechResponse(audio_data=b"tts_audio_0", audio_sample_rate=16000),
                    SpeechToSpeechResponse(audio_data=b"tts_audio_1", audio_sample_rate=16000),
                ]
            )

            # Run inference
            responses = list(service.infer(request_iterator(), context, "test-request"))

            # Verify responses
            self.assertGreater(len(responses), 0)

            # Verify audio responses
            audio_responses = [r for r in responses if r.audio_data]
            self.assertEqual(len(audio_responses), 2)
            self.assertEqual(audio_responses[0].audio_data, b"tts_audio_0")

            # Verify _s2s_impl was called
            mock_impl.assert_called_once()

    @patch("s2s_service.riva_utils.s2s.S2SRIVATransactionalService.download_input_audio")
    def test_transactional_without_zero_shot(
        self,
        mock_receive_audio,
        mock_healthy,
        mock_call,
        mock_ast_client_class,
        mock_tts_client_class,
    ):
        """Test transactional flow without zero-shot (using voice name fallback)."""
        from s2s_service.riva_utils.s2s import S2SRIVATransactionalService

        # Create service
        ast_server = RivaASRServer("localhost", 50051)
        tts_server = RivaTTSServer("localhost", 50052)

        # Create test audio file
        test_wav = create_test_wav_file()
        mock_receive_audio.return_value = test_wav

        # Mock ASR and TTS clients to pass health checks
        mock_ast_client = MagicMock()
        mock_ast_client_class.return_value = mock_ast_client
        mock_ast_client.is_healthy.return_value = True

        mock_tts_client = MagicMock()
        mock_tts_client_class.return_value = mock_tts_client
        mock_tts_client.is_healthy.return_value = True
        mock_tts_client.sample_rate_hz = 16000

        def _mock_tts(request_iterator, output_buffer, **_):
            output_buffer.put(MagicMock(audio=b"audio_chunk_0"))
            output_buffer.done = True
            return output_buffer

        mock_tts_client.__call__ = MagicMock(side_effect=_mock_tts)

        # Disable zero-shot for this test
        with patch.object(S2SRIVATransactionalService, "use_auto_zero_shot", False):
            service = S2SRIVATransactionalService(
                ast_server=ast_server,
                tts_server=tts_server,
                default_voice_name="Magpie-ZeroShot.Female-1",
            )

            context = DummyContext()

            # Create request iterator
            def request_iterator():
                config = SpeechToSpeechConfig(
                    source_language="en-US",
                    target_language="en-US",
                    voice_name="Magpie-ZeroShot.Male-1",
                )
                yield SpeechToSpeechRequest(config=config)
                yield SpeechToSpeechRequest(audio_data=b"audio_data")

            # Mock the _s2s_impl method to return responses
            with patch.object(service, "_s2s_impl") as mock_impl:
                mock_impl.return_value = iter(
                    [
                        SpeechToSpeechResponse(audio_data=b"tts_output", audio_sample_rate=16000),
                    ]
                )

                # Run inference
                responses = list(service.infer(request_iterator(), context, "test-request"))

                # Verify responses
                self.assertGreater(len(responses), 0)
                self.assertEqual(responses[0].audio_data, b"tts_output")

                # Verify _s2s_impl was called
                mock_impl.assert_called_once()


class TestS2SServiceComparison(unittest.TestCase):
    """Tests comparing different S2S service modes."""

    @patch("s2s_service.riva_utils.s2s.GRPCRIVATTSClient")
    @patch("s2s_service.riva_utils.s2s.GRPCRIVAStreamingASTClient")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    def test_streaming_vs_transactional_language_support(
        self, mock_healthy, mock_ast_streaming, mock_tts
    ):
        """Compare language support between streaming and transactional services."""
        from s2s_service.riva_utils.s2s import S2SRIVAStreamingService
        from s2s_service.riva_utils.s2s import S2SRIVATransactionalService

        ast_server = RivaASRServer("localhost", 50051)
        tts_server = RivaTTSServer("localhost", 50052)

        # Create streaming service
        streaming_service = S2SRIVAStreamingService(ast_server=ast_server, tts_server=tts_server)

        # Create transactional service
        with patch("s2s_service.riva_utils.s2s.GRPCRIVATransactionalASTClient"):
            transactional_service = S2SRIVATransactionalService(
                ast_server=ast_server, tts_server=tts_server
            )

        # Compare source language support
        # Transactional supports more source languages
        self.assertTrue(transactional_service.validate_source_language("de-DE"))
        self.assertFalse(streaming_service.validate_source_language("de-DE"))

        # Both streaming (Magpie Multilingual) and transactional (Magpie ZeroShot)
        # only support en-US as target language.
        self.assertFalse(streaming_service.validate_target_language("fr-FR"))
        self.assertFalse(transactional_service.validate_target_language("fr-FR"))
        self.assertTrue(streaming_service.validate_target_language("en-US"))
        self.assertTrue(transactional_service.validate_target_language("en-US"))

    @patch("s2s_service.riva_utils.s2s.GRPCRIVATTSClient")
    @patch("s2s_service.riva_utils.s2s.GRPCRIVAStreamingASTClient")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    def test_voice_configuration_modes(self, mock_healthy, mock_ast, mock_tts):
        """Test different voice configuration modes."""
        from s2s_service.riva_utils.s2s import S2SRIVAStreamingService
        from s2s_service.riva_utils.s2s import S2SRIVATransactionalService

        ast_server = RivaASRServer("localhost", 50051)
        tts_server = RivaTTSServer("localhost", 50052)

        # Streaming service - uses explicit voice names
        streaming_service = S2SRIVAStreamingService(ast_server=ast_server, tts_server=tts_server)

        self.assertFalse(streaming_service.use_auto_zero_shot)
        self.assertTrue(
            streaming_service.validate_voice_name("Magpie-Multilingual.EN-US.Sofia", "en-US")
        )

        # Transactional service - uses zero-shot
        with patch("s2s_service.riva_utils.s2s.GRPCRIVATransactionalASTClient"):
            transactional_service = S2SRIVATransactionalService(
                ast_server=ast_server, tts_server=tts_server
            )

        self.assertTrue(transactional_service.use_auto_zero_shot)
        self.assertTrue(
            transactional_service.validate_voice_name("Magpie-ZeroShot.Female-1", "en-US")
        )


if __name__ == "__main__":
    unittest.main()
