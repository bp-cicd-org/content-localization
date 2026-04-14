# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for RIVA S2S services (Transactional and Streaming)."""

import tempfile
import unittest
import wave
from pathlib import Path
from typing import NoReturn
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechConfig
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechRequest

from s2s_service.riva_utils.s2s import S2SRIVAStreamingService
from s2s_service.riva_utils.s2s import S2SRIVATransactionalService
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


class TestS2SRIVAStreamingService(unittest.TestCase):
    """Test cases for RIVA Streaming S2S service."""

    @patch("s2s_service.riva_utils.s2s.GRPCRIVATTSClient")
    @patch("s2s_service.riva_utils.s2s.GRPCRIVAStreamingASTClient")
    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50051")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    def test_initialization(self, mock_healthy, mock_call, mock_ast_client, mock_tts_client):
        """Test S2SRIVAStreamingService initialization."""
        ast_server = RivaASRServer("localhost", 50051)
        tts_server = RivaTTSServer("localhost", 50052)

        service = S2SRIVAStreamingService(
            ast_server=ast_server,
            tts_server=tts_server,
            sample_rate_hz=16000,
            default_source_language="en-US",
            default_target_language="es-US",
            default_voice_name="Magpie-Multilingual.ES-US.Isabela",
        )

        # Verify initialization
        self.assertEqual(service.sample_rate_hz, 16000)
        self.assertEqual(service.default_source_language, "en-US")
        self.assertEqual(service.default_target_language, "es-US")
        self.assertEqual(service.default_voice_name, "Magpie-Multilingual.ES-US.Isabela")
        self.assertFalse(service.use_auto_zero_shot)

    @patch("s2s_service.riva_utils.s2s.GRPCRIVATTSClient")
    @patch("s2s_service.riva_utils.s2s.GRPCRIVAStreamingASTClient")
    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50051")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    def test_supported_languages(self, mock_healthy, mock_call, mock_ast_client, mock_tts_client):
        """Test supported language validation."""
        ast_server = RivaASRServer("localhost", 50051)
        tts_server = RivaTTSServer("localhost", 50052)

        service = S2SRIVAStreamingService(ast_server=ast_server, tts_server=tts_server)

        # Test valid source languages
        self.assertTrue(service.validate_source_language("en-US"))
        self.assertTrue(service.validate_source_language("es-US"))
        self.assertTrue(service.validate_source_language("fr-FR"))

        # Test invalid source language
        self.assertFalse(service.validate_source_language("de-DE"))

        # Test valid target languages
        self.assertTrue(service.validate_target_language("en-US"))
        self.assertTrue(service.validate_target_language("es-US"))
        self.assertTrue(service.validate_target_language("fr-FR"))

        # Test invalid target language
        self.assertFalse(service.validate_target_language("ja-JP"))

    @patch("s2s_service.riva_utils.s2s.GRPCRIVATTSClient")
    @patch("s2s_service.riva_utils.s2s.GRPCRIVAStreamingASTClient")
    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50051")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    def test_voice_name_validation(self, mock_healthy, mock_call, mock_ast_client, mock_tts_client):
        """Test voice name validation."""
        ast_server = RivaASRServer("localhost", 50051)
        tts_server = RivaTTSServer("localhost", 50052)

        service = S2SRIVAStreamingService(ast_server=ast_server, tts_server=tts_server)

        # Test valid voice names for en-US
        self.assertTrue(service.validate_voice_name("Magpie-Multilingual.EN-US.Sofia", "en-US"))
        self.assertTrue(service.validate_voice_name("Magpie-Multilingual.EN-US.Ray", "en-US"))

        # Test valid voice names for es-US
        self.assertTrue(service.validate_voice_name("Magpie-Multilingual.ES-US.Isabela", "es-US"))
        self.assertTrue(service.validate_voice_name("Magpie-Multilingual.ES-US.Diego", "es-US"))

        # Test invalid voice name
        self.assertFalse(service.validate_voice_name("Invalid.Voice.Name", "en-US"))

        # Test None voice name (should be valid)
        self.assertTrue(service.validate_voice_name(None, "en-US"))

    @patch("s2s_service.riva_utils.s2s.GRPCRIVATTSClient")
    @patch("s2s_service.riva_utils.s2s.GRPCRIVAStreamingASTClient")
    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50051")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    def test_check_riva_health(self, mock_healthy, mock_call, mock_ast_client, mock_tts_client):
        """Test RIVA health check."""
        ast_server = RivaASRServer("localhost", 50051)
        tts_server = RivaTTSServer("localhost", 50052)

        # Mock the clients
        mock_ast_instance = MagicMock()
        mock_tts_instance = MagicMock()
        mock_ast_client.return_value = mock_ast_instance
        mock_tts_client.return_value = mock_tts_instance

        service = S2SRIVAStreamingService(ast_server=ast_server, tts_server=tts_server)

        # Test successful health check
        mock_ast_instance.is_healthy.return_value = True
        mock_tts_instance.is_healthy.return_value = True
        service._check_riva_health()  # Should not raise

        # Test failed health check
        mock_ast_instance.is_healthy.side_effect = ConnectionError("ASR unhealthy")
        with self.assertRaises(ConnectionError):
            service._check_riva_health()


class TestS2SRIVATransactionalService(unittest.TestCase):
    """Test cases for RIVA Transactional S2S service."""

    @patch("s2s_service.riva_utils.s2s.GRPCRIVATTSClient")
    @patch("s2s_service.riva_utils.s2s.GRPCRIVATransactionalASTClient")
    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50051")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    def test_initialization(self, mock_healthy, mock_call, mock_ast_client, mock_tts_client):
        """Test S2SRIVATransactionalService initialization."""
        ast_server = RivaASRServer("localhost", 50051)
        tts_server = RivaTTSServer("localhost", 50052)

        service = S2SRIVATransactionalService(
            ast_server=ast_server,
            tts_server=tts_server,
            sample_rate_hz=16000,
            default_source_language="en-US",
            default_target_language="en-US",
        )

        # Verify initialization
        self.assertEqual(service.sample_rate_hz, 16000)
        self.assertEqual(service.default_source_language, "en-US")
        self.assertEqual(service.default_target_language, "en-US")
        self.assertTrue(service.use_auto_zero_shot)

    @patch("s2s_service.riva_utils.s2s.GRPCRIVATTSClient")
    @patch("s2s_service.riva_utils.s2s.GRPCRIVATransactionalASTClient")
    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50051")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    def test_supported_languages(self, mock_healthy, mock_call, mock_ast_client, mock_tts_client):
        """Test supported language validation for transactional service."""
        ast_server = RivaASRServer("localhost", 50051)
        tts_server = RivaTTSServer("localhost", 50052)

        service = S2SRIVATransactionalService(ast_server=ast_server, tts_server=tts_server)

        # Test valid source languages (multiple supported)
        self.assertTrue(service.validate_source_language("en-US"))
        self.assertTrue(service.validate_source_language("es-ES"))
        self.assertTrue(service.validate_source_language("fr-FR"))
        self.assertTrue(service.validate_source_language("de-DE"))

        # Test invalid source language
        self.assertFalse(service.validate_source_language("zh-CN"))

        # Test target language (only en-US for transactional)
        self.assertTrue(service.validate_target_language("en-US"))
        self.assertFalse(service.validate_target_language("es-US"))

    @patch("s2s_service.riva_utils.s2s.GRPCRIVATTSClient")
    @patch("s2s_service.riva_utils.s2s.GRPCRIVATransactionalASTClient")
    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50051")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    def test_zero_shot_voice_names(self, mock_healthy, mock_call, mock_ast_client, mock_tts_client):
        """Test zero-shot voice name validation."""
        ast_server = RivaASRServer("localhost", 50051)
        tts_server = RivaTTSServer("localhost", 50052)

        service = S2SRIVATransactionalService(ast_server=ast_server, tts_server=tts_server)

        # Test valid zero-shot voice names
        self.assertTrue(service.validate_voice_name("Magpie-ZeroShot.Female-1", "en-US"))
        self.assertTrue(service.validate_voice_name("Magpie-ZeroShot.Male-Calm", "en-US"))
        self.assertTrue(service.validate_voice_name("Magpie-ZeroShot.Female-Happy", "en-US"))

        # Test invalid voice name
        self.assertFalse(service.validate_voice_name("Invalid.Voice", "en-US"))

    @patch("s2s_service.riva_utils.s2s.GRPCRIVATTSClient")
    @patch("s2s_service.riva_utils.s2s.GRPCRIVATransactionalASTClient")
    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50051")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    def test_extract_config_from_request(
        self, mock_healthy, mock_call, mock_ast_client, mock_tts_client
    ):
        """Test extracting configuration from request."""
        ast_server = RivaASRServer("localhost", 50051)
        tts_server = RivaTTSServer("localhost", 50052)

        service = S2SRIVATransactionalService(
            ast_server=ast_server,
            tts_server=tts_server,
            default_source_language="en-US",
            default_target_language="en-US",
            default_voice_name="Magpie-ZeroShot.Female-1",
        )

        # Test with complete config
        config = SpeechToSpeechConfig(
            source_language="es-ES",
            target_language="en-US",
            voice_name="Magpie-ZeroShot.Male-1",
        )
        request = SpeechToSpeechRequest(config=config)

        source, target, voice = service._extract_config_from_request(request)
        self.assertEqual(source, "es-ES")
        self.assertEqual(target, "en-US")
        self.assertEqual(voice, "Magpie-ZeroShot.Male-1")

        # Test with partial config
        config = SpeechToSpeechConfig(source_language="fr-FR")
        request = SpeechToSpeechRequest(config=config)

        source, target, voice = service._extract_config_from_request(request)
        self.assertEqual(source, "fr-FR")
        self.assertEqual(target, "en-US")  # default
        self.assertEqual(voice, "Magpie-ZeroShot.Female-1")  # default

        # Test with no config
        request = SpeechToSpeechRequest(audio_data=b"test")

        source, target, voice = service._extract_config_from_request(request)
        self.assertEqual(source, "en-US")  # default
        self.assertEqual(target, "en-US")  # default
        self.assertEqual(voice, "Magpie-ZeroShot.Female-1")  # default

    @patch("s2s_service.riva_utils.s2s.GRPCRIVATTSClient")
    @patch("s2s_service.riva_utils.s2s.GRPCRIVATransactionalASTClient")
    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50051")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    def test_validate_request_config(
        self, mock_healthy, mock_call, mock_ast_client, mock_tts_client
    ):
        """Test request configuration validation."""
        ast_server = RivaASRServer("localhost", 50051)
        tts_server = RivaTTSServer("localhost", 50052)

        service = S2SRIVATransactionalService(ast_server=ast_server, tts_server=tts_server)

        context = DummyContext()

        # Test valid configuration
        service._validate_request_config("en-US", "en-US", "Magpie-ZeroShot.Female-1", context)

        # Test invalid source language
        with self.assertRaises(Exception):
            service._validate_request_config("invalid-lang", "en-US", None, context)
        self.assertTrue(context.aborted)

        # Reset context
        context = DummyContext()

        # Test invalid target language
        with self.assertRaises(Exception):
            service._validate_request_config("en-US", "invalid-lang", None, context)
        self.assertTrue(context.aborted)

        # Reset context
        context = DummyContext()

        # Test invalid voice name
        with self.assertRaises(Exception):
            service._validate_request_config("en-US", "en-US", "Invalid.Voice", context)
        self.assertTrue(context.aborted)

    @patch("s2s_service.riva_utils.s2s.GRPCRIVATTSClient")
    @patch("s2s_service.riva_utils.s2s.GRPCRIVATransactionalASTClient")
    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50051")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    def test_extract_zeroshot_reference_audio(
        self, mock_healthy, mock_call, mock_ast_client, mock_tts_client
    ):
        """Test extracting zero-shot reference audio."""
        ast_server = RivaASRServer("localhost", 50051)
        tts_server = RivaTTSServer("localhost", 50052)

        service = S2SRIVATransactionalService(ast_server=ast_server, tts_server=tts_server)

        # Create a test WAV file
        test_wav = create_test_wav_file(duration_seconds=15.0)

        try:
            # Extract reference audio (should be clipped to 10 seconds)
            ref_file, sample_rate = service.extract_zeroshot_reference_audio(
                test_wav, max_duration=10.0
            )

            # Verify the reference file exists
            self.assertTrue(ref_file.exists())

            # Verify the reference audio is approximately 10 seconds
            with wave.open(str(ref_file), "rb") as wav:
                duration = wav.getnframes() / wav.getframerate()
                self.assertAlmostEqual(duration, 10.0, places=1)
                self.assertEqual(wav.getframerate(), sample_rate)

            # Clean up
            ref_file.unlink()
        finally:
            # Clean up test WAV
            Path(test_wav).unlink()

    @patch("s2s_service.riva_utils.s2s.GRPCRIVATTSClient")
    @patch("s2s_service.riva_utils.s2s.GRPCRIVATransactionalASTClient")
    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50051")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    def test_determine_voice_config(
        self, mock_healthy, mock_call, mock_ast_client, mock_tts_client
    ):
        """Test voice configuration determination."""
        ast_server = RivaASRServer("localhost", 50051)
        tts_server = RivaTTSServer("localhost", 50052)

        service = S2SRIVATransactionalService(ast_server=ast_server, tts_server=tts_server)

        # Test with zero-shot data (should return None for pure zero-shot)
        zero_shot_data = {"audio_prompt_file": "/tmp/test.wav"}
        voice = service._determine_voice_config(zero_shot_data, "Magpie-ZeroShot.Female-1", "en-US")
        self.assertIsNone(voice)

        # Test without zero-shot data and with voice name
        voice = service._determine_voice_config(None, "Magpie-ZeroShot.Male-1", "en-US")
        self.assertEqual(voice, "Magpie-ZeroShot.Male-1")

        # Test without zero-shot data and without voice name (should use default)
        voice = service._determine_voice_config(None, None, "en-US")
        self.assertEqual(voice, "Magpie-ZeroShot.Female-1")  # First voice in list

    @patch("s2s_service.riva_utils.s2s.GRPCRIVATTSClient")
    @patch("s2s_service.riva_utils.s2s.GRPCRIVATransactionalASTClient")
    @patch("common.servers.GRPCServer.__call__", return_value="localhost:50051")
    @patch("common.servers.GRPCServer.is_healthy", return_value=True)
    def test_cleanup_temp_files(self, mock_healthy, mock_call, mock_ast_client, mock_tts_client):
        """Test temporary file cleanup."""
        ast_server = RivaASRServer("localhost", 50051)
        tts_server = RivaTTSServer("localhost", 50052)

        service = S2SRIVATransactionalService(ast_server=ast_server, tts_server=tts_server)

        # Create test files
        zero_shot_file = Path(create_test_wav_file())
        full_audio_file = create_test_wav_file()

        # Verify files exist
        self.assertTrue(zero_shot_file.exists())
        self.assertTrue(Path(full_audio_file).exists())

        # Clean up
        service._cleanup_temp_files(zero_shot_file, full_audio_file)

        # Verify files are deleted
        self.assertFalse(zero_shot_file.exists())
        self.assertFalse(Path(full_audio_file).exists())


if __name__ == "__main__":
    unittest.main()
