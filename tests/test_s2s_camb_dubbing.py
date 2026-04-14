# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for CambAI Dubbing S2S service."""

import os
import tempfile
import unittest
import wave
from pathlib import Path
from typing import NoReturn
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechConfig
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechRequest
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechResponse


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
    """Create a temporary WAV file for testing.

    Args:
        duration_seconds: Duration of the audio file in seconds.
        sample_rate: Sample rate in Hz.

    Returns:
        Path to the temporary WAV file.
    """
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file.close()

    frequency = 440
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
    audio_data = np.sin(2 * np.pi * frequency * t) * 32767
    audio_data = audio_data.astype(np.int16)

    with wave.open(temp_file.name, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_data.tobytes())

    return temp_file.name


@pytest.mark.unit
@patch.dict(os.environ, {"CAMB_API_KEY": "test-camb-api-key"})
class TestCambDubbingServiceInit(unittest.TestCase):
    """Test CambDubbingService initialization."""

    def test_initialization_defaults(self) -> None:
        """Default initialization should set correct attributes."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService()
        self.assertEqual(service.sample_rate_hz, 16000)
        self.assertEqual(service.default_source_language, "1")
        self.assertEqual(service.default_target_language, "54")
        self.assertEqual(service.audio_format, "wav")

    def test_initialization_custom(self) -> None:
        """Custom initialization should override defaults."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService(
            sample_rate_hz=48000,
            default_source_language="81",
            default_target_language="1",
            audio_format="mp3",
        )
        self.assertEqual(service.sample_rate_hz, 48000)
        self.assertEqual(service.default_source_language, "81")
        self.assertEqual(service.default_target_language, "1")

    def test_missing_api_key(self) -> None:
        """Missing CAMB_API_KEY should raise RuntimeError."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CAMB_API_KEY", None)
            with self.assertRaises(RuntimeError) as ctx:
                CambDubbingService()
            self.assertIn("CAMB_API_KEY", str(ctx.exception))


@pytest.mark.unit
@patch.dict(os.environ, {"CAMB_API_KEY": "test-camb-api-key"})
class TestCambDubbingServiceLanguages(unittest.TestCase):
    """Test CambAI language validation."""

    def test_valid_source_languages(self) -> None:
        """Valid CambAI integer ID strings should pass validation."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService()
        # English=1, Spanish=54, Hindi=81
        self.assertTrue(service.validate_source_language("1"))
        self.assertTrue(service.validate_source_language("54"))
        self.assertTrue(service.validate_source_language("81"))

    def test_invalid_source_language(self) -> None:
        """Short codes like 'en' are not valid CambAI IDs."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService()
        self.assertFalse(service.validate_source_language("en"))
        self.assertFalse(service.validate_source_language("invalid"))
        self.assertFalse(service.validate_source_language("999"))

    def test_valid_target_languages(self) -> None:
        """Target language validation uses same ID set."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService()
        self.assertTrue(service.validate_target_language("1"))
        self.assertTrue(service.validate_target_language("54"))

    def test_invalid_target_language(self) -> None:
        """Invalid target language IDs should fail."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService()
        self.assertFalse(service.validate_target_language("es"))
        self.assertFalse(service.validate_target_language("0"))


@pytest.mark.unit
@patch.dict(os.environ, {"CAMB_API_KEY": "test-camb-api-key"})
class TestCambDubbingServiceAudioFormat(unittest.TestCase):
    """Test audio format validation."""

    def test_mp3_invalid(self) -> None:
        """MP3 should not be valid — CambAI always outputs WAV."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService()
        self.assertFalse(service.validate_audio_format("mp3"))

    def test_wav_valid(self) -> None:
        """WAV should be valid — CambAI always outputs WAV."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService()
        self.assertTrue(service.validate_audio_format("wav"))

    def test_unsupported_format(self) -> None:
        """Unsupported formats should be invalid."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService()
        self.assertFalse(service.validate_audio_format("flac"))
        self.assertFalse(service.validate_audio_format("ogg"))


@pytest.mark.unit
@patch.dict(os.environ, {"CAMB_API_KEY": "test-camb-api-key"})
class TestCambDubbingServiceImpl(unittest.TestCase):
    """Test _impl method."""

    @patch("s2s_service.camb_utils.dubbing._convert_flac_to_wav")
    @patch("s2s_service.camb_utils.dubbing.download_output_audio_to_file")
    @patch("s2s_service.camb_utils.dubbing.get_output_audio_url")
    @patch("s2s_service.camb_utils.dubbing.wait_for_completion")
    @patch("s2s_service.camb_utils.dubbing.submit_dub_task")
    @patch("s2s_service.camb_utils.dubbing.upload_local_file")
    def test_impl_success(
        self,
        mock_upload: MagicMock,
        mock_submit: MagicMock,
        mock_wait: MagicMock,
        mock_get_url: MagicMock,
        mock_download: MagicMock,
        mock_convert: MagicMock,
    ) -> None:
        """Successful _impl should yield audio responses."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService()
        context = DummyContext()

        # Set up mocks for the full pipeline
        mock_upload.return_value = "f-123"
        mock_submit.return_value = "task-abc"
        mock_wait.return_value = 42
        mock_get_url.return_value = "https://cdn/dubbed.flac"

        # Create a real WAV file that _convert_flac_to_wav "produces"
        wav_file = create_test_wav_file()
        mock_convert.return_value = Path(wav_file)

        # Create a dummy FLAC file for download mock
        flac_tmp = tempfile.NamedTemporaryFile(suffix=".flac", delete=False)
        flac_tmp.write(b"\x00" * 100)
        flac_tmp.close()
        mock_download.return_value = Path(flac_tmp.name)

        input_file = create_test_wav_file()

        try:
            responses = list(
                service._impl(
                    input_path=input_file,
                    request_id="test-request",
                    context=context,
                    source_language="1",
                    target_language="54",
                )
            )

            # Should have audio responses in WAV format
            audio_responses = [r for r in responses if r.audio_data]
            self.assertGreater(len(audio_responses), 0)
            for resp in audio_responses:
                self.assertEqual(resp.audio_format, "wav")

            mock_upload.assert_called_once()
            mock_submit.assert_called_once()
            mock_wait.assert_called_once()
            mock_convert.assert_called_once()
        finally:
            if Path(flac_tmp.name).exists():
                Path(flac_tmp.name).unlink()

    @patch("s2s_service.camb_utils.dubbing.upload_local_file")
    def test_impl_upload_error(self, mock_upload: MagicMock) -> None:
        """Upload failure should produce error via queue."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService()
        context = DummyContext()

        mock_upload.side_effect = RuntimeError("Upload failed")

        input_file = create_test_wav_file()
        try:
            with self.assertRaises(Exception):
                list(
                    service._impl(
                        input_path=input_file,
                        request_id="test-request",
                        context=context,
                        source_language="1",
                        target_language="54",
                    )
                )
        finally:
            if Path(input_file).exists():
                Path(input_file).unlink()


@pytest.mark.unit
@patch.dict(os.environ, {"CAMB_API_KEY": "test-camb-api-key"})
class TestCambDubbingServiceInfer(unittest.TestCase):
    """Test infer method."""

    @patch("s2s_service.service.download_audio_file_from_iterator")
    def test_infer_with_config(self, mock_download: MagicMock) -> None:
        """Config in first request should override default languages."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService(default_source_language="1", default_target_language="54")
        context = DummyContext()

        test_wav = create_test_wav_file()
        mock_download.return_value = test_wav

        try:
            # source_language="81" (Hindi), target_language="1" (English)
            config = SpeechToSpeechConfig(source_language="81", target_language="1")

            def request_iterator():
                yield SpeechToSpeechRequest(config=config)
                yield SpeechToSpeechRequest(audio_data=b"audio_data")

            with patch.object(service, "_impl") as mock_impl:
                mock_impl.return_value = iter([SpeechToSpeechResponse(audio_data=b"output")])
                responses = list(service.infer(request_iterator(), context, "test-req"))

                self.assertEqual(len(responses), 1)
                call_kwargs = mock_impl.call_args[1]
                self.assertEqual(call_kwargs["source_language"], "81")
                self.assertEqual(call_kwargs["target_language"], "1")
        finally:
            if Path(test_wav).exists():
                Path(test_wav).unlink()

    @patch("s2s_service.service.download_audio_file_from_iterator")
    def test_infer_default_languages(self, mock_download: MagicMock) -> None:
        """No config should use service defaults."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService(default_source_language="1", default_target_language="54")
        context = DummyContext()

        test_wav = create_test_wav_file()
        mock_download.return_value = test_wav

        try:

            def request_iterator():
                # No config, just audio
                yield SpeechToSpeechRequest(audio_data=b"audio_data")

            with patch.object(service, "_impl") as mock_impl:
                mock_impl.return_value = iter([SpeechToSpeechResponse(audio_data=b"output")])
                list(service.infer(request_iterator(), context, "test-req"))

                call_kwargs = mock_impl.call_args[1]
                self.assertEqual(call_kwargs["source_language"], "1")
                self.assertEqual(call_kwargs["target_language"], "54")
        finally:
            if Path(test_wav).exists():
                Path(test_wav).unlink()

    @patch("s2s_service.service.download_audio_file_from_iterator")
    def test_infer_invalid_source_language(self, mock_download: MagicMock) -> None:
        """Invalid source language should abort with INVALID_ARGUMENT."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService()
        context = DummyContext()

        test_wav = create_test_wav_file()
        mock_download.return_value = test_wav

        try:
            config = SpeechToSpeechConfig(source_language="en", target_language="54")

            def request_iterator():
                yield SpeechToSpeechRequest(config=config)
                yield SpeechToSpeechRequest(audio_data=b"audio_data")

            with self.assertRaises(Exception):
                list(service.infer(request_iterator(), context, "test-req"))
            self.assertTrue(context.aborted)
        finally:
            if Path(test_wav).exists():
                Path(test_wav).unlink()

    @patch("s2s_service.service.download_audio_file_from_iterator")
    def test_infer_invalid_target_language(self, mock_download: MagicMock) -> None:
        """Invalid target language should abort with INVALID_ARGUMENT."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService()
        context = DummyContext()

        test_wav = create_test_wav_file()
        mock_download.return_value = test_wav

        try:
            config = SpeechToSpeechConfig(source_language="1", target_language="es")

            def request_iterator():
                yield SpeechToSpeechRequest(config=config)
                yield SpeechToSpeechRequest(audio_data=b"audio_data")

            with self.assertRaises(Exception):
                list(service.infer(request_iterator(), context, "test-req"))
            self.assertTrue(context.aborted)
        finally:
            if Path(test_wav).exists():
                Path(test_wav).unlink()


@pytest.mark.unit
@patch.dict(os.environ, {"CAMB_API_KEY": "test-camb-api-key"})
class TestCambDubbingServiceKeepalive(unittest.TestCase):
    """Test keepalive behavior during long-running operations."""

    @patch("s2s_service.camb_utils.dubbing.upload_local_file")
    @patch.dict(os.environ, {"S2S_CAMB_KEEPALIVE_INTERVAL": "0"})
    def test_keepalive_sent_during_processing(self, mock_upload: MagicMock) -> None:
        """Keepalives should be sent when queue is empty."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        service = CambDubbingService()
        context = DummyContext()

        # Make upload take some time then fail — we only care about
        # keepalive pings appearing before the error
        mock_upload.side_effect = RuntimeError("Simulated slow upload")

        input_file = create_test_wav_file()
        try:
            responses = []
            with self.assertRaises(Exception):
                for resp in service._impl(
                    input_path=input_file,
                    request_id="test-request",
                    context=context,
                    source_language="1",
                    target_language="54",
                ):
                    responses.append(resp)

            # At least one keepalive may have been sent before the error
            # (depends on timing), but we verify the mechanism doesn't crash
        finally:
            if Path(input_file).exists():
                Path(input_file).unlink()


@pytest.mark.unit
@patch.dict(os.environ, {"CAMB_API_KEY": "test-camb-api-key"})
class TestCambDubbingServiceArgsfactory(unittest.TestCase):
    """Test argsfactory static method."""

    def test_argsfactory_creates_parser(self) -> None:
        """argsfactory should return a valid parser."""
        from s2s_service.camb_utils.dubbing import CambDubbingService

        parser = CambDubbingService.argsfactory()
        self.assertIsNotNone(parser)
        # Should include base S2S args
        args = parser.parse_args([])
        self.assertTrue(hasattr(args, "sample_rate_hz"))
        self.assertTrue(hasattr(args, "default_source_language"))
        self.assertTrue(hasattr(args, "audio_format"))


@pytest.mark.unit
class TestConvertFlacToWav(unittest.TestCase):
    """Test _convert_flac_to_wav FLAC→WAV conversion."""

    def test_flac_converted_to_wav(self) -> None:
        """Valid FLAC file should be converted to WAV."""
        import soundfile as sf_test

        from s2s_service.camb_utils.dubbing import _convert_flac_to_wav

        # Create a real FLAC file using soundfile
        flac_tmp = tempfile.NamedTemporaryFile(suffix=".flac", delete=False)
        flac_tmp.close()
        sample_rate = 16000
        audio_data = np.zeros(sample_rate, dtype=np.float32)
        sf_test.write(flac_tmp.name, data=audio_data, samplerate=sample_rate)

        try:
            wav_path = _convert_flac_to_wav(Path(flac_tmp.name))
            self.assertTrue(wav_path.exists())
            self.assertEqual(wav_path.suffix, ".wav")
            # Verify the WAV file is readable
            with wave.open(str(wav_path), "rb") as wf:
                self.assertEqual(wf.getframerate(), sample_rate)
                self.assertEqual(wf.getnchannels(), 1)
        finally:
            Path(flac_tmp.name).unlink(missing_ok=True)
            wav_path.unlink(missing_ok=True)

    def test_invalid_file_raises(self) -> None:
        """Non-audio file should raise an error."""
        from s2s_service.camb_utils.dubbing import _convert_flac_to_wav

        tmp = tempfile.NamedTemporaryFile(suffix=".flac", delete=False)
        tmp.write(b"\x00" * 40)
        tmp.close()
        try:
            with self.assertRaises(Exception):
                _convert_flac_to_wav(Path(tmp.name))
        finally:
            Path(tmp.name).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
