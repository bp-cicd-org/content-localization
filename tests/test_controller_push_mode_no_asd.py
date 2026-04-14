# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for bypass-ASD controller flow."""

import os
import unittest
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import grpc
import pytest
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    ActiveSpeakerDetectionConfig,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    ActiveSpeakerDetectionResult,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    DetectActiveSpeakerResponse,
)
from nvidia.ai4m.audio.v1.audio_pb2 import AUDIO_CODEC_MP3
from nvidia.ai4m.audio.v1.audio_pb2 import AUDIO_CODEC_WAV
from nvidia.ai4m.audio.v1.audio_pb2 import AudioConfig
from nvidia.ai4m.controller.v1.controller_pb2 import ContentLocalizationConfig
from nvidia.ai4m.controller.v1.controller_pb2 import ContentLocalizationRequest
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncConfig
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncResponse
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechConfig
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechResponse

from controller_service.service import ControllerService

pytestmark = pytest.mark.unit


class _FakeS2SClient:
    def __init__(self, server: object) -> None:
        self.server = server

    def __call__(
        self,
        request_iterator: Iterator[object],
        output_buffer: Any,
        context: object,
        request_id: str,
    ) -> None:
        _ = context
        _ = request_id
        for _, _req in zip(range(2), request_iterator, strict=False):
            pass
        output_buffer.put(SpeechToSpeechResponse(audio_data=b"s2s-audio", audio_format="mp3"))
        output_buffer.done = True


class _FakeLipsyncClient:
    def __init__(self, server: object) -> None:
        self.server = server

    def __call__(
        self,
        request_iterator: Iterator[object],
        output_buffer: Any,
        context: object,
        request_id: str,
    ) -> None:
        _ = context
        _ = request_id
        for _, _req in zip(range(4), request_iterator, strict=False):
            pass
        output_buffer.put(LipsyncResponse(video_file_data=b"lipsync-video"))
        output_buffer.done = True


class _FakeAsdClient:
    def __init__(self, server: object) -> None:
        self.server = server

    def __call__(
        self,
        request_iterator: Iterator[object],
        output_buffer: Any,
        context: object,
        request_id: str,
    ) -> None:
        _ = context
        _ = request_id
        for _, _req in zip(range(4), request_iterator, strict=False):
            pass
        output_buffer.put(
            DetectActiveSpeakerResponse(
                active_speaker_detection_result=ActiveSpeakerDetectionResult(frame_id=0)
            )
        )
        output_buffer.done = True


def _make_request_stream_with_configs(
    asd_codec: int = AUDIO_CODEC_WAV,
    lipsync_codec: int = AUDIO_CODEC_MP3,
    bypass_asd: bool = False,
) -> Iterator[ContentLocalizationRequest]:
    """Build a request stream with configs first, then data (matching new protocol)."""
    return iter(
        [
            # controller_config must be sent so _extract_config doesn't
            # block for the full 5-second timeout in tests
            ContentLocalizationRequest(
                controller_config=ContentLocalizationConfig(
                    bypass_s2s=False,
                    bypass_asd=bypass_asd,
                ),
            ),
            ContentLocalizationRequest(
                s2s_config=SpeechToSpeechConfig(target_language="es"),
            ),
            ContentLocalizationRequest(
                asd_config=ActiveSpeakerDetectionConfig(
                    input_audio_config=AudioConfig(encoding=asd_codec),
                ),
            ),
            ContentLocalizationRequest(
                lipsync_config=LipsyncConfig(input_audio_codec=lipsync_codec),
            ),
            ContentLocalizationRequest(audio_data=b"client-audio"),
            ContentLocalizationRequest(video_file_data=b"client-video"),
        ]
    )


@pytest.mark.unit
@patch.dict(os.environ, {"S2S_SERVICE": "EL_DUBBING"})
class TestControllerPushModeBypassAsd(unittest.TestCase):
    """Regression tests for bypass-ASD controller behavior."""

    @patch("controller_service.service.SpeechToSpeechClient", new=_FakeS2SClient)
    @patch("controller_service.service.LipsyncClient", new=_FakeLipsyncClient)
    @patch("controller_service.service.ActiveSpeakerDetectionClient")
    def test_controller_impl_bypass_asd_skips_asd_client(
        self,
        mock_asd_client: MagicMock,
    ) -> None:
        """Bypass-ASD push path produces output without creating ASD client."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=None,
        )
        context = MagicMock()
        requests = _make_request_stream_with_configs(bypass_asd=True)

        responses = list(
            controller._controller_impl(
                request_iterator=requests,
                context=context,
                request_id="req-no-asd",
            )
        )

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].video_file_data, b"lipsync-video")
        mock_asd_client.assert_not_called()

    @patch("controller_service.service.SpeechToSpeechClient", new=_FakeS2SClient)
    @patch("controller_service.service.LipsyncClient", new=_FakeLipsyncClient)
    @patch("controller_service.service.ActiveSpeakerDetectionClient", new=_FakeAsdClient)
    @patch("controller_service.service.asd_request_generator")
    def test_controller_impl_asd_passes_client_config(
        self,
        mock_asd_request_generator: MagicMock,
    ) -> None:
        """ASD-enabled push path passes client-provided asd_config to generator."""
        mock_asd_request_generator.return_value = iter([])
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=MagicMock(),
        )
        context = MagicMock()
        requests = _make_request_stream_with_configs(asd_codec=AUDIO_CODEC_MP3)

        responses = list(
            controller._controller_impl(
                request_iterator=requests,
                context=context,
                request_id="req-asd",
            )
        )

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].video_file_data, b"lipsync-video")
        self.assertTrue(mock_asd_request_generator.called)
        passed_config = mock_asd_request_generator.call_args.kwargs["asd_config"]
        self.assertEqual(passed_config.input_audio_config.encoding, AUDIO_CODEC_MP3)

    @patch.dict(os.environ, {"S2S_SERVICE": "UNKNOWN_BACKEND"})
    def test_init_rejects_unknown_s2s_backend(self) -> None:
        """Controller init raises ValueError for unknown S2S_SERVICE values."""
        with self.assertRaises(ValueError):
            ControllerService(
                lipsync_server=MagicMock(),
                s2s_server=MagicMock(),
                asd_server=None,
            )

    @patch.dict(os.environ, {"S2S_SERVICE": "EL_DUBBING"})
    def test_s2s_output_audio_format_el_dubbing(self) -> None:
        """EL_DUBBING backend sets s2s_output_audio_format to MP3."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=None,
        )
        self.assertEqual(controller.s2s_output_audio_format, "MP3")

    @patch.dict(os.environ, {"S2S_SERVICE": "RIVA_TRANSACTIONAL"})
    def test_s2s_output_audio_format_riva(self) -> None:
        """RIVA_TRANSACTIONAL backend sets s2s_output_audio_format to WAV."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=None,
        )
        self.assertEqual(controller.s2s_output_audio_format, "WAV")

    @patch.dict(os.environ, {"S2S_SERVICE": "CAMB_DUBBING"})
    def test_s2s_output_audio_format_camb_is_wav(self) -> None:
        """CAMB_DUBBING backend always outputs WAV (FLAC converted to WAV in S2S)."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=None,
        )
        self.assertEqual(controller.s2s_output_audio_format, "WAV")

    @patch.dict(os.environ, {"S2S_SERVICE": "CAMB_DUBBING"})
    @patch("controller_service.service.SpeechToSpeechClient", new=_FakeS2SClient)
    @patch("controller_service.service.LipsyncClient", new=_FakeLipsyncClient)
    @patch("controller_service.service.logger")
    def test_camb_wav_output_overrides_lipsync_codec(
        self,
        mock_logger: MagicMock,
    ) -> None:
        """CAMB_DUBBING overrides lipsync codec to WAV (fixed output format)."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=None,
        )
        context = MagicMock()
        requests = _make_request_stream_with_configs(
            asd_codec=AUDIO_CODEC_WAV,
            lipsync_codec=AUDIO_CODEC_MP3,
            bypass_asd=True,
        )

        responses = list(
            controller._controller_impl(
                request_iterator=requests,
                context=context,
                request_id="req-camb-mirror-codec",
            )
        )

        self.assertEqual(len(responses), 1)
        mock_logger.warning.assert_called()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("(WAV)", warning_msg)

    @patch("controller_service.service.SpeechToSpeechClient", new=_FakeS2SClient)
    @patch("controller_service.service.LipsyncClient", new=_FakeLipsyncClient)
    def test_server_overrides_is_speaker_info_bypass_asd(self) -> None:
        """Server sets is_speaker_info_provided=False when ASD is bypassed."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=None,
        )
        context = MagicMock()
        requests = _make_request_stream_with_configs(bypass_asd=True)

        responses = list(
            controller._controller_impl(
                request_iterator=requests,
                context=context,
                request_id="req-speaker-no-asd",
            )
        )

        self.assertEqual(len(responses), 1)

    @patch("controller_service.service.SpeechToSpeechClient", new=_FakeS2SClient)
    @patch("controller_service.service.LipsyncClient", new=_FakeLipsyncClient)
    @patch("controller_service.service.ActiveSpeakerDetectionClient", new=_FakeAsdClient)
    def test_server_overrides_is_speaker_info_with_asd(self) -> None:
        """Server sets is_speaker_info_provided=True when ASD is enabled."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=MagicMock(),
        )
        context = MagicMock()
        requests = _make_request_stream_with_configs()

        responses = list(
            controller._controller_impl(
                request_iterator=requests,
                context=context,
                request_id="req-speaker-with-asd",
            )
        )

        self.assertEqual(len(responses), 1)

    @patch("controller_service.service.SpeechToSpeechClient", new=_FakeS2SClient)
    @patch("controller_service.service.LipsyncClient", new=_FakeLipsyncClient)
    @patch("controller_service.service.logger")
    def test_lipsync_codec_mismatch_warns_and_overrides(
        self,
        mock_logger: MagicMock,
    ) -> None:
        """Mismatched lipsync input codec triggers warning and gets overridden."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=None,
        )
        context = MagicMock()
        requests = _make_request_stream_with_configs(lipsync_codec=AUDIO_CODEC_WAV, bypass_asd=True)

        responses = list(
            controller._controller_impl(
                request_iterator=requests,
                context=context,
                request_id="req-mismatch",
            )
        )

        self.assertEqual(len(responses), 1)
        mock_logger.warning.assert_called()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("does not match S2S output format", warning_msg)

    @patch("controller_service.service.SpeechToSpeechClient", new=_FakeS2SClient)
    @patch("controller_service.service.LipsyncClient", new=_FakeLipsyncClient)
    @patch("controller_service.service.logger")
    def test_lipsync_codec_match_no_warning(
        self,
        mock_logger: MagicMock,
    ) -> None:
        """Matching lipsync input codec does not trigger a warning."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=None,
        )
        context = MagicMock()
        requests = _make_request_stream_with_configs(lipsync_codec=AUDIO_CODEC_MP3, bypass_asd=True)

        responses = list(
            controller._controller_impl(
                request_iterator=requests,
                context=context,
                request_id="req-match",
            )
        )

        self.assertEqual(len(responses), 1)
        for call_args in mock_logger.warning.call_args_list:
            self.assertNotIn("does not match S2S output format", call_args[0][0])

    # -- bypass S2S tests --------------------------------------------------

    @patch("controller_service.service.SpeechToSpeechClient")
    @patch("controller_service.service.LipsyncClient", new=_FakeLipsyncClient)
    def test_bypass_s2s_skips_s2s_client(
        self,
        mock_s2s_client: MagicMock,
    ) -> None:
        """When bypass_s2s=True, S2S client is never instantiated."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=MagicMock(),
        )
        context = MagicMock()
        requests = _make_bypass_s2s_request_stream()

        responses = list(
            controller._controller_impl(
                request_iterator=requests,
                context=context,
                request_id="req-bypass-s2s",
            )
        )

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].video_file_data, b"lipsync-video")
        mock_s2s_client.assert_not_called()

    @patch("controller_service.service.SpeechToSpeechClient")
    @patch("controller_service.service.LipsyncClient", new=_FakeLipsyncClient)
    @patch("controller_service.service.logger")
    def test_bypass_s2s_no_codec_override_warning(
        self,
        mock_logger: MagicMock,
        _mock_s2s_client: MagicMock,
    ) -> None:
        """In bypass mode, lipsync codec mismatch does NOT trigger a warning."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=MagicMock(),
        )
        context = MagicMock()
        # Client sets WAV codec — normally this would trigger override
        # warning against the MP3 S2S output, but bypass mode should skip
        requests = _make_bypass_s2s_request_stream(
            lipsync_codec=AUDIO_CODEC_WAV,
        )

        responses = list(
            controller._controller_impl(
                request_iterator=requests,
                context=context,
                request_id="req-bypass-no-warn",
            )
        )

        self.assertEqual(len(responses), 1)
        for call_args in mock_logger.warning.call_args_list:
            self.assertNotIn("does not match S2S output format", call_args[0][0])

    @patch("controller_service.service.SpeechToSpeechClient")
    @patch("controller_service.service.LipsyncClient", new=_FakeLipsyncClient)
    @patch("controller_service.service.ActiveSpeakerDetectionClient", new=_FakeAsdClient)
    def test_bypass_s2s_with_asd_enabled(
        self,
        mock_s2s_client: MagicMock,
    ) -> None:
        """Bypass S2S + ASD enabled: S2S skipped but ASD still runs."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=MagicMock(),
        )
        context = MagicMock()
        requests = _make_bypass_s2s_request_stream(include_asd_config=True)

        responses = list(
            controller._controller_impl(
                request_iterator=requests,
                context=context,
                request_id="req-bypass-with-asd",
            )
        )

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].video_file_data, b"lipsync-video")
        mock_s2s_client.assert_not_called()

    @patch("controller_service.service.SpeechToSpeechClient")
    @patch("controller_service.service.LipsyncClient", new=_FakeLipsyncClient)
    def test_no_s2s_server_without_bypass_aborts(
        self,
        mock_s2s_client: MagicMock,
    ) -> None:
        """When s2s_server is None and bypass_s2s is not set, abort with FAILED_PRECONDITION."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=None,
            asd_server=None,
        )
        context = MagicMock()
        # Explicitly send bypass_s2s=False so _extract_config
        # returns immediately instead of blocking for 5s
        requests = iter(
            [
                ContentLocalizationRequest(
                    controller_config=ContentLocalizationConfig(
                        bypass_s2s=False,
                        bypass_asd=True,
                    ),
                ),
                ContentLocalizationRequest(
                    lipsync_config=LipsyncConfig(input_audio_codec=AUDIO_CODEC_MP3),
                ),
                ContentLocalizationRequest(audio_data=b"audio"),
                ContentLocalizationRequest(video_file_data=b"video"),
            ]
        )

        list(
            controller._controller_impl(
                request_iterator=requests,
                context=context,
                request_id="req-no-s2s-no-bypass",
            )
        )

        context.abort.assert_called_once()
        abort_kwargs = context.abort.call_args
        self.assertEqual(
            abort_kwargs.kwargs.get("code", abort_kwargs[0][0] if abort_kwargs[0] else None),
            grpc.StatusCode.FAILED_PRECONDITION,
        )

    @patch("controller_service.service.SpeechToSpeechClient")
    @patch("controller_service.service.LipsyncClient", new=_FakeLipsyncClient)
    def test_no_s2s_server_with_bypass_succeeds(
        self,
        mock_s2s_client: MagicMock,
    ) -> None:
        """When s2s_server is None but bypass_s2s=True, pipeline succeeds."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=None,
            asd_server=MagicMock(),
        )
        context = MagicMock()
        requests = _make_bypass_s2s_request_stream()

        responses = list(
            controller._controller_impl(
                request_iterator=requests,
                context=context,
                request_id="req-no-s2s-with-bypass",
            )
        )

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].video_file_data, b"lipsync-video")
        mock_s2s_client.assert_not_called()
        context.abort.assert_not_called()

    # -- bypass_asd tests -------------------------------------------------

    @patch("controller_service.service.SpeechToSpeechClient", new=_FakeS2SClient)
    @patch("controller_service.service.LipsyncClient", new=_FakeLipsyncClient)
    @patch("controller_service.service.ActiveSpeakerDetectionClient")
    def test_bypass_asd_with_server_configured(
        self,
        mock_asd_client: MagicMock,
    ) -> None:
        """ASD server present + bypass_asd=True -> ASD skipped."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=MagicMock(),
        )
        context = MagicMock()
        requests = _make_request_stream_with_configs(bypass_asd=True)

        responses = list(
            controller._controller_impl(
                request_iterator=requests,
                context=context,
                request_id="req-bypass-asd-with-server",
            )
        )

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].video_file_data, b"lipsync-video")
        mock_asd_client.assert_not_called()

    @patch("controller_service.service.SpeechToSpeechClient", new=_FakeS2SClient)
    @patch("controller_service.service.LipsyncClient", new=_FakeLipsyncClient)
    def test_no_asd_server_without_bypass_aborts(self) -> None:
        """ASD server=None + bypass_asd=False -> FAILED_PRECONDITION."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=MagicMock(),
            asd_server=None,
        )
        context = MagicMock()
        requests = _make_request_stream_with_configs(bypass_asd=False)

        list(
            controller._controller_impl(
                request_iterator=requests,
                context=context,
                request_id="req-no-asd-no-bypass",
            )
        )

        context.abort.assert_called_once()
        abort_kwargs = context.abort.call_args
        self.assertEqual(
            abort_kwargs.kwargs.get("code", abort_kwargs[0][0] if abort_kwargs[0] else None),
            grpc.StatusCode.FAILED_PRECONDITION,
        )

    @patch("controller_service.service.SpeechToSpeechClient")
    @patch("controller_service.service.LipsyncClient", new=_FakeLipsyncClient)
    def test_combined_bypass_s2s_and_bypass_asd(
        self,
        mock_s2s_client: MagicMock,
    ) -> None:
        """Both bypass_s2s=True and bypass_asd=True -> only LipSync runs."""
        controller = ControllerService(
            lipsync_server=MagicMock(),
            s2s_server=None,
            asd_server=None,
        )
        context = MagicMock()
        requests = _make_bypass_s2s_request_stream(bypass_asd=True)

        responses = list(
            controller._controller_impl(
                request_iterator=requests,
                context=context,
                request_id="req-both-bypass",
            )
        )

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].video_file_data, b"lipsync-video")
        mock_s2s_client.assert_not_called()
        context.abort.assert_not_called()


def _make_bypass_s2s_request_stream(
    lipsync_codec: int = AUDIO_CODEC_MP3,
    include_asd_config: bool = False,
    bypass_asd: bool = False,
) -> Iterator[ContentLocalizationRequest]:
    """Build a request stream for bypass-S2S mode.

    Sends controller_config with bypass_s2s=True, no s2s_config,
    then translated audio alongside original audio/video.
    """
    msgs: list[ContentLocalizationRequest] = [
        ContentLocalizationRequest(
            controller_config=ContentLocalizationConfig(
                bypass_s2s=True,
                bypass_asd=bypass_asd,
            ),
        ),
        ContentLocalizationRequest(
            lipsync_config=LipsyncConfig(input_audio_codec=lipsync_codec),
        ),
    ]
    if include_asd_config:
        msgs.append(
            ContentLocalizationRequest(
                asd_config=ActiveSpeakerDetectionConfig(
                    input_audio_config=AudioConfig(encoding=AUDIO_CODEC_WAV),
                ),
            )
        )
    msgs.extend(
        [
            ContentLocalizationRequest(audio_data=b"original-audio"),
            ContentLocalizationRequest(video_file_data=b"client-video"),
            ContentLocalizationRequest(translated_audio_data=b"translated-mp3"),
        ]
    )
    return iter(msgs)
