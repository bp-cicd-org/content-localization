# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the standalone proto conversion functions and adapter generators."""

import unittest

import pytest
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    ActiveSpeakerDetectionConfig,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    ActiveSpeakerDetectionData,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    ActiveSpeakerDetectionResult,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    DetectActiveSpeakerResponse,
)
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    SpeakerInfo as AsdSpeakerInfo,
)
from nvidia.ai4m.audio.v1.audio_pb2 import AUDIO_CODEC_MP3
from nvidia.ai4m.audio.v1.audio_pb2 import AUDIO_CODEC_WAV
from nvidia.ai4m.audio.v1.audio_pb2 import AudioConfig
from nvidia.ai4m.common.v1.common_pb2 import BoundingBox
from nvidia.ai4m.controller.v1.controller_pb2 import ContentLocalizationRequest
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncConfig
from nvidia.ai4m.lipsync.v1.lipsync_pb2 import LipsyncInputData
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechConfig
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechResponse

from controller_service.conversions import to_asd_audio_data
from controller_service.conversions import to_asd_video_data
from controller_service.conversions import to_lipsync_translated_audio
from controller_service.conversions import to_lipsync_video
from controller_service.conversions import to_s2s_request
from controller_service.stream_adapters import asd_request_generator
from controller_service.stream_adapters import asd_response_to_lipsync_speaker_info
from controller_service.stream_adapters import lipsync_request_generator
from controller_service.stream_adapters import s2s_audio_to_lipsync_audio
from controller_service.stream_adapters import translated_audio_to_lipsync_audio

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Proto conversion functions
# ---------------------------------------------------------------------------


class TestToS2sRequest(unittest.TestCase):
    """Tests for to_s2s_request()."""

    def test_audio_data_conversion(self) -> None:
        req = ContentLocalizationRequest(audio_data=b"\x00\x01")
        s2s = to_s2s_request(req)
        self.assertEqual(s2s.audio_data, b"\x00\x01")
        self.assertEqual(s2s.audio_sample_rate, 16000)

    def test_config_only(self) -> None:
        req = ContentLocalizationRequest(s2s_config=SpeechToSpeechConfig())
        s2s = to_s2s_request(req)
        self.assertTrue(s2s.HasField("config"))

    def test_no_audio_or_config_raises(self) -> None:
        req = ContentLocalizationRequest(video_file_data=b"\x00")
        with self.assertRaises(ValueError):
            to_s2s_request(req)


class TestToAsdVideoData(unittest.TestCase):
    """Tests for to_asd_video_data()."""

    def test_video_data_conversion(self) -> None:
        req = ContentLocalizationRequest(video_file_data=b"\x99")
        asd_data = to_asd_video_data(req)
        self.assertEqual(asd_data.video_data, b"\x99")

    def test_missing_video_raises(self) -> None:
        req = ContentLocalizationRequest(audio_data=b"\x00")
        with self.assertRaises(ValueError):
            to_asd_video_data(req)


class TestToAsdAudioData(unittest.TestCase):
    """Tests for to_asd_audio_data()."""

    def test_audio_data_conversion(self) -> None:
        req = ContentLocalizationRequest(audio_data=b"\xbb")
        asd_data = to_asd_audio_data(req)
        self.assertEqual(asd_data.audio_data, b"\xbb")

    def test_missing_audio_raises(self) -> None:
        req = ContentLocalizationRequest(video_file_data=b"\x00")
        with self.assertRaises(ValueError):
            to_asd_audio_data(req)


class TestToLipsyncVideo(unittest.TestCase):
    """Tests for to_lipsync_video()."""

    def test_video_data_conversion(self) -> None:
        req = ContentLocalizationRequest(video_file_data=b"\xaa")
        lipsync_input = to_lipsync_video(req)
        self.assertEqual(lipsync_input.video_file_data, b"\xaa")

    def test_missing_video_raises(self) -> None:
        req = ContentLocalizationRequest(audio_data=b"\x00")
        with self.assertRaises(ValueError):
            to_lipsync_video(req)


# ---------------------------------------------------------------------------
# asd_request_generator
# ---------------------------------------------------------------------------


class TestAsdRequestGenerator(unittest.TestCase):
    """Tests for ASD request merging generator."""

    def test_emits_provided_config_first(self) -> None:
        """Client-provided ASD config is yielded as the first message."""
        asd_config = ActiveSpeakerDetectionConfig(
            input_audio_config=AudioConfig(encoding=AUDIO_CODEC_WAV),
        )
        results = list(
            asd_request_generator(
                video_iter=iter([ActiveSpeakerDetectionData(video_data=b"v1")]),
                audio_iter=iter([ActiveSpeakerDetectionData(audio_data=b"a1")]),
                asd_config=asd_config,
            )
        )
        self.assertGreaterEqual(len(results), 3)
        self.assertTrue(results[0].HasField("config"))
        self.assertEqual(results[0].config.input_audio_config.encoding, AUDIO_CODEC_WAV)

    def test_mp3_config_passed_through(self) -> None:
        """MP3 codec in client config is forwarded as-is."""
        asd_config = ActiveSpeakerDetectionConfig(
            input_audio_config=AudioConfig(encoding=AUDIO_CODEC_MP3),
        )
        results = list(
            asd_request_generator(
                video_iter=iter([]),
                audio_iter=iter([]),
                asd_config=asd_config,
            )
        )
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].HasField("config"))
        self.assertEqual(results[0].config.input_audio_config.encoding, AUDIO_CODEC_MP3)


# ---------------------------------------------------------------------------
# s2s_audio_to_lipsync_audio
# ---------------------------------------------------------------------------


class TestS2sAudioToLipsyncAudio(unittest.TestCase):
    """Tests for the S2S audio adapter generator."""

    def _make_response(self, audio_data=b"", fmt="mp3", keepalive=False):
        if keepalive:
            resp = SpeechToSpeechResponse()
            resp.keepalive.SetInParent()
            return resp
        return SpeechToSpeechResponse(
            audio_data=audio_data,
            audio_format=fmt,
        )

    def test_mp3_passthrough(self) -> None:
        responses = [
            self._make_response(audio_data=b"chunk1", fmt="mp3"),
            self._make_response(audio_data=b"chunk2", fmt="mp3"),
        ]
        results = list(s2s_audio_to_lipsync_audio(iter(responses), audio_format="mp3"))
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].audio_file_data, b"chunk1")

    def test_keepalive_skipped(self) -> None:
        responses = [
            self._make_response(keepalive=True),
            self._make_response(audio_data=b"data", fmt="mp3"),
        ]
        results = list(s2s_audio_to_lipsync_audio(iter(responses), audio_format="mp3"))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].audio_file_data, b"data")

    def test_wav_header_emitted(self) -> None:
        responses = [
            SpeechToSpeechResponse(
                audio_data=b"pcm",
                audio_format="wav",
                audio_sample_rate=16000,
                audio_num_channels=1,
            ),
        ]
        results = list(s2s_audio_to_lipsync_audio(iter(responses), audio_format="wav"))
        # First result is the WAV header, second is the audio data
        self.assertEqual(len(results), 2)
        # WAV header should start with RIFF
        self.assertTrue(results[0].audio_file_data.startswith(b"RIFF"))
        self.assertEqual(results[1].audio_file_data, b"pcm")

    def test_format_mismatch_warns_and_continues(self) -> None:
        """Mismatch should log a warning and continue with detected format."""
        responses = [self._make_response(audio_data=b"x", fmt="wav")]
        results = list(s2s_audio_to_lipsync_audio(iter(responses), audio_format="mp3"))
        # WAV detected → WAV header emitted + the audio chunk
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].audio_file_data.startswith(b"RIFF"))

    def test_empty_stream(self) -> None:
        results = list(s2s_audio_to_lipsync_audio(iter([]), audio_format="mp3"))
        self.assertEqual(results, [])


# ---------------------------------------------------------------------------
# asd_response_to_lipsync_speaker_info
# ---------------------------------------------------------------------------


class TestAsdResponseToLipsyncSpeakerInfo(unittest.TestCase):
    """Tests for the ASD speaker info adapter generator."""

    def _make_asd_response(self, x=10, y=20, w=100, h=200, frame_id=0, is_speaking=True):
        return DetectActiveSpeakerResponse(
            active_speaker_detection_result=ActiveSpeakerDetectionResult(
                frame_id=frame_id,
                speaker_data=[
                    AsdSpeakerInfo(
                        speaker_bbox=BoundingBox(x=x, y=y, width=w, height=h),
                        face_id=1,
                        is_speaking=is_speaking,
                    )
                ],
            )
        )

    def test_normal_speaker_info(self) -> None:
        resp = self._make_asd_response(x=10, y=20, w=100, h=200, is_speaking=True)
        results = list(asd_response_to_lipsync_speaker_info(iter([resp])))
        self.assertEqual(len(results), 1)
        speaker_info = results[0].per_frame_speaker_infos[0].speaker_infos[0]
        self.assertEqual(speaker_info.speaker_bbox.x, 10)
        self.assertEqual(speaker_info.speaker_bbox.width, 100)
        self.assertTrue(speaker_info.is_speaking)

    def test_not_speaking(self) -> None:
        resp = self._make_asd_response(is_speaking=False)
        results = list(asd_response_to_lipsync_speaker_info(iter([resp])))
        self.assertEqual(len(results), 1)
        speaker_info = results[0].per_frame_speaker_infos[0].speaker_infos[0]
        self.assertFalse(speaker_info.is_speaking)

    def test_empty_speaker_data(self) -> None:
        """Response with no speakers yields empty speaker_infos."""
        resp = DetectActiveSpeakerResponse(
            active_speaker_detection_result=ActiveSpeakerDetectionResult(
                frame_id=0,
                speaker_data=[],
            )
        )
        results = list(asd_response_to_lipsync_speaker_info(iter([resp])))
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0].per_frame_speaker_infos[0].speaker_infos), 0)

    def test_frame_id_preserved(self) -> None:
        resp = self._make_asd_response(frame_id=42)
        results = list(asd_response_to_lipsync_speaker_info(iter([resp])))
        self.assertEqual(results[0].per_frame_speaker_infos[0].frame_id, 42)


# ---------------------------------------------------------------------------
# lipsync_request_generator
# ---------------------------------------------------------------------------


class TestLipsyncRequestGenerator(unittest.TestCase):
    """Tests for the LipSync request merging generator."""

    _DEFAULT_CONFIG = LipsyncConfig(input_audio_codec=AUDIO_CODEC_MP3)

    def test_config_emitted_first(self) -> None:
        """Client-provided config is yielded as the first message."""
        results = list(
            lipsync_request_generator(
                video_iter=iter([]),
                audio_iter=iter([]),
                speaker_info_iter=None,
                lipsync_config=self._DEFAULT_CONFIG,
            )
        )
        # At minimum, the config message should be yielded
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].HasField("config"))
        self.assertEqual(results[0].config.input_audio_codec, AUDIO_CODEC_MP3)

    def test_interleaving_video_audio(self) -> None:
        video = [LipsyncInputData(video_file_data=b"v1")]
        audio = [LipsyncInputData(audio_file_data=b"a1")]
        results = list(
            lipsync_request_generator(
                video_iter=iter(video),
                audio_iter=iter(audio),
                speaker_info_iter=None,
                lipsync_config=self._DEFAULT_CONFIG,
            )
        )
        # config + video + audio = 3
        self.assertEqual(len(results), 3)
        self.assertTrue(results[0].HasField("config"))
        self.assertEqual(results[1].input.video_file_data, b"v1")
        self.assertEqual(results[2].input.audio_file_data, b"a1")

    def test_interleaving_with_speaker_info(self) -> None:
        video = [LipsyncInputData(video_file_data=b"v")]
        audio = [LipsyncInputData(audio_file_data=b"a")]
        speaker_info = [LipsyncInputData(per_frame_speaker_infos=[])]
        results = list(
            lipsync_request_generator(
                video_iter=iter(video),
                audio_iter=iter(audio),
                speaker_info_iter=iter(speaker_info),
                lipsync_config=self._DEFAULT_CONFIG,
            )
        )
        # config + video + audio + speaker_info = 4
        self.assertEqual(len(results), 4)

    def test_uneven_streams_zip_longest(self) -> None:
        """Shorter streams are padded with None (no yield)."""
        video = [LipsyncInputData(video_file_data=b"v1"), LipsyncInputData(video_file_data=b"v2")]
        audio = [LipsyncInputData(audio_file_data=b"a1")]
        results = list(
            lipsync_request_generator(
                video_iter=iter(video),
                audio_iter=iter(audio),
                speaker_info_iter=None,
                lipsync_config=self._DEFAULT_CONFIG,
            )
        )
        # config + v1 + a1 + v2 = 4
        self.assertEqual(len(results), 4)


# ---------------------------------------------------------------------------
# to_lipsync_translated_audio
# ---------------------------------------------------------------------------


class TestToLipsyncTranslatedAudio(unittest.TestCase):
    """Tests for to_lipsync_translated_audio()."""

    def test_translated_audio_conversion(self) -> None:
        """Translated audio bytes are mapped to LipsyncInputData.audio_file_data."""
        req = ContentLocalizationRequest(translated_audio_data=b"\xaa\xbb")
        result = to_lipsync_translated_audio(req)
        self.assertEqual(result.audio_file_data, b"\xaa\xbb")

    def test_missing_translated_audio_raises(self) -> None:
        """Raises ValueError when translated_audio_data is absent."""
        req = ContentLocalizationRequest(audio_data=b"\x00")
        with self.assertRaises(ValueError):
            to_lipsync_translated_audio(req)

    def test_empty_bytes_converted(self) -> None:
        """Empty translated audio bytes are still forwarded (not rejected)."""
        req = ContentLocalizationRequest(translated_audio_data=b"")
        result = to_lipsync_translated_audio(req)
        self.assertEqual(result.audio_file_data, b"")


# ---------------------------------------------------------------------------
# translated_audio_to_lipsync_audio
# ---------------------------------------------------------------------------


class TestTranslatedAudioToLipsyncAudio(unittest.TestCase):
    """Tests for translated_audio_to_lipsync_audio() stream adapter."""

    def test_single_chunk(self) -> None:
        """A single translated audio request yields one LipsyncInputData."""
        reqs = [ContentLocalizationRequest(translated_audio_data=b"\x01\x02")]
        results = list(translated_audio_to_lipsync_audio(iter(reqs)))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].audio_file_data, b"\x01\x02")

    def test_multiple_chunks(self) -> None:
        """Multiple translated audio requests each yield one LipsyncInputData."""
        reqs = [
            ContentLocalizationRequest(translated_audio_data=b"c1"),
            ContentLocalizationRequest(translated_audio_data=b"c2"),
            ContentLocalizationRequest(translated_audio_data=b"c3"),
        ]
        results = list(translated_audio_to_lipsync_audio(iter(reqs)))
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].audio_file_data, b"c1")
        self.assertEqual(results[1].audio_file_data, b"c2")
        self.assertEqual(results[2].audio_file_data, b"c3")

    def test_empty_stream(self) -> None:
        """Empty iterator yields nothing."""
        results = list(translated_audio_to_lipsync_audio(iter([])))
        self.assertEqual(results, [])
