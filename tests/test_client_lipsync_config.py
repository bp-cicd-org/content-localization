# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from argparse import Namespace
from unittest.mock import MagicMock
from unittest.mock import patch

from client.lipsync.args import lipsync_config_from_args
from client.lipsync.config import LipSyncConfig
from client.lipsync.encoding import create_output_video_encoding


class TestLipSyncConfig(unittest.TestCase):
    def setUp(self):
        self.args = MagicMock()
        self.args.audio_input = "audio.wav"
        self.args.video_input = "video.mp4"
        self.args.speaker_info_input = "speaker_info.csv"
        self.args.output = "output.mp4"
        self.args.lipsync_extend_audio = "silence"
        self.args.lipsync_extend_video = "forward"
        self.args.lipsync_output_bitrate_mbps = 20
        self.args.lipsync_output_idr_interval = 8
        self.args.lipsync_lossless = False
        self.args.lipsync_custom_encoding_params = None

    def test_from_args(self):
        config = LipSyncConfig.from_args(self.args)
        self.assertEqual(config.audio_filepath, "audio.wav")
        self.assertEqual(config.video_filepath, "video.mp4")
        self.assertEqual(config.speaker_info_filepath, "speaker_info.csv")
        self.assertEqual(config.output_filepath, "output.mp4")
        self.assertEqual(config.extend_audio, "silence")
        self.assertEqual(config.extend_video, "forward")
        self.assertEqual(config.bitrate_mbps, 20)
        self.assertEqual(config.idr_interval, 8)
        self.assertFalse(config.lossless)
        self.assertIsNone(config.audio_codec)
        self.assertIsNone(config.is_speaker_info_provided)
        self.assertIsNone(config.custom_encoding_params)

    def test_str(self):
        config = LipSyncConfig.from_args(self.args)
        output = str(config)
        self.assertIn("LipSync Configuration", output)
        self.assertIn("audio.wav", output)
        self.assertIn("video.mp4", output)
        self.assertIn("output.mp4", output)

    def test_from_args_invalid_custom_encoding_json_raises_value_error(self):
        self.args.lipsync_custom_encoding_params = "{invalid-json}"
        with self.assertRaises(ValueError):
            LipSyncConfig.from_args(self.args)

    def test_lipsync_config_from_args_rejects_non_object_custom_encoding_json(self):
        parsed_args = Namespace(
            lipsync_input_audio_codec="MP3",
            lipsync_extend_audio="unspecified",
            lipsync_extend_video="unspecified",
            lipsync_output_bitrate_mbps=20,
            lipsync_output_idr_interval=8,
            lipsync_head_movement_speed=None,
            lipsync_output_audio_codec=None,
            lipsync_is_speaker_info_provided=False,
            lipsync_lossless=False,
            lipsync_custom_encoding_params="[1]",
            background_audio_input=None,
            lipsync_background_audio_codec=None,
            lipsync_background_audio_volume=1.0,
        )
        with self.assertRaises(ValueError):
            lipsync_config_from_args(parsed_args)

    def test_create_output_video_encoding_rejects_non_object_custom_encoding_json(self):
        self.args.lipsync_custom_encoding_params = "[1]"
        config = LipSyncConfig.from_args(self.args)
        with self.assertRaises(ValueError):
            create_output_video_encoding(config=config)

    @patch("client.lipsync.config.is_file_available")
    def test_validate_lipsync_config(self, mock_is_file_available):
        # Setup mocks
        mock_is_file_available.side_effect = lambda _path, _exts: True
        config = LipSyncConfig.from_args(self.args)
        # Should not raise
        self.assertTrue(config.validate_lipsync_config())

    @patch("client.lipsync.config.is_file_available")
    def test_validate_lipsync_config_invalid_video(self, mock_is_file_available):
        mock_is_file_available.side_effect = lambda path, _exts: path != "video.mp4"
        config = LipSyncConfig.from_args(self.args)
        with self.assertRaises(RuntimeError):
            config.validate_lipsync_config()

    @patch("client.lipsync.config.is_file_available")
    def test_validate_lipsync_config_invalid_audio(self, mock_is_file_available):
        mock_is_file_available.side_effect = lambda path, _exts: path != "audio.wav"
        config = LipSyncConfig.from_args(self.args)
        with self.assertRaises(RuntimeError):
            config.validate_lipsync_config()

    @patch("client.lipsync.config.is_file_available")
    def test_validate_lipsync_config_invalid_speaker_info(self, mock_is_file_available):
        mock_is_file_available.side_effect = lambda path, _exts: path != "speaker_info.csv"
        config = LipSyncConfig.from_args(self.args)
        with self.assertRaises(RuntimeError):
            config.validate_lipsync_config()


if __name__ == "__main__":
    unittest.main()
