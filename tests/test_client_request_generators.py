# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for request generators module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from nvidia.ai4m.activespeakerdetection.v1.activespeakerdetection_pb2 import (
    DetectActiveSpeakerRequest,
)

from client.source_simulators.audio import simulated_audio_chunk_generator
from client.source_simulators.video import VideoSourceSimulator
from client.source_simulators.video import simulated_asd_video_chunk_generator


@pytest.mark.unit
class TestSimulatedAudioChunkGenerator:
    """Test cases for simulated_audio_chunk_generator function."""

    @patch("client.source_simulators.audio.AudioSourceSimulator")
    def test_generator_yields_requests(self, mock_simulator_class):
        """Test that generator yields SpeechToSpeechRequest objects."""
        # Mock the simulator
        mock_simulator = MagicMock()
        mock_simulator.frame_rate = 16000
        mock_simulator.n_channels = 1
        mock_simulator.read.return_value = [b"audio_chunk_1", b"audio_chunk_2"]
        mock_simulator_class.return_value = mock_simulator

        # Test the generator
        requests = list(simulated_audio_chunk_generator(mock_simulator, chunk_size_secs=0.128))

        # Should yield 2 requests
        assert len(requests) == 2

        # Check first request
        assert requests[0].audio_data == b"audio_chunk_1"
        assert requests[0].audio_sample_rate == 16000
        assert requests[0].audio_num_channels == 1
        assert requests[0].audio_format == "LINEAR_PCM"

        # Check second request
        assert requests[1].audio_data == b"audio_chunk_2"
        assert requests[1].audio_sample_rate == 16000
        assert requests[1].audio_num_channels == 1
        assert requests[1].audio_format == "LINEAR_PCM"

    @patch("client.source_simulators.audio.AudioSourceSimulator")
    def test_generator_with_different_chunk_size(self, mock_simulator_class):
        """Test generator with different chunk size."""
        mock_simulator = MagicMock()
        mock_simulator.frame_rate = 44100
        mock_simulator.n_channels = 2
        mock_simulator.read.return_value = [b"audio_chunk"]
        mock_simulator_class.return_value = mock_simulator

        requests = list(simulated_audio_chunk_generator(mock_simulator, chunk_size_secs=0.5))

        assert len(requests) == 1
        assert requests[0].audio_sample_rate == 44100
        assert requests[0].audio_num_channels == 2

    @patch("client.source_simulators.audio.AudioSourceSimulator")
    def test_generator_empty_simulator(self, mock_simulator_class):
        """Test generator with empty simulator."""
        mock_simulator = MagicMock()
        mock_simulator.frame_rate = 16000
        mock_simulator.n_channels = 1
        mock_simulator.read.return_value = []
        mock_simulator_class.return_value = mock_simulator

        requests = list(simulated_audio_chunk_generator(mock_simulator, chunk_size_secs=0.128))

        assert len(requests) == 0


@pytest.mark.unit
class TestSimulatedAsdVideoChunkGenerator:
    """Test cases for simulated_asd_video_chunk_generator function."""

    def test_generator_yields_video_requests(self):
        """Test that generator yields DetectActiveSpeakerRequest objects from file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            # Create test video data
            video_data = b"video_chunk_1" * 100
            tmp_file.write(video_data)
            tmp_file.flush()

            try:
                # Test with default chunk size
                simulator = VideoSourceSimulator(tmp_file.name)
                requests = list(simulated_asd_video_chunk_generator(simulator, chunk_size=64))

                # Should generate DetectActiveSpeakerRequest objects
                assert len(requests) > 0
                for req in requests:
                    assert isinstance(req, DetectActiveSpeakerRequest)
                    assert req.data.video_data != b""

            finally:
                Path(tmp_file.name).unlink()

    def test_generator_with_custom_chunk_size(self):
        """Test generator with custom chunk size."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            video_data = b"video_chunk_1" * 50
            tmp_file.write(video_data)
            tmp_file.flush()

            try:
                # Test with larger chunk size
                simulator = VideoSourceSimulator(tmp_file.name)
                requests = list(simulated_asd_video_chunk_generator(simulator, chunk_size=200))

                # Should generate DetectActiveSpeakerRequest objects
                assert len(requests) > 0
                for req in requests:
                    assert isinstance(req, DetectActiveSpeakerRequest)
                    assert req.data.video_data != b""

            finally:
                Path(tmp_file.name).unlink()

    def test_generator_empty_file(self):
        """Test generator with empty file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            try:
                simulator = VideoSourceSimulator(tmp_file.name)
                requests = list(simulated_asd_video_chunk_generator(simulator))
                assert len(requests) == 0
            finally:
                Path(tmp_file.name).unlink()

    def test_generator_file_not_found(self):
        """Test generator with non-existent file."""
        with pytest.raises(FileNotFoundError):
            VideoSourceSimulator("nonexistent_file.mp4")


if __name__ == "__main__":
    pytest.main([__file__])
