# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for latency analysis module."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from client.s2s.latency_analysis import calculate_per_chunk_latencies
from client.s2s.latency_analysis import plot_latency


class TestCalculateLatency:
    """Test cases for calculate_per_chunk_latencies function."""

    def test_calculate_latency_basic(self):
        """Test basic latency calculation."""
        # Create mock ledgers
        input_ledger = {
            0: 1000.0,  # chunk_id: timestamp
            1: 1001.0,
            2: 1002.0,
        }

        output_ledger = {
            0: 1000.5,  # chunk_id: timestamp
            1: 1001.5,
            2: 1002.5,
        }

        # Calculate latency
        latency_data = calculate_per_chunk_latencies(input_ledger, output_ledger)

        # Check results
        assert len(latency_data) == 3
        assert latency_data[0] == 0.5  # 1000.5 - 1000.0
        assert latency_data[1] == 0.5  # 1001.5 - 1001.0
        assert latency_data[2] == 0.5  # 1002.5 - 1002.0

    def test_calculate_latency_different_timings(self):
        """Test latency calculation with different timings."""
        input_ledger = {
            0: 1000.0,
            1: 1001.0,
            2: 1002.0,
        }

        output_ledger = {
            0: 1000.2,  # Faster processing
            1: 1001.8,  # Slower processing
            2: 1002.1,  # Normal processing
        }

        latency_data = calculate_per_chunk_latencies(input_ledger, output_ledger)

        assert len(latency_data) == 3
        assert latency_data[0] == pytest.approx(0.2)
        assert latency_data[1] == pytest.approx(0.8)
        assert latency_data[2] == pytest.approx(0.1)

    def test_calculate_latency_missing_chunks(self):
        """Test latency calculation with missing chunks."""
        input_ledger = {
            0: 1000.0,
            1: 1001.0,
            2: 1002.0,
        }

        output_ledger = {
            0: 1000.5,
            # Missing chunk 1
            2: 1002.5,
        }

        latency_data = calculate_per_chunk_latencies(input_ledger, output_ledger)

        # Should only include chunks that exist in both ledgers
        assert len(latency_data) == 2
        assert latency_data[0] == 0.5  # chunk 0
        assert latency_data[1] == 0.5  # chunk 2

    def test_calculate_latency_empty_ledgers(self):
        """Test latency calculation with empty ledgers."""
        input_ledger = {}
        output_ledger = {}

        latency_data = calculate_per_chunk_latencies(input_ledger, output_ledger)

        assert len(latency_data) == 0

    def test_calculate_latency_one_empty_ledger(self):
        """Test latency calculation with one empty ledger."""
        input_ledger = {0: 1000.0, 1: 1001.0}
        output_ledger = {}

        latency_data = calculate_per_chunk_latencies(input_ledger, output_ledger)

        assert len(latency_data) == 0

    def test_calculate_latency_negative_latency(self):
        """Test latency calculation with negative latency (sink before source)."""
        input_ledger = {
            0: 1000.0,
        }

        output_ledger = {
            0: 999.5,  # Sink timestamp before source timestamp
        }

        latency_data = calculate_per_chunk_latencies(input_ledger, output_ledger)

        assert len(latency_data) == 1
        assert latency_data[0] == -0.5  # Negative latency

    def test_calculate_latency_large_numbers(self):
        """Test latency calculation with large timestamp values."""
        input_ledger = {
            0: 1640995200.0,  # Unix timestamp
            1: 1640995201.0,
        }

        output_ledger = {
            0: 1640995200.5,
            1: 1640995201.5,
        }

        latency_data = calculate_per_chunk_latencies(input_ledger, output_ledger)

        assert len(latency_data) == 2
        assert latency_data[0] == 0.5
        assert latency_data[1] == 0.5


class TestPlotLatencyAnalysis:
    """Test cases for plot_latency function."""

    def test_plot_latency_basic(self):
        """Test basic latency plotting."""
        # Create sample latency data
        output_stream_latencies = [0.1, 0.2, 0.15, 0.3, 0.25]
        per_chunk_latencies = [0.05, 0.1, 0.08, 0.12, 0.09]
        chunk_size_secs = 0.128

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "latency_plot.png"

            # Test the function
            plot_latency(
                output_stream_latencies, per_chunk_latencies, chunk_size_secs, str(output_path)
            )

            # Check that the plot file was created
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_plot_latency_empty_data(self):
        """Test latency plotting with empty data."""
        output_stream_latencies = []
        per_chunk_latencies = []
        chunk_size_secs = 0.128

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "latency_plot.png"

            # Test the function
            plot_latency(
                output_stream_latencies, per_chunk_latencies, chunk_size_secs, str(output_path)
            )

            # Check that the plot file was created even with empty data
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_plot_latency_single_point(self):
        """Test latency plotting with single data point."""
        output_stream_latencies = [0.5]
        per_chunk_latencies = [0.2]
        chunk_size_secs = 0.128

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "latency_plot.png"

            # Test the function
            plot_latency(
                output_stream_latencies, per_chunk_latencies, chunk_size_secs, str(output_path)
            )

            # Check that the plot file was created
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_plot_latency_large_dataset(self):
        """Test latency plotting with large dataset."""
        # Create larger dataset
        np.random.seed(42)  # For reproducible results
        output_stream_latencies = np.random.normal(0.2, 0.05, 100).tolist()
        per_chunk_latencies = np.random.normal(0.1, 0.02, 100).tolist()
        chunk_size_secs = 0.128

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "latency_plot.png"

            # Test the function
            plot_latency(
                output_stream_latencies, per_chunk_latencies, chunk_size_secs, str(output_path)
            )

            # Check that the plot file was created
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_plot_latency_negative_values(self):
        """Test latency plotting with negative values."""
        output_stream_latencies = [-0.1, 0.2, -0.15, 0.3, -0.25]
        per_chunk_latencies = [-0.05, 0.1, -0.08, 0.12, -0.09]
        chunk_size_secs = 0.128

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "latency_plot.png"

            # Test the function
            plot_latency(
                output_stream_latencies, per_chunk_latencies, chunk_size_secs, str(output_path)
            )

            # Check that the plot file was created
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_plot_latency_extreme_values(self):
        """Test latency plotting with extreme values."""
        output_stream_latencies = [0.001, 10.0, 0.0001, 100.0]
        per_chunk_latencies = [0.0005, 5.0, 0.00005, 50.0]
        chunk_size_secs = 0.128

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "latency_plot.png"

            # Test the function
            plot_latency(
                output_stream_latencies, per_chunk_latencies, chunk_size_secs, str(output_path)
            )

            # Check that the plot file was created
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    @patch("client.s2s.latency_analysis.plt.savefig")
    def test_plot_latency_savefig_called(self, mock_savefig):
        """Test that plt.savefig is called correctly."""
        output_stream_latencies = [0.1, 0.2, 0.15]
        per_chunk_latencies = [0.05, 0.1, 0.08]
        chunk_size_secs = 0.128

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "latency_plot.png"

            # Test the function
            plot_latency(
                output_stream_latencies, per_chunk_latencies, chunk_size_secs, str(output_path)
            )

            # Check that savefig was called
            mock_savefig.assert_called_once()

    @patch("client.s2s.latency_analysis.plt.close")
    def test_plot_latency_plt_close_called(self, mock_close):
        """Test that plt.close is called correctly."""
        output_stream_latencies = [0.1, 0.2, 0.15]
        per_chunk_latencies = [0.05, 0.1, 0.08]
        chunk_size_secs = 0.128

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "latency_plot.png"

            # Test the function
            plot_latency(
                output_stream_latencies, per_chunk_latencies, chunk_size_secs, str(output_path)
            )

            # Check that plt.close was called
            mock_close.assert_called_once()

    def test_plot_latency_creates_directory(self):
        """Test that plot_latency creates output directory if it doesn't exist."""
        output_stream_latencies = [0.1, 0.2, 0.15]
        per_chunk_latencies = [0.05, 0.1, 0.08]
        chunk_size_secs = 0.128

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a nested directory path that doesn't exist
            nested_dir = Path(temp_dir) / "nested" / "subdirectory"
            output_path = nested_dir / "latency_plot.png"

            # Test that the function creates the directory and saves the plot
            plot_latency(
                output_stream_latencies,
                per_chunk_latencies,
                chunk_size_secs,
                str(output_path),
            )

            # Check that the directory was created and the plot file exists
            assert nested_dir.exists()
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_plot_latency_statistics_calculation(self):
        """Test that latency plotting handles statistics correctly."""
        output_stream_latencies = [0.1, 0.2, 0.15, 0.3, 0.25]
        per_chunk_latencies = [0.05, 0.1, 0.08, 0.12, 0.09]
        chunk_size_secs = 0.128

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "latency_plot.png"

            # Test the function
            plot_latency(
                output_stream_latencies, per_chunk_latencies, chunk_size_secs, str(output_path)
            )

            # The function should handle the statistics internally
            assert output_path.exists()
            assert output_path.stat().st_size > 0


if __name__ == "__main__":
    pytest.main([__file__])
