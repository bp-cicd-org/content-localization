# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Profiler Interface Module

Defines the base interface for all profiler implementations.
All profilers (cProfile, yappi) inherit from this interface.
"""

import os


class ProfilerInterface:
    """
    Base interface for profiler implementations.

    Attributes:
        output_base_dir (str): Base directory for profiler outputs
        profiling_active (bool): Whether profiling is currently active
    """

    def __init__(self):
        """Initialize profiler with output directory from environment variable."""
        self.output_base_dir = os.environ.get("CONTROLLER_PROFILER_OUTPUT_DIR", default="./")
        self.profiling_active = False

    def start(self, func_name: str = ""):
        """
        Start profiling for the named operation.

        Args:
            func_name (str): Name of operation being profiled
        """

    def stop(self):
        """Stop profiling and save results."""

    @property
    def profiling_active(self) -> bool:
        """
        Check if profiling is currently active.

        Returns:
            True if profiling is active, False otherwise
        """
        return self._profiling_active

    @profiling_active.setter
    def profiling_active(self, value: bool):
        """
        Set the profiling active state.

        Args:
            value: True to mark profiling as active, False otherwise
        """
        self._profiling_active = value
