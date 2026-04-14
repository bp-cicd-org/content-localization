# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstract classes for S2S source simulators."""

import os
from abc import ABC
from abc import abstractmethod


class BaseFileSimulator(ABC):
    """Base class for an audio source or sink simulator."""

    @property
    def file_path(self) -> os.PathLike:
        """Get the file path of the WAV file."""
        return self._file_path

    @file_path.setter
    def file_path(self, value: os.PathLike):
        """Set the file path of the WAV file."""
        self.validate_file_path(value)
        self._file_path = value

    @abstractmethod
    def validate_file_path(self, value: os.PathLike) -> None:
        """Validate the file path of the WAV file."""

    def __init__(self, file_path: os.PathLike):
        """Initialize AudioSimulator with a WAV file path."""
        self.file_path = file_path

        # Create a ledger to track the timestamps of the audio samples going, in/out.
        self.ledger = {}

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Call to close the input WAV file."""
        if hasattr(self, "_file_opened") and self._file_opened is not None:
            self._file_opened.close()
            self._file_opened = None

    def is_open(self) -> bool:
        """Check if the output WAV file is open."""
        return hasattr(self, "_file_opened") and self._file_opened is not None
