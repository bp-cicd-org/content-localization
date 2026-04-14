# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test configuration and fixtures."""

import os
import sys

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Ensure src/ is on the Python path for test imports."""
    src_path = os.path.join(os.path.dirname(__file__), "..", "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    yield
