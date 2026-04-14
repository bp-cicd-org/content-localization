# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("language")
    default_source_language = os.environ.get("TEST_SOURCE_LANGUAGE")
    default_target_language = os.environ.get("TEST_TARGET_LANGUAGE")
    default_audio_format = os.environ.get("TEST_AUDIO_FORMAT")
    group.addoption(
        "--source-language",
        action="store",
        default=default_source_language,
        help=(
            "Source language code to use in functional tests. "
            "Uses TEST_SOURCE_LANGUAGE if set, otherwise client defaults."
        ),
    )
    group.addoption(
        "--target-language",
        action="store",
        default=default_target_language,
        help=(
            "Target language code to use in functional tests. "
            "Uses TEST_TARGET_LANGUAGE if set, otherwise client defaults."
        ),
    )
    group.addoption(
        "--audio-format",
        action="store",
        default=default_audio_format,
        help=(
            "Audio format for outputs in functional tests: wav or mp3. "
            "Uses TEST_AUDIO_FORMAT if set, otherwise client defaults."
        ),
    )


@pytest.fixture
def source_language(pytestconfig):
    return pytestconfig.getoption("--source-language")


@pytest.fixture
def target_language(pytestconfig):
    return pytestconfig.getoption("--target-language")


@pytest.fixture
def audio_format(pytestconfig):
    value = pytestconfig.getoption("--audio-format")
    if value is None:
        return None
    return str(value).lower()
