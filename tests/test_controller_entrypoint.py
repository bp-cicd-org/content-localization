# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for controller entrypoint argument wiring."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from controller_service import entrypoint
from controller_service.service import ControllerService

pytestmark = pytest.mark.unit


def _base_argv() -> list[str]:
    return [
        "controller-entrypoint",
        "--service-uri",
        "controller:50056",
        "--max-concurrency",
        "1",
        "--concurrency-mode",
        "threading",
        "--threads-per-process",
        "1",
        "--s2s-server",
        "speech-to-speech:50050",
        "--lipsync-server",
        "lipsync:50054",
    ]


@pytest.mark.unit
class TestControllerEntrypoint(unittest.TestCase):
    """Regression tests for controller startup arg wiring."""

    def test_argsfactory_does_not_expose_service_mode(self) -> None:
        """Controller args are push-only and do not require --service-mode."""
        parser = ControllerService.argsfactory()
        all_options = {
            option for action in parser._actions for option in getattr(action, "option_strings", [])
        }
        self.assertNotIn("--service-mode", all_options)

    @patch("controller_service.entrypoint.ControllerService")
    @patch("controller_service.entrypoint.GRPCServer.from_string")
    @patch("controller_service.entrypoint.SpeechToSpeechServer")
    @patch("controller_service.entrypoint.LipsyncServer")
    @patch("controller_service.entrypoint.ActiveSpeakerDetectionServer")
    def test_main_wires_controller_without_service_mode(
        self,
        mock_asd_server_ctor: MagicMock,
        mock_lipsync_server_ctor: MagicMock,
        mock_s2s_server_ctor: MagicMock,
        mock_from_string: MagicMock,
        mock_controller_service: MagicMock,
    ) -> None:
        """Entrypoint builds ControllerService without obsolete service_mode."""
        # from_string call order: service_uri, lipsync, s2s, asd
        # (mock parse_args returns truthy attrs, so all branches execute)
        mock_from_string.side_effect = [
            SimpleNamespace(host="controller", port=50056),
            SimpleNamespace(host="lipsync", port=50054),
            SimpleNamespace(host="speech-to-speech", port=50050),
            SimpleNamespace(host="asd", port=50055),
        ]
        mock_s2s_server_ctor.return_value = MagicMock()
        mock_lipsync_server_ctor.return_value = MagicMock()
        mock_asd_server_ctor.return_value = MagicMock()
        mock_controller_service.return_value = MagicMock()

        argv = _base_argv()
        with patch("sys.argv", argv):
            entrypoint.main()

        kwargs = mock_controller_service.call_args.kwargs
        self.assertNotIn("service_mode", kwargs)
        mock_controller_service.return_value.serve.assert_called_once()

    def test_argsfactory_supports_asd_server(self) -> None:
        """Parser accepts explicit ASD endpoint."""
        parser = ControllerService.argsfactory()
        args = parser.parse_args([*_base_argv()[1:], "--asd-server", "asd:50055"])

        self.assertEqual(args.asd_server, "asd:50055")
