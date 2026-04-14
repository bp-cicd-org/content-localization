# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for push-only controller docker/config wiring."""

import unittest
from pathlib import Path

import pytest

from controller_service.service import ControllerService

pytestmark = pytest.mark.unit


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.unit
class TestControllerDockerConfigConsistency(unittest.TestCase):
    """Regression checks for push-only controller runtime wiring."""

    def test_service_mode_removed_from_runtime_wiring(self) -> None:
        """Compose and entrypoint shell do not reference removed service-mode."""
        compose_text = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        shell_text = (REPO_ROOT / "src/docker_entrypoints/controller/entrypoint.sh").read_text(
            encoding="utf-8"
        )
        default_env = (REPO_ROOT / "configs/elevenlabs.env").read_text(encoding="utf-8")
        riva_env = (REPO_ROOT / "configs/riva.env").read_text(encoding="utf-8")

        self.assertNotIn("CONTROLLER_SERVICE_MODE", compose_text)
        self.assertNotIn("--service-mode", shell_text)
        self.assertNotIn("CONTROLLER_SERVICE_MODE", default_env)
        self.assertNotIn("CONTROLLER_SERVICE_MODE", riva_env)

        # bypass_asd is now per-request, not deployment-time
        self.assertNotIn("CONTROLLER_NO_ASD", compose_text)
        self.assertNotIn("CONTROLLER_NO_ASD", default_env)
        self.assertNotIn("CONTROLLER_NO_ASD", riva_env)
        self.assertNotIn("--no-asd", shell_text)

    def test_argsfactory_omits_service_mode(self) -> None:
        """Controller parser omits removed service-mode argument."""
        parser = ControllerService.argsfactory()
        all_options = {
            option for action in parser._actions for option in getattr(action, "option_strings", [])
        }
        self.assertNotIn("--service-mode", all_options)
        self.assertNotIn("--no-asd", all_options)
        self.assertIn("--asd-server", all_options)
