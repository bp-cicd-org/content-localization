# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source and sink simulators for testing and development.

This package provides utilities for simulating audio and video input/output streams
for testing the Content Localization Blueprint services.

Components
==========

- audio: Audio source and sink simulators
- video: Video source and sink simulators
- base: Base classes for simulator implementations

Usage
=====

.. code-block:: python

   from client.source_simulators.audio import AudioSourceSimulator
   from client.source_simulators.video import VideoSourceSimulator
   from client.source_simulators.base import BaseFileSimulator

   # Create audio simulator
   audio_sim = AudioSourceSimulator("input.wav")

   # Create video simulator
   video_sim = VideoSourceSimulator("input.mp4")

Features
========

- Simulate audio and video input streams
- Support for various file formats
- Configurable chunk sizes and timing
- Iterator-based streaming interface
- Testing utilities for client applications
"""
