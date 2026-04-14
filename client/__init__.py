# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""All client packages.

This package contains client implementations for the Content Localization Blueprint services.
It provides clients for:
- Controller Service (orchestrates all services)
- Direct Service clients (individual service access)
- ASD (Active Speaker Detection) Service
- S2S (Speech-to-Speech) Service
- LipSync Service
- Batch Processing (runs the pipeline on every video in a directory)
- Source simulators for testing

Client Types
============

Controller Client
-----------------
The main client that connects to the Controller Service, which orchestrates
all downstream services (S2S, ASD, LipSync).

Direct Clients
--------------
Individual clients that connect directly to specific services:
- ASD client for speaker detection
- S2S client for speech-to-speech translation
- LipSync client for lip synchronization

Batch Processing Client
-----------------------
Runs the end-to-end pipeline on every video in a directory and produces
a timing report.

Source Simulators
------------------
Utilities for simulating audio and video input streams for testing.
"""

__version__ = "0.1.0"
