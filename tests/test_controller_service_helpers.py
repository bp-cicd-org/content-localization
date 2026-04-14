# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
from nvidia.ai4m.controller.v1.controller_pb2 import ContentLocalizationResponse

from controller_service.conversions import create_wav_header
from controller_service.service import ControllerService
from controller_service.service import ControllerServiceServicer


def test_create_wav_header_basic():
    header = create_wav_header(n_channels=1, sample_width=2, frame_rate=16000, n_frames=0)
    assert header.startswith(b"RIFF")
    assert b"WAVE" in header
    assert len(header) == 44


def test_check_services_health_calls_servers():
    lipsync_server = MagicMock()
    s2s_server = MagicMock()
    asd_server = MagicMock()
    controller = ControllerService(
        lipsync_server=lipsync_server,
        s2s_server=s2s_server,
        asd_server=asd_server,
    )
    controller._check_services_health()
    lipsync_server.is_healthy.assert_called_once()
    s2s_server.is_healthy.assert_called_once()
    asd_server.is_healthy.assert_called_once()


def test_servicer_streams_responses():
    response_iter = iter([ContentLocalizationResponse(), ContentLocalizationResponse()])
    service = MagicMock()
    service.infer.return_value = response_iter
    service.intermediate_audio_format = "MP3"
    servicer = ControllerServiceServicer(service)
    context = MagicMock()
    context.peer.return_value = "peer"

    results = list(
        servicer.StreamContentLocalization(
            request_iterator=iter([]),
            context=context,
        )
    )
    assert len(results) == 2
    service.infer.assert_called_once()


def test_servicer_aborts_on_infer_error():
    service = MagicMock()
    service.infer.side_effect = RuntimeError("boom")
    service.intermediate_audio_format = "MP3"
    servicer = ControllerServiceServicer(service)
    context = MagicMock()
    context.peer.return_value = "peer"
    context.abort.side_effect = RuntimeError("aborted")

    with pytest.raises(RuntimeError, match="aborted"):
        list(
            servicer.StreamContentLocalization(
                request_iterator=iter([]),
                context=context,
            )
        )
    context.abort.assert_called_once()
