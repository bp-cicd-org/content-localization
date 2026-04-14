# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock

import grpc

from profiler import cprofile_profiler
from profiler import yappi_profiler
from profiler.cprofile_profiler import CProfileProfiler
from profiler.metrics_tracker import MetricsTracker
from profiler.profiler_grpc_interceptor import ProfileInterceptor
from profiler.profiler_grpc_interceptor import get_rpc_handler
from profiler.profiler_grpc_interceptor import run_profiler
from profiler.profiler_grpc_interceptor import split_method_call
from profiler.profiler_interface import ProfilerInterface


def _reset_metrics_tracker(monkeypatch, enabled=True):
    monkeypatch.setattr(MetricsTracker, "_enable_metric_tracking", enabled)
    MetricsTracker._instance = None
    tracker = MetricsTracker()
    tracker.metrics = {}
    tracker.registered_metrics = set()
    return tracker


def test_metrics_tracker_records_and_stats(monkeypatch):
    tracker = _reset_metrics_tracker(monkeypatch, enabled=True)
    tracker.register_metric("foo")
    tracker.record_metric("foo", timestamp=1.0)
    tracker.record_metric("foo", timestamp=3.0)

    stats = tracker.get_metric_stats("foo")
    assert stats["count"] == 2
    assert stats["min_interval"] == 2.0
    assert tracker.get_metric_intervals("foo") == [2.0]


def test_metrics_tracker_dump_raw_csv(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTROLLER_PROFILER_OUTPUT_DIR", str(tmp_path))
    tracker = _reset_metrics_tracker(monkeypatch, enabled=True)
    tracker.register_metric("bar")
    tracker.record_metric("bar", timestamp=1.0)
    tracker.dump_metrics_to_file("metrics", raw_format=True)

    assert (tmp_path / "metrics" / "bar.csv").exists()


def test_profiler_interface_state():
    profiler = ProfilerInterface()
    profiler.profiling_active = True
    assert profiler.profiling_active is True


def test_cprofile_profiler_start_stop(tmp_path, monkeypatch):
    class DummyProfile:
        def __init__(self):
            self.dumped_path = None

        def enable(self):
            return None

        def disable(self):
            return None

        def dump_stats(self, path):
            self.dumped_path = path

    monkeypatch.setenv("CONTROLLER_PROFILER_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(cprofile_profiler.cProfile, "Profile", DummyProfile)
    monkeypatch.setattr(
        cprofile_profiler, "pstats_to_trace_events_hierarchical", lambda *a, **k: None
    )

    profiler = CProfileProfiler()
    profiler.start("unit")
    assert profiler.profiling_active is True
    assert profiler.output_dir.startswith(str(tmp_path))
    assert Path(profiler.output_dir).exists()

    profiler.stop()
    assert profiler.profiling_active is False
    assert profiler.profiler is None


def test_yappi_profiler_start_stop(tmp_path, monkeypatch):
    class DummyYappiStats:
        def save(self, *args, **kwargs):
            return None

        def sort(self, *args, **kwargs):
            return None

    class DummyThread:
        def __init__(self, thread_id, name, os_tid):
            self.id = thread_id
            self.name = name
            self._os_tid = os_tid

        def __getitem__(self, idx):
            if idx == 2:
                return self._os_tid
            raise IndexError

    class DummyPstats:
        def __init__(self, *args, **kwargs):
            return None

    monkeypatch.setenv("CONTROLLER_PROFILER_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(yappi_profiler.yappi, "set_clock_type", lambda *a, **k: None)
    monkeypatch.setattr(yappi_profiler.yappi, "start", lambda *a, **k: None)
    monkeypatch.setattr(yappi_profiler.yappi, "stop", lambda *a, **k: None)
    monkeypatch.setattr(yappi_profiler.yappi, "get_func_stats", lambda *a, **k: DummyYappiStats())
    monkeypatch.setattr(
        yappi_profiler.yappi,
        "get_thread_stats",
        lambda *a, **k: [DummyThread(1, "worker", 123)],
    )
    monkeypatch.setattr(yappi_profiler.yappi, "clear_stats", lambda *a, **k: None)
    monkeypatch.setattr(yappi_profiler.pstats, "Stats", DummyPstats)
    monkeypatch.setattr(
        yappi_profiler,
        "pstats_to_trace_events",
        lambda *a, **k: [{"name": "event"}],
    )

    profiler = yappi_profiler.YappiProfiler()
    profiler.start("unit")
    profiler.stop()

    trace_file = Path(profiler.output_dir) / "profile_trace.json"
    assert trace_file.exists()


def test_split_method_call():
    details = MagicMock()
    details.method = "/svc/Method"
    assert split_method_call(details) == ("svc", "Method")


def test_get_rpc_handler_streaming_types():
    handler = MagicMock()
    handler.request_streaming = True
    handler.response_streaming = True
    handler.stream_stream = "stream_stream"
    behavior, factory = get_rpc_handler(handler)
    assert behavior == "stream_stream"
    assert factory == grpc.stream_stream_rpc_method_handler


def test_run_profiler_skips_health_check():
    profiler = MagicMock()
    behavior = MagicMock(return_value="ok")

    result = run_profiler(
        profiler=profiler,
        behavior=behavior,
        request_or_iterator="req",
        servicer_context=MagicMock(),
        grpc_service_name="grpc.health.v1.Health",
        grpc_method_name="Check",
    )

    assert result == "ok"
    profiler.start.assert_not_called()
    profiler.stop.assert_not_called()


def test_profile_interceptor_wraps_call(monkeypatch):
    behavior = MagicMock(return_value="resp")

    class DummyHandler:
        request_streaming = False
        response_streaming = False
        unary_unary = behavior
        request_deserializer = None
        response_serializer = None

    continuation = MagicMock(return_value=DummyHandler())
    interceptor = ProfileInterceptor(profiler=ProfilerInterface())
    call_details = MagicMock()
    call_details.method = "/svc/Method"

    handler = interceptor.intercept_service(continuation, call_details)
    response = handler.unary_unary("req", MagicMock())
    assert response == "resp"
