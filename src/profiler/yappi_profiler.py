# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Yappi Profiler Module

Provides multi-threaded CPU profiling using yappi with ~20-30% overhead.
Generates per-thread profiles and merged Chrome trace visualization.

Best for multi-threaded applications. For single-threaded code, use CProfileProfiler.

See PROFILING_GUIDE.md for complete documentation.
"""

import datetime
import json
import os
import pstats
from types import SimpleNamespace
from typing import Any

try:
    import yappi

    _YAPPI_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    # Allow module import so tests can monkeypatch a stub.
    yappi = SimpleNamespace()
    _YAPPI_IMPORT_ERROR = exc
from base_utils import logger
from profiler.profiler_interface import ProfilerInterface


def pstats_to_trace_events(
    stats: pstats.Stats,
    pid: int = 0,
    tid: int = 0,
) -> list[dict[str, Any]]:
    """Convert pstats to Chrome trace event format.

    Creates ``"Complete"`` events (``ph: "X"``) for each profiled
    function, sorted by cumulative time.

    Args:
        stats (pstats.Stats): pstats object containing profiling data.
        pid (int): Process ID to use in trace events. Defaults to ``0``.
        tid (int): Thread ID to use in trace events. Defaults to ``0``.

    Returns:
        list[dict[str, Any]]: List of trace event dictionaries.

    Examples:
        >>> import pstats
        >>> s = pstats.Stats()  # doctest: +SKIP
        >>> events = pstats_to_trace_events(stats=s)  # doctest: +SKIP
    """
    events = []
    timestamp = 0

    sorted_stats = sorted(
        stats.stats.items(),
        key=lambda x: x[1][3],
        reverse=True,
    )

    for func_key, func_stats in sorted_stats:
        filename, line, func_name = func_key
        cc, nc, tt, ct, callers = func_stats

        event = {
            "name": f"{func_name}",
            "cat": "python",
            "ph": "X",
            "ts": timestamp,
            "dur": int(ct * 1_000_000),
            "pid": pid,
            "tid": tid,
            "args": {
                "file": filename,
                "line": line,
                "primitive_calls": cc,
                "total_calls": nc,
                "total_time": tt,
                "cumulative_time": ct,
            },
        }
        events.append(event)
        timestamp += int(ct * 1_000_000) + 1

    return events


class YappiProfiler(ProfilerInterface):
    """
    Yappi-based profiler for multi-threaded CPU profiling with ~20-30% overhead.

    Generates profile_overall.prof, per-thread profile_thread_N.prof, and merged
    profile_trace.json in volumes/profiler/<timestamp>/<func_name>/ directory.

    Attributes:
        output_dir (str): Directory where profile outputs are saved
        func_name (str): Name of the profiled operation
    """

    def __init__(self):
        """Initialize the yappi profiler with default state."""
        super().__init__()
        self.output_dir = None
        self.func_name = None

    @staticmethod
    def _ensure_yappi_available():
        required_attrs = (
            "set_clock_type",
            "start",
            "stop",
            "get_func_stats",
            "get_thread_stats",
            "clear_stats",
        )
        missing = [name for name in required_attrs if not hasattr(yappi, name)]
        if missing:
            message = (
                "yappi is required for YappiProfiler but is not installed. "
                "Install it with `pip install yappi`."
            )
            if _YAPPI_IMPORT_ERROR is not None:
                raise ModuleNotFoundError(message) from _YAPPI_IMPORT_ERROR
            raise ModuleNotFoundError(message)

    def start(self, func_name: str = ""):
        """
        Start profiling for the named operation.

        Args:
            func_name (str): Name of operation being profiled (used for output directory)

        Note:
            Warns and returns if profiler is already active.
        """
        logger.info(f"Starting profiler for {func_name}...")
        self._ensure_yappi_available()
        if self.profiling_active:
            logger.warning("Profiler is already active, skipping start.")
            return
        self.func_name = func_name
        self.output_dir = os.path.join(
            self.output_base_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), func_name
        )
        logger.info(f"Profiler output path set to {self.output_dir}...")
        os.makedirs(self.output_dir, exist_ok=True)

        # Configure Yappi to use wall clock time instead of CPU time
        yappi.set_clock_type("WALL")
        logger.info("Yappi configured to use wall clock time")

        yappi.start()
        self.profiling_active = True

    def stop(self):
        """
        Stop profiling and save results.

        Saves profile_overall.prof, per-thread profile_thread_N.prof files, and merged
        profile_trace.json with all threads. Conversion failures are logged as warnings.

        Note:
            Warns and returns if profiler is not active.
        """
        logger.info(f"Stopping profiler for {self.func_name}...")
        if not self.profiling_active:
            logger.warning("Profiler is not active, skipping stop.")
            return
        self._ensure_yappi_available()
        yappi.stop()
        self.profiling_active = False

        output_path = os.path.join(self.output_dir, "profile_overall.prof")
        yappi.get_func_stats().save(output_path, type="pstat")
        logger.info(f"Overall profiler output saved to {output_path}")

        # Get thread stats and create merged trace with all threads
        thread_stats = yappi.get_thread_stats()
        all_trace_events = []

        for thread in thread_stats:
            yfn_obj = yappi.get_func_stats(ctx_id=thread.id)
            yfn_obj.sort(sort_type="ttot", sort_order="desc")
            thread_prof_path = os.path.join(self.output_dir, f"profile_thread_{thread.id}.prof")
            yfn_obj.save(thread_prof_path, type="pstat")
            logger.info(f"Thread {thread.id} profiler output saved to {thread_prof_path}")

            # Convert thread profile to trace events with proper thread ID
            try:
                # Load the thread's pstats
                stats = pstats.Stats(thread_prof_path)

                # Get the actual OS thread ID from yappi (thread[2])
                os_thread_id = thread[2]

                # Convert to trace events with the thread's actual OS thread ID
                thread_events = pstats_to_trace_events(stats, pid=0, tid=os_thread_id)
                all_trace_events.extend(thread_events)

                logger.info(
                    f"Thread {thread.name} (ctx={thread.id}, tid={os_thread_id}) "
                    f"added {len(thread_events)} trace events"
                )
            except Exception as e:
                logger.warning(f"Failed to convert thread {thread.id} to trace format: {e}")

        # Write merged trace file with all threads
        if all_trace_events:
            try:
                merged_trace_path = os.path.join(self.output_dir, "profile_trace.json")
                trace_data = {
                    "traceEvents": all_trace_events,
                    "displayTimeUnit": "ms",
                    "meta_user": "yappi_profiler",
                    "meta_cpu_count": 1,
                }

                with open(merged_trace_path, "w") as f:
                    json.dump(trace_data, f, indent=2)

                logger.info(
                    f"Merged trace with {len(all_trace_events)} events from "
                    f"{len(thread_stats)} threads saved to {merged_trace_path}"
                )
                logger.info("Open chrome://tracing and load the file to visualize all threads")
            except Exception as e:
                logger.warning(f"Failed to write merged trace file: {e}")

        yappi.clear_stats()
