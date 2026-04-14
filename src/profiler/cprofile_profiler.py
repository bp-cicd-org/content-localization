# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
cProfile Profiler Module

Provides CPU profiling using Python's built-in cProfile with ~10-20% overhead.
Generates pstats format (.prof) and Chrome trace format (.json) for visualization.

Best for single-threaded applications. For multi-threaded code, use YappiProfiler.

See PROFILING_GUIDE.md for complete documentation.
"""

import cProfile
import datetime
import json
import os
import pstats
from typing import Any

from base_utils import logger
from profiler.profiler_interface import ProfilerInterface


def pstats_to_trace_events_hierarchical(
    stats: pstats.Stats,
    pid: int = 0,
    tid: int = 0,
) -> list[dict[str, Any]]:
    """Convert pstats to Chrome trace event format with hierarchy.

    Creates a timeline showing nested function calls, where each root
    function and its callees are represented as ``"Complete"`` events.

    Args:
        stats (pstats.Stats): pstats object containing profiling data.
        pid (int): Process ID to use in trace events. Defaults to ``0``.
        tid (int): Thread ID to use in trace events. Defaults to ``0``.

    Returns:
        list[dict[str, Any]]: List of trace event dictionaries.

    Examples:
        >>> import pstats
        >>> s = pstats.Stats()  # doctest: +SKIP
        >>> events = pstats_to_trace_events_hierarchical(
        ...     stats=s,
        ... )  # doctest: +SKIP
    """
    events: list[dict[str, Any]] = []

    call_graph: dict = {}
    for func_key, func_stats in stats.stats.items():
        filename, line, func_name = func_key
        cc, nc, tt, ct, callers = func_stats

        call_graph[func_key] = {
            "filename": filename,
            "line": line,
            "func_name": func_name,
            "primitive_calls": cc,
            "total_calls": nc,
            "total_time": tt,
            "cumulative_time": ct,
            "callers": callers,
        }

    root_funcs = [key for key, data in call_graph.items() if not data["callers"]]

    if not root_funcs:
        sorted_funcs = sorted(
            call_graph.items(),
            key=lambda x: x[1]["cumulative_time"],
            reverse=True,
        )
        root_funcs = [sorted_funcs[0][0]] if sorted_funcs else []

    current_time = [0]

    def _add_function_trace(
        func_key: tuple,
        depth: int = 0,
        max_depth: int = 50,
    ) -> None:
        if depth > max_depth or func_key not in call_graph:
            return

        func_data = call_graph[func_key]
        start_time = current_time[0]
        duration = int(func_data["cumulative_time"] * 1_000_000)

        events.append(
            {
                "name": func_data["func_name"],
                "cat": "python",
                "ph": "X",
                "ts": start_time,
                "dur": duration,
                "pid": pid,
                "tid": tid + depth,
                "args": {
                    "file": func_data["filename"],
                    "line": func_data["line"],
                    "primitive_calls": func_data["primitive_calls"],
                    "total_calls": func_data["total_calls"],
                    "total_time": func_data["total_time"],
                    "cumulative_time": func_data["cumulative_time"],
                },
            }
        )
        current_time[0] += duration

    for root_func in root_funcs[:10]:
        _add_function_trace(func_key=root_func)

    return events


class CProfileProfiler(ProfilerInterface):
    """
    cProfile-based profiler for CPU profiling with ~10-20% overhead.

    Generates profile_overall.prof (pstats) and profile_trace.json (Chrome trace)
    in volumes/profiler/<timestamp>/<func_name>/ directory.

    Attributes:
        output_dir (str): Directory where profile outputs are saved
        func_name (str): Name of the profiled operation
        profiler (cProfile.Profile): Underlying cProfile instance
    """

    def __init__(self):
        """Initialize the cProfile profiler with default state."""
        super().__init__()
        self.output_dir = None
        self.func_name = None
        self.profiler = None

    def start(self, func_name: str = ""):
        """
        Start profiling for the named operation.

        Args:
            func_name (str): Name of operation being profiled (used for output directory)

        Note:
            Warns and returns if profiler is already active.
        """
        logger.info(f"Starting cProfile profiler for {func_name}...")
        if self.profiling_active:
            logger.warning("Profiler is already active, skipping start.")
            return
        self.func_name = func_name
        self.output_dir = os.path.join(
            self.output_base_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), func_name
        )
        logger.info(f"Profiler output path set to {self.output_dir}...")
        os.makedirs(self.output_dir, exist_ok=True)

        # Create and start cProfile profiler
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        self.profiling_active = True

    def stop(self):
        """
        Stop profiling and save results.

        Saves profile_overall.prof (pstats) and profile_trace.json (Chrome trace).
        Conversion failures are logged as warnings but don't prevent saving pstats.

        Note:
            Warns and returns if profiler is not active.
        """
        logger.info(f"Stopping cProfile profiler for {self.func_name}...")
        if not self.profiling_active or self.profiler is None:
            logger.warning("Profiler is not active, skipping stop.")
            return

        # Stop the profiler
        self.profiler.disable()
        self.profiling_active = False

        # Save overall profiler output
        output_path = os.path.join(self.output_dir, "profile_overall.prof")
        self.profiler.dump_stats(output_path)
        logger.info(f"Overall profiler output saved to {output_path}")

        # Convert to Chrome trace event format
        try:
            trace_output_path = os.path.join(
                self.output_dir,
                "profile_trace.json",
            )
            stats = pstats.Stats(output_path)
            trace_events = pstats_to_trace_events_hierarchical(stats=stats)
            trace_data = {
                "traceEvents": trace_events,
                "displayTimeUnit": "ms",
                "meta_user": "cprofile_profiler",
                "meta_cpu_count": 1,
            }
            with open(trace_output_path, "w") as f:
                json.dump(trace_data, f, indent=2)
            logger.info(f"Chrome trace format saved to {trace_output_path}")
            logger.info("Open chrome://tracing and load the file to visualize")
        except Exception as e:
            logger.warning(f"Failed to convert to trace format: {e}")

        # Clear the profiler
        self.profiler = None
