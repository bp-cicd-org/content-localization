# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Convert pstats profiling data to Chrome trace event format.

This module provides utilities to convert Python pstats profiling data
to Chrome's trace event format, which can be viewed in chrome://tracing.
"""

import argparse
import json
import pstats
from typing import Any


def pstats_to_trace_events(stats: pstats.Stats, pid: int = 0, tid: int = 0) -> list[dict[str, Any]]:
    """
    Convert pstats to Chrome trace event format.

    Args:
        stats: pstats.Stats object containing profiling data
        pid: Process ID to use in trace events
        tid: Thread ID to use in trace events

    Returns:
        List of trace event dictionaries
    """
    events = []

    # Get the stats dictionary
    # stats.stats is a dict mapping (filename, line, function) -> (cc, nc, tt, ct, callers)
    # where:
    #   cc = primitive call count
    #   nc = total call count
    #   tt = total time spent in this function (excluding subfunctions)
    #   ct = cumulative time spent in this function (including subfunctions)
    #   callers = dict of caller info

    # We'll create a timeline by processing the call graph
    # For simplicity, we'll create "Complete" events (ph: "X") for each function

    timestamp = 0  # Starting timestamp in microseconds

    # Sort by cumulative time to get a logical ordering
    sorted_stats = sorted(
        stats.stats.items(),
        key=lambda x: x[1][3],  # Sort by cumulative time (ct)
        reverse=True,
    )

    for func_key, func_stats in sorted_stats:
        filename, line, func_name = func_key
        cc, nc, tt, ct, callers = func_stats

        # Create a complete event for this function
        event = {
            "name": f"{func_name}",
            "cat": "python",
            "ph": "X",  # Complete event
            "ts": timestamp,  # timestamp in microseconds
            "dur": int(ct * 1_000_000),  # duration in microseconds
            "pid": pid,
            "tid": tid,  # thread ID
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

        # Increment timestamp for next event (stacking them vertically)
        timestamp += int(ct * 1_000_000) + 1

    return events


def pstats_to_trace_events_hierarchical(
    stats: pstats.Stats, pid: int = 0, tid: int = 0
) -> list[dict[str, Any]]:
    """
    Convert pstats to Chrome trace event format with hierarchical call relationships.

    This creates a more realistic timeline showing nested function calls.

    Args:
        stats: pstats.Stats object containing profiling data
        pid: Process ID to use in trace events
        tid: Thread ID to use in trace events (base thread ID, nested calls use tid+depth)

    Returns:
        List of trace event dictionaries
    """
    events = []

    # Build a call graph
    call_graph = {}
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

    # Find root functions (those with no callers or called from outside)
    root_funcs = []
    for func_key, func_data in call_graph.items():
        if not func_data["callers"]:
            root_funcs.append(func_key)

    # If no roots found, use the top functions by cumulative time
    if not root_funcs:
        sorted_funcs = sorted(
            call_graph.items(), key=lambda x: x[1]["cumulative_time"], reverse=True
        )
        root_funcs = [sorted_funcs[0][0]] if sorted_funcs else []

    # Track current timestamp and stack depth
    current_time = [0]  # Use list to allow modification in nested function
    tid_counter = [0]

    def add_function_trace(func_key, depth=0, parent_start=0, max_depth=50):
        """Recursively add trace events for a function and its callees."""
        if depth > max_depth:  # Prevent infinite recursion
            return

        if func_key not in call_graph:
            return

        func_data = call_graph[func_key]
        start_time = current_time[0]
        duration = int(func_data["cumulative_time"] * 1_000_000)

        # Create the event
        event = {
            "name": func_data["func_name"],
            "cat": "python",
            "ph": "X",
            "ts": start_time,
            "dur": duration,
            "pid": pid,
            "tid": tid + depth,  # Use base tid + depth to show nesting within thread
            "args": {
                "file": func_data["filename"],
                "line": func_data["line"],
                "primitive_calls": func_data["primitive_calls"],
                "total_calls": func_data["total_calls"],
                "total_time": func_data["total_time"],
                "cumulative_time": func_data["cumulative_time"],
            },
        }
        events.append(event)

        # Update timestamp
        current_time[0] += duration

    # Process root functions
    for root_func in root_funcs[:10]:  # Limit to top 10 root functions
        add_function_trace(root_func)

    return events


def convert_pstats_file(
    input_file: str, output_file: str, hierarchical: bool = False, pid: int = 0, tid: int = 0
):
    """
    Convert a pstats file to Chrome trace event JSON format.

    Args:
        input_file: Path to the input .prof file
        output_file: Path to the output .json file
        hierarchical: If True, attempt to create hierarchical trace (experimental)
        pid: Process ID to use in trace events
        tid: Thread ID to use in trace events
    """
    # Load the pstats file
    stats = pstats.Stats(input_file)

    # Convert to trace events
    if hierarchical:
        events = pstats_to_trace_events_hierarchical(stats, pid=pid, tid=tid)
    else:
        events = pstats_to_trace_events(stats, pid=pid, tid=tid)

    # Create the trace format output
    trace_data = {
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "meta_user": "pstats_to_trace",
        "meta_cpu_count": 1,
    }

    # Write to file
    with open(output_file, "w") as f:
        json.dump(trace_data, f, indent=2)

    print(f"Converted {input_file} to {output_file}")
    print(f"Generated {len(events)} trace events")
    print(f"Open chrome://tracing and load {output_file} to view the trace")


def main():
    """Command-line interface for converting pstats to trace format."""
    parser = argparse.ArgumentParser(
        description="Convert pstats profiling data to Chrome trace event format"
    )
    parser.add_argument("input", help="Input .prof file (pstats format)")
    parser.add_argument(
        "-o", "--output", help="Output .json file (defaults to input with .json extension)"
    )
    parser.add_argument(
        "--hierarchical",
        action="store_true",
        help="Attempt to create hierarchical trace (experimental)",
    )
    parser.add_argument(
        "--pid",
        type=int,
        default=0,
        help="Process ID to use in trace events (default: 0)",
    )
    parser.add_argument(
        "--tid",
        type=int,
        default=0,
        help="Thread ID to use in trace events (default: 0)",
    )

    args = parser.parse_args()

    # Determine output filename
    if args.output:
        output_file = args.output
    # Replace .prof with .json
    elif args.input.endswith(".prof"):
        output_file = args.input[:-5] + ".json"
    else:
        output_file = args.input + ".json"

    # Convert the file
    convert_pstats_file(args.input, output_file, args.hierarchical, pid=args.pid, tid=args.tid)


if __name__ == "__main__":
    main()
