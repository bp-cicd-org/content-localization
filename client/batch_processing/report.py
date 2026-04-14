# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch processing reporting: result storage, console output, JSON export."""

import json
import os
from dataclasses import asdict
from dataclasses import dataclass

_SECONDS_PER_MINUTE = 60
KB = 1024
MB = KB * KB


@dataclass
class BatchResult:
    """Result of processing a single video through the pipeline.

    Attributes:
        video_name: Input video filename.
        video_duration_secs: Duration of the input video.
        preprocess_time_secs: Audio extraction time.
        pipeline_time_secs: Controller service processing time.
        total_time_secs: Wall-clock time (preprocess + pipeline).
        output_path: Path to the output video file.
        output_size_bytes: Size of the output file in bytes.
        success: Whether the pipeline completed successfully.
        error_message: Error details if success is False.
    """

    video_name: str
    video_duration_secs: float
    preprocess_time_secs: float
    pipeline_time_secs: float
    total_time_secs: float
    output_path: str
    output_size_bytes: int
    success: bool
    error_message: str | None = None

    @property
    def realtime_factor(self) -> float:
        """Pipeline time divided by video duration.

        Returns:
            float: Real-time factor (< 1.0 = faster than real-time).

        Examples:
            >>> r = BatchResult(
            ...     "v.mp4",
            ...     10.0,
            ...     1.0,
            ...     5.0,
            ...     6.0,
            ...     "o.mp4",
            ...     100,
            ...     True,
            ... )
            >>> r.realtime_factor
            0.5
        """
        if self.video_duration_secs <= 0:
            return 0.0
        return self.pipeline_time_secs / self.video_duration_secs


def _fmt_duration(seconds: float) -> str:
    """Format seconds as a human-readable string.

    Args:
        seconds (float): Duration in seconds.

    Returns:
        str: Formatted string (e.g. ``'1m 23.4s'``).

    Examples:
        >>> _fmt_duration(83.4)
        '1m 23.4s'
    """
    if seconds < _SECONDS_PER_MINUTE:
        return f"{seconds:.1f}s"
    minutes = int(seconds // _SECONDS_PER_MINUTE)
    remaining = seconds % _SECONDS_PER_MINUTE
    return f"{minutes}m {remaining:.1f}s"


def _fmt_size(size_bytes: int) -> str:
    """Format bytes as a human-readable string.

    Args:
        size_bytes (int): Size in bytes.

    Returns:
        str: Formatted string (e.g. ``'12.3 MB'``).

    Examples:
        >>> _fmt_size(12_345_678)
        '11.8 MB'
    """
    if size_bytes < KB:
        return f"{size_bytes} B"
    if size_bytes < MB:
        return f"{size_bytes / KB:.1f} KB"
    return f"{size_bytes / MB:.1f} MB"


def print_report(results: list[BatchResult]) -> None:
    """Print a formatted batch processing summary to the console.

    Args:
        results (list[BatchResult]): Batch processing results.

    Examples:
        >>> print_report([result1, result2])  # doctest: +SKIP
    """
    sep = "=" * 72
    print(f"\n{sep}")
    print("BATCH PROCESSING REPORT")
    print(sep)

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    for result in results:
        status = "OK" if result.success else "FAIL"
        print(f"\n  [{status}] {result.video_name}")
        print(f"    Duration:    {_fmt_duration(result.video_duration_secs)}")
        print(f"    Preprocess:  {_fmt_duration(result.preprocess_time_secs)}")
        print(f"    Pipeline:    {_fmt_duration(result.pipeline_time_secs)}")
        print(f"    Total:       {_fmt_duration(result.total_time_secs)}")
        if result.success:
            print(f"    RT factor:   {result.realtime_factor:.2f}x")
            print(f"    Output size: {_fmt_size(result.output_size_bytes)}")
        else:
            print(f"    Error:       {result.error_message}")

    print(f"\n{sep}")
    print("SUMMARY")
    print(sep)
    print(f"  Total videos:  {len(results)}")
    print(f"  Successful:    {len(successful)}")
    print(f"  Failed:        {len(failed)}")

    if successful:
        avg_rt = sum(r.realtime_factor for r in successful) / len(successful)
        total_dur = sum(r.video_duration_secs for r in successful)
        total_pipe = sum(r.pipeline_time_secs for r in successful)
        total_wall = sum(r.total_time_secs for r in successful)
        print(f"  Avg RT factor: {avg_rt:.2f}x")
        print(f"  Total input:   {_fmt_duration(total_dur)}")
        print(f"  Total pipe:    {_fmt_duration(total_pipe)}")
        print(f"  Total wall:    {_fmt_duration(total_wall)}")

    print(f"{sep}\n")


def save_report(
    results: list[BatchResult],
    output_path: str,
) -> None:
    """Save batch processing results to a JSON file.

    Args:
        results (list[BatchResult]): Batch processing results.
        output_path (str): Path to the output JSON file.

    Examples:
        >>> save_report([result], "report.json")  # doctest: +SKIP
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    successful = [r for r in results if r.success]
    summary = {
        "total_videos": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "avg_realtime_factor": (
            sum(r.realtime_factor for r in successful) / len(successful) if successful else 0.0
        ),
        "total_input_duration_secs": sum(r.video_duration_secs for r in successful),
        "total_pipeline_time_secs": sum(r.pipeline_time_secs for r in successful),
        "total_wall_time_secs": sum(r.total_time_secs for r in successful),
    }

    data = {
        "results": [{**asdict(r), "realtime_factor": r.realtime_factor} for r in results],
        "summary": summary,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Report saved to: {output_path}")
