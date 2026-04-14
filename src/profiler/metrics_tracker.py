# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import os
import threading
import time
from pathlib import Path

from base_utils import logger


class MetricsTracker:
    """A singleton class to track and compute statistics for various metrics."""

    _instance = None
    _lock = threading.Lock()
    _enable_metric_tracking = os.environ.get("CONTROLLER_METRIC_TRACKER", 0) == 1

    def __new__(cls):
        """Create a singleton instance of MetricsTracker."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(MetricsTracker, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the metrics tracker (only once for singleton)."""
        # Only initialize if not already initialized
        if not hasattr(self, "_initialized"):
            self.metrics: dict[str, list[float]] = {}
            self.registered_metrics: set = set()
            self._initialized = True

    def register_metric(self, metric_name: str) -> None:
        """
        Register a new metric for tracking.

        Args:
            metric_name: The name of the metric to register
        """
        if not self._enable_metric_tracking:
            return

        if metric_name in self.registered_metrics:
            logger.warning(f"Metric '{metric_name}' is already registered")
            return

        self.registered_metrics.add(metric_name)
        self.metrics[metric_name] = []
        logger.debug(f"Registered metric: {metric_name}")

    def record_metric(self, metric_name: str, timestamp: float | None = None) -> None:
        """
        Record a timestamp for a specific metric event.

        Args:
            metric_name: The name of the metric to record
            timestamp: Optional timestamp (defaults to current time)
        """
        if not self._enable_metric_tracking:
            return

        if metric_name not in self.registered_metrics:
            try:
                self.register_metric(metric_name)
            except Exception as e:
                logger.error(f"Error registering metric '{metric_name}': {e}")
                return

        if timestamp is None:
            timestamp = time.time()

        self.metrics[metric_name].append(timestamp)

    def get_metric_stats(self, metric_name: str) -> dict[str, float]:
        """
        Get statistics for a specific metric.

        Args:
            metric_name: The name of the metric

        Returns:
            Dictionary containing count, first_timestamp, last_timestamp,
            and timing statistics (min_interval, max_interval, avg_interval)
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {
                "count": 0,
                "first_timestamp": 0.0,
                "last_timestamp": 0.0,
                "min_interval": 0.0,
                "max_interval": 0.0,
                "avg_interval": 0.0,
            }

        timestamps = self.metrics[metric_name]

        # Calculate time intervals between consecutive events
        intervals = []
        if len(timestamps) > 1:
            for i in range(1, len(timestamps)):
                interval = timestamps[i] - timestamps[i - 1]
                intervals.append(interval)

        # Calculate interval statistics
        if intervals:
            min_interval = min(intervals)
            max_interval = max(intervals)
            avg_interval = sum(intervals) / len(intervals)
        else:
            min_interval = 0.0
            max_interval = 0.0
            avg_interval = 0.0

        return {
            "count": len(timestamps),
            "first_timestamp": min(timestamps),
            "last_timestamp": max(timestamps),
            "min_interval": min_interval,
            "max_interval": max_interval,
            "avg_interval": avg_interval,
        }

    def get_all_metrics_stats(self) -> dict[str, dict[str, float]]:
        """
        Get statistics for all registered metrics.

        Returns:
            Dictionary with metric names as keys and their statistics as values
        """
        all_stats = {}
        for metric_name in self.registered_metrics:
            all_stats[metric_name] = self.get_metric_stats(metric_name)
        return all_stats

    def dump_metrics_to_file(self, file_name: str, raw_format: bool = False) -> None:
        """
        Dump all metric statistics or raw data to a CSV file.

        Args:
            filepath: Path to the output CSV file
            raw_format: If True, dump raw timestamped data; if False, dump statistics
        """
        if not self._enable_metric_tracking:
            return

        file_path = Path(
            os.path.join(os.environ.get("CONTROLLER_PROFILER_OUTPUT_DIR", "./"), file_name)
        )

        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if raw_format:
            data = self.get_all_metrics_timestamps()
            self._dump_raw_csv(data, file_path)
        else:
            stats = self.get_all_metrics_stats()
            self._dump_stats_csv(stats, file_path)

        logger.info(f"Metrics dumped to {file_path}")

    def _dump_stats_csv(self, stats: dict[str, dict[str, float]], file_path: Path) -> None:
        """
        Dump metrics statistics in CSV format.

        Args:
            stats: Dictionary of metric statistics from get_all_metrics_stats()
            file_path: Path where the CSV file will be written
        """
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(
                [
                    "metric_name",
                    "count",
                    "first_timestamp",
                    "last_timestamp",
                    "first_datetime",
                    "last_datetime",
                    "min_interval",
                    "max_interval",
                    "avg_interval",
                ]
            )

            # Write data rows
            for metric_name, metric_stats in stats.items():
                first_datetime = (
                    time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(metric_stats["first_timestamp"])
                    )
                    if metric_stats["first_timestamp"] > 0
                    else ""
                )
                last_datetime = (
                    time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(metric_stats["last_timestamp"])
                    )
                    if metric_stats["last_timestamp"] > 0
                    else ""
                )

                writer.writerow(
                    [
                        metric_name,
                        int(metric_stats["count"]),
                        metric_stats["first_timestamp"],
                        metric_stats["last_timestamp"],
                        first_datetime,
                        last_datetime,
                        metric_stats["min_interval"],
                        metric_stats["max_interval"],
                        metric_stats["avg_interval"],
                    ]
                )

    def _dump_raw_csv(self, data: dict[str, list[float]], file_path: Path) -> None:
        """
        Dump raw metrics timestamps in CSV format.

        Args:
            data: Dictionary of metric timestamps from get_all_metrics_timestamps()
            file_path: Base directory path where CSV files will be written

        Note:
            Each metric is saved to a separate file named after the metric
            in the format <metric_name>.csv within the base directory.
        """
        # Get the directory from the file_path
        base_dir = file_path.parent if file_path.is_file() else file_path
        base_dir.mkdir(parents=True, exist_ok=True)

        # Save each metric to a separate file
        for metric_name, timestamps in data.items():
            # Create filename based on metric name
            metric_file_path = base_dir / f"{metric_name}.csv"

            with open(metric_file_path, "w", newline="") as f:
                writer = csv.writer(f)

                # Write header (without metric_name column)
                writer.writerow(["timestamp", "datetime"])

                # Write data rows
                for timestamp in timestamps:
                    datetime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
                    writer.writerow([timestamp, datetime_str])

    def clear_metrics(self) -> None:
        """
        Clear all recorded metric values but keep registered metrics.

        Note:
            This removes all timestamp data but retains the metric registrations,
            allowing you to continue recording to the same metrics.
        """
        for metric_name in self.metrics:
            self.metrics[metric_name] = []

    def reset(self) -> None:
        """
        Reset the tracker completely (clear all metrics and registrations).

        Note:
            This removes both the metric data and registrations. You will need
            to re-register metrics before recording new data.
        """
        self.metrics.clear()
        self.registered_metrics.clear()

    def get_metric_count(self, metric_name: str) -> int:
        """
        Get the number of recorded values for a metric.

        Args:
            metric_name: The name of the metric

        Returns:
            Number of recorded timestamp values for the metric (0 if metric doesn't exist)
        """
        return len(self.metrics.get(metric_name, []))

    def get_total_metrics_count(self) -> int:
        """
        Get the total number of recorded values across all metrics.

        Returns:
            Total count of all timestamp values across all registered metrics
        """
        return sum(len(metric_data) for metric_data in self.metrics.values())

    def get_metric_timestamps(self, metric_name: str) -> list[float]:
        """
        Get raw timestamps for a specific metric.

        Args:
            metric_name: The name of the metric

        Returns:
            List of timestamps for each recorded entry
        """
        if metric_name not in self.metrics:
            return []
        return self.metrics[metric_name].copy()

    def get_all_metrics_timestamps(self) -> dict[str, list[float]]:
        """
        Get raw timestamps for all registered metrics.

        Returns:
            Dictionary with metric names as keys and lists of timestamps as values
        """
        return {
            metric_name: self.get_metric_timestamps(metric_name)
            for metric_name in self.registered_metrics
        }

    def get_metric_intervals(self, metric_name: str) -> list[float]:
        """
        Get time intervals between consecutive events for a specific metric.

        Args:
            metric_name: The name of the metric

        Returns:
            List of time intervals in seconds between consecutive events
        """
        if metric_name not in self.metrics or len(self.metrics[metric_name]) < 2:
            return []

        timestamps = self.metrics[metric_name]

        intervals = []
        for i in range(1, len(timestamps)):
            interval = timestamps[i] - timestamps[i - 1]
            intervals.append(interval)

        return intervals
