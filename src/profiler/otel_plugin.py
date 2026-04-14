# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
OpenTelemetry Plugin Module

Provides OpenTelemetry integration for gRPC observability.
Exports metrics to file-based storage for analysis.

Note: This is an optional feature requiring opentelemetry dependencies.
"""

import json
import os
from datetime import datetime

import grpc_observability
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics._internal.export import MetricData
from opentelemetry.sdk.metrics.export import MetricExporter
from opentelemetry.sdk.metrics.export import MetricExportResult
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader


class FileMetricExporter(MetricExporter):
    """
    Custom OpenTelemetry metric exporter that writes metrics to a JSON file.

    Exports metrics in JSON format for offline analysis.

    Attributes:
        file_path (str): Path to output JSON file
        metrics_data (list): Accumulated metrics data
    """

    def __init__(self, file_path: str):
        """
        Initialize the file metric exporter.

        Args:
            file_path (str): Path to output JSON file
        """
        self.file_path = file_path
        self.metrics_data = []

    def export(self, metric_data: MetricData) -> MetricExportResult:
        """
        Export metric data to JSON file.

        Args:
            metric_data: OpenTelemetry metric data to export

        Returns:
            MetricExportResult indicating success or failure
        """
        try:
            # Convert metric data to a serializable format
            metric_info = {
                "timestamp": datetime.now().isoformat(),
                "resource": str(metric_data.resource),
                "scope": str(metric_data.scope),
                "metrics": [],
            }

            for metric in metric_data.metrics:
                metric_dict = {
                    "name": metric.name,
                    "description": metric.description,
                    "unit": metric.unit,
                    "data": [],
                }

                for data_point in metric.data.data_points:
                    point_dict = {
                        "attributes": dict(data_point.attributes),
                        "start_time": str(data_point.start_time),
                        "time": str(data_point.time),
                    }

                    # Add value based on metric type
                    if hasattr(data_point, "value"):
                        point_dict["value"] = data_point.value
                    elif hasattr(data_point, "sum"):
                        point_dict["sum"] = data_point.sum
                    elif hasattr(data_point, "count"):
                        point_dict["count"] = data_point.count

                    metric_dict["data"].append(point_dict)

                metric_info["metrics"].append(metric_dict)

            # Append to file
            with open(self.file_path, "a") as f:
                f.write(json.dumps(metric_info, indent=2) + "\n")

            return MetricExportResult.SUCCESS
        except Exception as e:
            print(f"Failed to export metrics to file: {e}")
            return MetricExportResult.FAILURE

    def shutdown(self) -> None:
        """Shutdown the exporter (no-op for file exporter)."""


class OTelPlugin:
    """
    OpenTelemetry plugin for gRPC observability.

    Registers OpenTelemetry plugin with file-based metric export.
    Metrics are exported to otel_metrics.json in the profiler output directory.

    Attributes:
        otel_plugin: OpenTelemetry plugin instance
    """

    def __init__(self):
        """Initialize the OpenTelemetry plugin."""
        self.otel_plugin = None

    def register(self):
        """
        Register the OpenTelemetry plugin globally.

        Creates file-based metric exporter and registers it with gRPC observability.
        Metrics are exported to otel_metrics.json every 5 seconds.

        Raises:
            Exception: If registration fails
        """
        try:
            # Create output directory for metrics
            output_dir = os.environ.get("CONTROLLER_PROFILER_OUTPUT_DIR", "./")
            os.makedirs(output_dir, exist_ok=True)

            # Create file-based metric exporter
            metrics_file = os.path.join(output_dir, "otel_metrics.json")
            file_exporter = FileMetricExporter(metrics_file)

            # Create metric reader with file exporter
            reader = PeriodicExportingMetricReader(
                exporter=file_exporter, export_interval_millis=5000
            )
            provider = MeterProvider(metric_readers=[reader])

            self.otel_plugin = grpc_observability.OpenTelemetryPlugin(meter_provider=provider)
            self.otel_plugin.register_global()
            print(
                f"OpenTelemetry plugin registered successfully - metrics will be exported to {metrics_file}"
            )
        except Exception as e:
            print(f"Failed to register OpenTelemetry plugin: {e}")
            raise

    def deregister(self):
        """Deregister the OpenTelemetry plugin globally."""
        if self.otel_plugin:
            try:
                self.otel_plugin.deregister_global()
                print("OpenTelemetry plugin deregistered successfully")
            except Exception as e:
                print(f"Failed to deregister OpenTelemetry plugin: {e}")
        self.otel_plugin = None
