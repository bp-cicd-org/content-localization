# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot metrics data from CSV files.

This script reads metric CSV files and creates visualizations including:
- Timeline plot showing when events occurred
- Interval histogram showing time between events
- Statistics summary

Usage:
    python plot_metrics.py <csv_file_path>
    python plot_metrics.py <directory_path>  # Plots all CSV files in directory
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_metric_data(file_path: Path) -> tuple[str, pd.DataFrame]:
    """Load metric data from a CSV file.

    Args:
        file_path: Path to the CSV file

    Returns:
        Tuple of (metric_name, dataframe)
    """
    # Get metric name from filename
    metric_name = file_path.stem

    # Load the CSV
    df = pd.read_csv(file_path)

    # Ensure we have the expected columns
    if "timestamp" not in df.columns:
        raise ValueError(f"CSV file must contain 'timestamp' column: {file_path}")

    # Convert datetime column if present
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])

    return metric_name, df


def calculate_intervals(timestamps: np.ndarray) -> np.ndarray:
    """Calculate time intervals between consecutive timestamps.

    Args:
        timestamps: Array of timestamp values

    Returns:
        Array of intervals in seconds
    """
    if len(timestamps) < 2:
        return np.array([])

    return np.diff(timestamps)


def plot_timeline(metric_name: str, df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot timeline of events.

    Args:
        metric_name: Name of the metric
        df: DataFrame with timestamp data
        ax: Matplotlib axes to plot on
    """
    timestamps = df["timestamp"].values

    # Normalize timestamps to start from 0
    timestamps_normalized = timestamps - timestamps[0]

    # Create event markers
    ax.scatter(timestamps_normalized, np.ones(len(timestamps)), alpha=0.6, s=20, label=metric_name)

    ax.set_xlabel("Time (seconds since start)")
    ax.set_ylabel("Events")
    ax.set_title(f"{metric_name} - Event Timeline")
    ax.set_ylim(0.5, 1.5)
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9)


def plot_intervals(metric_name: str, df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot histogram of intervals between events.

    Args:
        metric_name: Name of the metric
        df: DataFrame with timestamp data
        ax: Matplotlib axes to plot on
    """
    timestamps = df["timestamp"].values
    intervals = calculate_intervals(timestamps)

    if len(intervals) == 0:
        ax.text(
            0.5, 0.5, "Not enough data points", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_title(f"{metric_name} - Intervals")
        return

    # Convert to milliseconds for better readability
    intervals_ms = intervals * 1000

    ax.hist(intervals_ms, bins=50, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Interval (milliseconds)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{metric_name} - Interval Distribution")
    ax.grid(True, alpha=0.3)

    # Add statistics as text
    stats_text = (
        f"Mean: {np.mean(intervals_ms):.2f} ms\n"
        f"Median: {np.median(intervals_ms):.2f} ms\n"
        f"Std: {np.std(intervals_ms):.2f} ms\n"
        f"Min: {np.min(intervals_ms):.2f} ms\n"
        f"Max: {np.max(intervals_ms):.2f} ms"
    )
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=9,
    )


def plot_rate(metric_name: str, df: pd.DataFrame, ax: plt.Axes, window_size: float = 1.0) -> None:
    """Plot event rate over time.

    Args:
        metric_name: Name of the metric
        df: DataFrame with timestamp data
        ax: Matplotlib axes to plot on
        window_size: Window size in seconds for rate calculation
    """
    timestamps = df["timestamp"].values

    if len(timestamps) < 2:
        ax.text(
            0.5, 0.5, "Not enough data points", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_title(f"{metric_name} - Event Rate")
        return

    # Normalize timestamps
    timestamps_normalized = timestamps - timestamps[0]

    # Create time bins
    max_time = timestamps_normalized[-1]
    bins = np.arange(0, max_time + window_size, window_size)

    # Count events in each bin
    counts, bin_edges = np.histogram(timestamps_normalized, bins=bins)

    # Calculate rate (events per second)
    rates = counts / window_size

    # Use bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax.plot(bin_centers, rates, marker="o", markersize=3, alpha=0.7)
    ax.set_xlabel("Time (seconds since start)")
    ax.set_ylabel("Event Rate (events/second)")
    ax.set_title(f"{metric_name} - Event Rate Over Time (window={window_size}s)")
    ax.grid(True, alpha=0.3)

    # Add mean rate line
    mean_rate = np.mean(rates)
    ax.axhline(
        y=mean_rate, color="r", linestyle="--", alpha=0.5, label=f"Mean: {mean_rate:.2f} events/s"
    )
    ax.legend(loc="best", framealpha=0.9)


def plot_single_metric(file_path: Path, output_dir: Path = None) -> None:
    """Plot data from a single CSV file.

    Args:
        file_path: Path to the CSV file
        output_dir: Directory to save plots (if None, displays interactively)
    """
    metric_name, df = load_metric_data(file_path)

    print(f"\nProcessing: {metric_name}")
    print(f"  Total events: {len(df)}")

    if len(df) > 0:
        duration = df["timestamp"].max() - df["timestamp"].min()
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Average rate: {len(df) / duration:.2f} events/second")

    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    fig.suptitle(f"Metrics Analysis: {metric_name}", fontsize=16, fontweight="bold")

    # Plot timeline
    plot_timeline(metric_name, df, ax)

    plt.tight_layout()

    # Save or display
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{metric_name}_analysis.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"  Saved plot to: {output_file}")
        plt.close()
    else:
        plt.show()


def plot_multiple_metrics(file_paths: list[Path], output_dir: Path = None) -> None:
    """Plot comparison of multiple metrics aligned by timestamps.

    All metrics are aligned to the first event across all metrics,
    with time displayed in seconds since that first event.

    Args:
        file_paths: List of CSV file paths
        output_dir: Directory to save plots (if None, displays interactively)
    """
    # Load all metrics
    metrics_data = []
    for file_path in file_paths:
        try:
            metric_name, df = load_metric_data(file_path)
            metrics_data.append((metric_name, df))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    if not metrics_data:
        print("No valid metrics data found")
        return

    # Create comparison plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.suptitle("Metrics Comparison (Aligned by Timestamp)", fontsize=16, fontweight="bold")

    # All timelines on same axis
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_data)))

    # Find global time range
    min_timestamp = min(df["timestamp"].min() for _, df in metrics_data if len(df) > 0)
    max_timestamp = max(df["timestamp"].max() for _, df in metrics_data if len(df) > 0)
    time_range = max_timestamp - min_timestamp
    print(f"  Overall time range: {time_range:.2f} seconds")
    print(f"  First event (across all metrics): {min_timestamp:.2f}s")
    print(f"  Last event (across all metrics): {max_timestamp:.2f}s")

    for idx, (metric_name, df) in enumerate(metrics_data):
        timestamps = df["timestamp"].values
        if len(timestamps) > 0:
            # Convert to seconds since first event
            timestamps_normalized = timestamps - min_timestamp

            y_offset = idx + 1
            ax.scatter(
                timestamps_normalized,
                np.ones(len(timestamps)) * y_offset,
                alpha=0.6,
                s=10,
                label=metric_name,
                color=colors[idx],
            )

    ax.set_xlabel("Time (seconds since first event)")
    ax.set_title("All Metrics Timeline (Aligned by Timestamp)")
    # Add some padding (5% on each side)
    padding = time_range * 0.05 if time_range > 0 else 0.5
    ax.set_xlim(-padding, time_range + padding)
    ax.set_ylabel("Metrics")
    ax.set_yticks(range(1, len(metrics_data) + 1))
    ax.set_yticklabels([name for name, _ in metrics_data])
    ax.grid(True, alpha=0.3)
    # Place legend outside plot area to avoid obscuring data
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", framealpha=0.9)

    # Adjust layout to accommodate legend outside plot area
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save or display
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "metrics_comparison.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"\nSaved comparison plot to: {output_file}")
        plt.close()
    else:
        plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Plot metrics data from CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot a single metric file
  python plot_metrics.py s2s_response_tag.csv
  
  # Plot all metrics in a directory (comparison aligns by timestamp)
  python plot_metrics.py outputs/metrics/

  # Save plots to output directory
  python plot_metrics.py outputs/metrics/ --output plots/
  
  # Plot with custom window size for rate calculation
  python plot_metrics.py metrics/ --window-size 0.5
        """,
    )

    parser.add_argument("path", type=str, help="Path to CSV file or directory containing CSV files")

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output directory for plots (default: display interactively)",
    )

    parser.add_argument(
        "-w",
        "--window-size",
        type=float,
        default=1.0,
        help="Window size in seconds for rate calculation (default: 1.0)",
    )

    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="Skip comparison plot when multiple files are found",
    )

    args = parser.parse_args()

    # Parse input path
    input_path = Path(args.path)

    if not input_path.exists():
        print(f"Error: Path does not exist: {input_path}")
        sys.exit(1)

    # Parse output directory
    output_dir = Path(args.output) if args.output else None

    # Find CSV files
    if input_path.is_file():
        if input_path.suffix.lower() != ".csv":
            print(f"Error: File must be a CSV file: {input_path}")
            sys.exit(1)
        csv_files = [input_path]
    else:
        csv_files = sorted(input_path.glob("*.csv"))
        if not csv_files:
            print(f"Error: No CSV files found in directory: {input_path}")
            sys.exit(1)

    print(f"Found {len(csv_files)} CSV file(s) to process")

    # Plot individual metrics
    for csv_file in csv_files:
        try:
            plot_single_metric(csv_file, output_dir)
        except Exception as e:
            print(f"Error plotting {csv_file}: {e}")
            continue

    # Plot comparison if multiple files
    if len(csv_files) > 1 and not args.no_comparison:
        print("\nGenerating comparison plot...")
        try:
            plot_multiple_metrics(csv_files, output_dir)
        except Exception as e:
            print(f"Error creating comparison plot: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
