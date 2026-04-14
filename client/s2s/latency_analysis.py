# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Latency analysis functions for S2S client."""

import os

import matplotlib.pyplot as plt


def calculate_per_chunk_latencies(input_ledger: dict, output_ledger: dict) -> list:
    """Calculate the latencies between the input and output chunks in dictionary format.

    The key is the chunk id, the value is the timestamp of the output chunk for the ledgers.
    The latencies are calculated as the difference between the output chunk timestamp
    and the input chunk timestamp as registered by the ledgers.
    Accomodations are made for the case where the input and output
    ledgers are not of the same length.

    Args:
        input_ledger (dict): The ledger of the input chunks.
        output_ledger (dict): The ledger of the output chunks.

    Returns:
        list: A list of latencies for each chunk.
    """
    latencies = []
    if not input_ledger or not output_ledger:
        return latencies

    max_chunk_id = min(max(input_ledger.keys()), max(output_ledger.keys()))
    for chunk_id in range(max_chunk_id + 1):
        if chunk_id in input_ledger and chunk_id in output_ledger:
            input_timestamp = input_ledger[chunk_id]
            output_timestamp = output_ledger[chunk_id]
            latencies.append(output_timestamp - input_timestamp)
    return latencies


def calculate_output_stream_latencies(input_ledger: dict, output_ledger: dict) -> list:
    """Calculate the latencies between the output + 1 frame and output chunks in dictionary format.

    The key is the chunk id, the value is the timestamp of the output chunk for the ledgers.
    The latencies are calculated as the difference between the output chunk timestamps
    of the current and the next chunk of audio.

    For real-time, we want this latency to be less than chunk size.

    Args:
        input_ledger (dict): The ledger of the input chunks.
        output_ledger (dict): The ledger of the output chunks.

    Returns:
        list: A list of latencies for each chunk.
    """
    latencies = []
    if not input_ledger or not output_ledger:
        return latencies

    max_chunk_id = min(max(input_ledger.keys()), max(output_ledger.keys()))
    for chunk_id in range(max_chunk_id):
        if chunk_id in output_ledger and chunk_id + 1 in output_ledger:
            latencies.append(output_ledger[chunk_id + 1] - output_ledger[chunk_id])
    return latencies


def plot_latency(
    output_stream_latencies: list,
    per_chunk_latencies: list,
    chunk_size_secs: float,
    output_path: str,
) -> None:
    """Plot both output stream and per-chunk latencies on dual y-axes.

    Args:
        output_stream_latencies (list): List of output stream latency values.
        per_chunk_latencies (list): List of per-chunk latency values.
        chunk_size_secs (float): Chunk size in seconds for reference line.
        output_path (str): Path to save the plot.
    """
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot output stream latencies on left y-axis
    color1 = "tab:blue"
    ax1.set_xlabel("Chunk Index")
    ax1.set_ylabel("Output Stream Latency (seconds)", color=color1)
    line1 = ax1.plot(
        output_stream_latencies,
        color=color1,
        label="Output Stream Latency",
        linewidth=2,
        alpha=0.8,
    )
    ax1.tick_params(axis="y", labelcolor=color1)

    # Create second y-axis for per-chunk latencies
    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.set_ylabel("Per-Chunk Latency (seconds)", color=color2)
    line2 = ax2.plot(
        per_chunk_latencies, color=color2, label="Per-Chunk Latency", linewidth=2, alpha=0.8
    )
    ax2.tick_params(axis="y", labelcolor=color2)

    # Add chunk size reference line on both axes
    line3 = ax1.axhline(
        y=chunk_size_secs,
        color="r",
        linestyle="--",
        label="Real-time latency bound",
        alpha=0.7,
    )

    # Combine legends from both axes
    lines = line1 + line2 + [line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.title("Speech-to-Speech Latency Analysis")
    ax1.grid(True, alpha=0.3)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
