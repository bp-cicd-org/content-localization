# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch processing: run the full pipeline on every video in a directory."""

import os
import time
from pathlib import Path

from client.asd.diarization import load_diarization_info
from client.batch_processing.args import argsfactory
from client.batch_processing.diarization import ensure_diarization
from client.batch_processing.preprocessing import preprocess_video
from client.batch_processing.report import BatchResult
from client.batch_processing.report import print_report
from client.batch_processing.report import save_report
from client.batch_processing.runner import run_single_video
from client.controller.config import ControllerConfig
from client.utils import check_service_health

VIDEO_EXTENSIONS = {".mp4"}


def discover_videos(input_dir: str) -> list[str]:
    """Find all video files in a directory, sorted by name.

    Args:
        input_dir (str): Directory to scan for video files.

    Returns:
        list[str]: Sorted list of absolute video file paths.

    Raises:
        FileNotFoundError: If *input_dir* does not exist.

    Examples:
        >>> videos = discover_videos("videos/")
        >>> len(videos) > 0
        True
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    return [
        os.path.join(input_dir, f)
        for f in sorted(os.listdir(input_dir))
        if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
    ]


def _diarization_format_for_service(s2s_service: str) -> str:
    """Return the diarization format for the given S2S service.

    Args:
        s2s_service (str): S2S service identifier
            (e.g. ``"CAMB_DUBBING"``).

    Returns:
        str: Diarization format name compatible with
            ``load_diarization_info``.

    Examples:
        >>> _diarization_format_for_service("CAMB_DUBBING")
        'camb'
        >>> _diarization_format_for_service("EL_DUBBING")
        'elevenlabs'
    """
    if s2s_service == "CAMB_DUBBING":
        return "camb"
    return "elevenlabs"


def _process_single_video(
    video_path: str,
    output_dir: str,
    target_language: str,
    config: ControllerConfig,
    s2s_service: str = "EL_DUBBING",
) -> BatchResult:
    """Preprocess and run a single video, returning its result.

    Args:
        video_path (str): Path to the input video.
        output_dir (str): Base output directory.
        target_language (str): Target language code for naming.
        config (ControllerConfig): Pipeline configuration bundle.
        s2s_service (str): S2S backend identifier. Routes
            diarization to Camb AI when ``"CAMB_DUBBING"``.

    Returns:
        BatchResult: Timing and status for this video.

    Examples:
        >>> result = _process_single_video(
        ...     video_path="v.mp4",
        ...     output_dir="out/",
        ...     target_language="de",
        ...     config=cfg,
        ... )  # doctest: +SKIP
    """
    video_name = os.path.basename(video_path)
    stem = Path(video_path).stem
    total_start = time.time()

    try:
        # -- Preprocess --
        preprocess_start = time.time()
        wav_path, duration = preprocess_video(
            video_path=video_path,
            output_dir=output_dir,
        )
        preprocess_time = time.time() - preprocess_start
        print(
            f"[{video_name}] Preprocessed: "
            f"duration={duration:.1f}s, "
            f"preprocess={preprocess_time:.1f}s"
        )

        # -- Diarization --
        diarization_dir = os.path.join(output_dir, "diarization")
        diarization_path = ensure_diarization(
            audio_path=wav_path,
            diarization_dir=diarization_dir,
            video_stem=stem,
            s2s_service=s2s_service,
        )
        diarization_format = _diarization_format_for_service(s2s_service)
        diarization_info = load_diarization_info(
            diarization_file=diarization_path,
            diarization_format=diarization_format,
        )
        if diarization_info:
            print(f"[{video_name}] Diarization: {len(diarization_info.segments)} segments")

        # -- Pipeline --
        output_mp4 = os.path.join(output_dir, f"{stem}_{target_language}.mp4")
        pipeline_start = time.time()
        run_single_video(
            audio_path=wav_path,
            video_path=video_path,
            output_path=output_mp4,
            config=config,
            diarization_info=diarization_info,
        )
        pipeline_time = time.time() - pipeline_start

        output_size = os.path.getsize(output_mp4)
        total_time = time.time() - total_start
        print(f"[{video_name}] Done: pipeline={pipeline_time:.1f}s, total={total_time:.1f}s")

        return BatchResult(
            video_name=video_name,
            video_duration_secs=duration,
            preprocess_time_secs=preprocess_time,
            pipeline_time_secs=pipeline_time,
            total_time_secs=total_time,
            output_path=output_mp4,
            output_size_bytes=output_size,
            success=True,
        )

    except Exception as exc:
        total_time = time.time() - total_start
        print(f"[{video_name}] FAILED: {exc}")
        return BatchResult(
            video_name=video_name,
            video_duration_secs=0.0,
            preprocess_time_secs=0.0,
            pipeline_time_secs=0.0,
            total_time_secs=total_time,
            output_path="",
            output_size_bytes=0,
            success=False,
            error_message=str(exc),
        )


def main() -> None:
    """Run the batch processing pipeline.

    Discovers videos in ``--input-dir``, preprocesses each one
    (audio extraction), runs it through the controller pipeline,
    and produces a batch processing report.

    Examples:
        >>> main()  # doctest: +SKIP
    """
    args = argsfactory().parse_args()

    # Discover videos
    videos = discover_videos(args.input_dir)
    if not videos:
        print(f"No video files found in {args.input_dir}")
        return
    print(f"Found {len(videos)} video(s) in {args.input_dir}")

    # Check controller health
    check_service_health(server=args.controller_server)
    print("Controller service is healthy")

    # Build pipeline config
    pipeline_config = ControllerConfig.from_args(args)

    os.makedirs(args.output_dir, exist_ok=True)

    # Process each video
    results: list[BatchResult] = []
    for idx, video_path in enumerate(videos, start=1):
        print(f"\n{'=' * 72}\n[{idx}/{len(videos)}] {os.path.basename(video_path)}\n{'=' * 72}")
        result = _process_single_video(
            video_path=video_path,
            output_dir=args.output_dir,
            target_language=args.target_language,
            config=pipeline_config,
            s2s_service=args.s2s_service,
        )
        results.append(result)

    # Report
    print_report(results)
    report_path = os.path.join(args.output_dir, "batch_processing_report.json")
    save_report(results=results, output_path=report_path)


if __name__ == "__main__":
    main()
