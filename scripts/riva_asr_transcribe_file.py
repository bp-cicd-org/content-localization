#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Standalone RIVA ASR client matching content localization (RIVA Transactional).
# Uses offline_recognize (Recognize RPC) like S2SRIVATransactionalService.
# Run with:
#   .venv/bin/python scripts/riva_asr_transcribe_file.py --input-file <path>.wav

import argparse
import wave

import riva.client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send WAV file to RIVA ASR (Canary) in transactional (offline) mode, "
        "matching content localization. Use to verify how much audio RIVA processes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to WAV file (16 kHz mono 16-bit PCM recommended).",
    )
    parser.add_argument(
        "--server",
        default="localhost:50051",
        help="RIVA ASR gRPC address (host:port).",
    )
    parser.add_argument(
        "--source-language",
        default="es-US",
        help="Source language code for ASR/translation.",
    )
    parser.add_argument(
        "--target-language",
        default="en-US",
        help="Target language for translation (used in custom_config).",
    )
    return parser.parse_args()


def read_wav_pcm(wav_path: str) -> tuple[bytes, int, int]:
    """Read raw PCM bytes from WAV (no header). Returns (pcm_bytes, sample_rate_hz, nchannels)."""
    with wave.open(wav_path, "rb") as wav:
        nch = wav.getnchannels()
        framerate = wav.getframerate()
        nframes = wav.getnframes()
        total_sec = nframes / float(framerate)
        print(f"WAV: {framerate} Hz, {nch} ch, {nframes} frames ({total_sec:.2f} s)")
        pcm = wav.readframes(nframes)
    return pcm, framerate, nch


def main() -> None:
    args = parse_args()

    auth = riva.client.Auth(uri=args.server)
    asr_service = riva.client.ASRService(auth)

    # Same config as content localization: GRPCRIVATransactionalASTClient._config_ast_transactional
    config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        max_alternatives=1,
        enable_automatic_punctuation=True,
        verbatim_transcripts=False,
        sample_rate_hertz=16000,
        language_code=args.source_language,
    )
    riva.client.add_custom_configuration_to_config(
        config=config,
        custom_configuration=f"target_language:{args.target_language},task:translate",
    )
    print(
        f"Config: language_code={args.source_language}, "
        f"translation -> {args.target_language} (transactional/offline)"
    )

    pcm_bytes, sample_rate, nch = read_wav_pcm(args.input_file)
    bytes_per_sec = sample_rate * nch * 2  # 16-bit
    duration_sec = len(pcm_bytes) / bytes_per_sec
    print(f"Sending {len(pcm_bytes)} PCM bytes ({duration_sec:.2f} s) to RIVA Recognize...")

    response = asr_service.offline_recognize(audio_bytes=pcm_bytes, config=config)

    print("--- Result ---")
    results = getattr(response, "results", None)
    if not results:
        print("No transcript in response.")
        print("--- Done ---")
        return

    transcript_chunks: list[str] = []
    audio_processed_values: list[float] = []
    for idx, result in enumerate(results):
        audio_processed = getattr(result, "audio_processed", None)
        if audio_processed is not None:
            audio_processed_values.append(float(audio_processed))
            print(f"[debug] result[{idx}] audio_processed={float(audio_processed):.2f}s")

        if not result.alternatives:
            continue
        chunk_text = result.alternatives[0].transcript.strip()
        if chunk_text:
            transcript_chunks.append(chunk_text)

    if not transcript_chunks:
        print("No transcript chunks in response.")
        print("--- Done ---")
        return

    full_text = " ".join(transcript_chunks)
    print(full_text)
    print("--- Done ---")
    print(f"Audio processed: {duration_sec:.2f} s")
    if audio_processed_values:
        print(f"RIVA cumulative audio_processed max: {max(audio_processed_values):.2f} s")
        print(
            "[debug] RIVA audio_processed progression: "
            + ", ".join(f"{v:.2f}" for v in audio_processed_values)
        )
    print(f"Transcript chunks merged: {len(transcript_chunks)}")


if __name__ == "__main__":
    main()
