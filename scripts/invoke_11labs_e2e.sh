#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Runs ElevenLabs S2S dubbing pipeline.
# Assumes all files are present locally.

mkdir -p outputs

python3 scripts/el_s2s_infer.py \
   --source-language-code en \
   --target-language-code es \
   --input-file inputs/GTC_clip_3.mp4 \
   --output-file outputs/GTC_clip_3_es.wav
