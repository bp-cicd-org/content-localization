#!/bin/sh

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

MODEL_RMIR_LOC=/data/tritonserver
MODELS_LOC=/data/models
MODEL_RMIR=asr-parakeet-1.1b-unified-ml-ast-multi-streaming-throughput-nemo.rmir
MODEL_KEY=tlt_encode


if [ ! -e $MODEL_RMIR_LOC/$MODEL_RMIR ]; then
    wget -P $MODEL_RMIR_LOC 10.40.136.52:7890/$MODEL_RMIR
fi
if [ ! -e $MODELS_LOC/riva-nemo-parakeet-rnnt-1.1b-en-US-asr-am-streaming ]; then
    riva-deploy $MODEL_RMIR_LOC/$MODEL_RMIR:$MODEL_KEY $MODELS_LOC
fi

if [ "$MODE" != "init" ]; then
    echo "Starting RIVA"
    start-riva
fi
