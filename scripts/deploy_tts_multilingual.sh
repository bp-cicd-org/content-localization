#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

export CONTAINER_ID=magpie-tts-multilingual
export NIM_TAGS_SELECTOR="name=${CONTAINER_ID}, batch_size=8, model_type=rmir"
export NIM_TAG=latest

docker run -it --rm --name=$CONTAINER_ID \
   --runtime=nvidia \
   --gpus '"device=0"' \
   --shm-size=8GB \
   -e NGC_API_KEY=${NGC_API_KEY} \
   -e NIM_HTTP_API_PORT=9003 \
   -e NIM_GRPC_API_PORT=50053 \
   -e NIM_DISABLE_MODEL_DOWNLOAD=false \
   -p 9003:9003 \
   -p 50053:50053 \
   -e NIM_TAGS_SELECTOR \
   -v $(pwd)/volumes/models/tts-multilingual:/opt/nim/.cache:rw \
   -v $(pwd)/volumes/models/tts-multilingual/inner_cache:/data/models:rw \
   -v $(pwd)/volumes/models/tts-multilingual/nim_export:/opt/nim/export:rw \
   ${TTS_IMAGE:-nvcr.io/nim/nvidia/${CONTAINER_ID}:${NIM_TAG}}