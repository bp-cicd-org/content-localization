#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

export CONTAINER_ID=magpie-tts-zeroshot
export NIM_TAG=latest
export LOCAL_NIM_CACHE=$(pwd)/volumes/models/tts-zeroshot
export NIM_TAGS_SELECTOR="name=magpie-tts-zeroshot"

# Create cache directories if they don't exist
mkdir -p ${LOCAL_NIM_CACHE}/nim_export
mkdir -p ${LOCAL_NIM_CACHE}/inner_cache

# Run the NIM container
# Download the models for the first time in the cache directory.
docker run -it --rm --name=$CONTAINER_ID \
   --runtime=nvidia \
   --gpus '"device=0"' \
   --shm-size=8GB \
   -e NGC_API_KEY=${TTS_API_KEY} \
   -e NIM_DISABLE_MODEL_DOWNLOAD=false \
   -e NIM_HTTP_API_PORT=9003 \
   -e NIM_GRPC_API_PORT=50053 \
   -e NIM_TAGS_SELECTOR="${NIM_TAGS_SELECTOR}" \
   -p 9003:9003 \
   -p 50053:50053 \
   -v ${LOCAL_NIM_CACHE}:/opt/nim/.cache:rw \
   -v ${LOCAL_NIM_CACHE}/inner_cache:/data/models:rw \
   -v ${LOCAL_NIM_CACHE}/nim_export:/opt/nim/export:rw \
   ${TTS_IMAGE:-nvcr.io/nim/nvidia/${CONTAINER_ID}:${NIM_TAG}}
