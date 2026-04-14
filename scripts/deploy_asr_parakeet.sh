#!/usr/bin/env sh

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

export DOCKER_REPO="nvcr.io/nim/nvidia"
export CONTAINER_ID="riva-asr"
export NIM_TAGS_SELECTOR="mode=all"
export NIM_TAG="latest"

docker run -it --rm --name=$CONTAINER_ID \
   --runtime=nvidia \
   --gpus '"device=0"' \
   --shm-size=8GB \
   -e NGC_API_KEY=${NGC_API_KEY} \
   -e NIM_HTTP_API_PORT=8003 \
   -e NIM_GRPC_API_PORT=50053 \
   -p 8003:8003 \
   -p 50053:50053 \
   -e NIM_TAGS_SELECTOR=$NIM_TAGS_SELECTOR \
   -v $(pwd)/volumes/cache:/opt/nim/.cache \
   -v $(pwd)/src/docker_entrypoints/ast:/scripts:ro \
   -v $(pwd)/volumes/models/ast-parakeet/tritonserver:/data/tritonserver:rw \
   -v $(pwd)/volumes/models/ast-parakeet/data:/data/models:rw \
   ${AST_IMAGE:-${DOCKER_REPO}/${CONTAINER_ID}:${NIM_TAG}} \
   # /scripts/initialize.sh
