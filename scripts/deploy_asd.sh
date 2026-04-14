#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

export CONTAINER_ID="asd"
export NIM_TAG="1.0.0"
export ASD_IMAGE_DEFAULT="nvcr.io/nim/nvidia/active-speaker-detection:${NIM_TAG}"
export ASD_NIM_HTTP_API_PORT="${ASD_NIM_HTTP_API_PORT:-8005}"
export ASD_GRPC_API_PORT="${ASD_GRPC_API_PORT:-50055}"
export ASD_MODEL_MOUNT_PATH="${ASD_MODEL_MOUNT_PATH:-$(pwd)/volumes/models/asd}"

mkdir -p "${ASD_MODEL_MOUNT_PATH}"

docker run -it --rm --name="${CONTAINER_ID}" \
   --runtime=nvidia \
   --gpus all \
   --shm-size=4GB \
   -e NGC_API_KEY="${ASD_API_KEY}" \
   -e NIM_HTTP_API_PORT="${ASD_NIM_HTTP_API_PORT}" \
   -e NIM_GRPC_API_PORT="${ASD_GRPC_API_PORT}" \
   -e AI4M_LOG_LEVEL="${ASD_LOG_LEVEL:-INFO}" \
   -e NIM_LOG_LEVEL="${ASD_LOG_LEVEL:-INFO}" \
   -e MAXINE_MAX_CONCURRENCY_PER_GPU=1 \
   -e LOG_LEVEL="${ASD_LOG_LEVEL:-INFO}" \
   -p "${ASD_NIM_HTTP_API_PORT}:${ASD_NIM_HTTP_API_PORT}" \
   -p "${ASD_GRPC_API_PORT}:${ASD_GRPC_API_PORT}" \
   -v "${ASD_MODEL_MOUNT_PATH}:/opt/nim/.cache:rw" \
   "${ASD_IMAGE:-${ASD_IMAGE_DEFAULT}}"
