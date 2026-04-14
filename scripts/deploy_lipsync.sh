#!/usr/bin/env sh

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

docker run -it --rm \
 --name=lipsync  \
 --runtime=nvidia  \
 --gpus all  \
 --shm-size=8GB  \
 -e NGC_API_KEY=${NGC_API_KEY}  \
 -e AI4M_MAX_CONCURRENCY_PER_GPU=1  \
 -e NIM_HTTP_API_PORT=8000  \
 -e NIM_GRPC_API_PORT=8001  \
 -e NIM_CACHE_DIR=/opt/nim/.cache  \
 -v $(pwd)/volumes/models/lipsync:/opt/nim/.cache  \
 -p 8000:8000  \
 -p 8001:8001  \
${LIP_SYNC_IMAGE:-nvcr.io/nim/nvidia/lipsync:1.2.0}
