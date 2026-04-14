# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


ARG BASE_IMAGE="python:3.12.13-slim-bookworm"  
FROM ${BASE_IMAGE} AS final

ENV DEBIAN_FRONTEND=noninteractive 

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1 

# suppress warning about installing as root (pip 22+)
ENV PIP_ROOT_USER_ACTION=ignore 

# Keeps Python from buffering stdout and stderr to avoid situations where the application crashes
# without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Security: pin OpenSSL to latest Debian security versions
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends --allow-downgrades \
        libssl3=3.0.19-1~deb12u2 \
        openssl=3.0.19-1~deb12u2; \
    rm -rf /var/lib/apt/lists/*

# Mitigate CVE-2025-59375 by installing libexpat >= 2.7.2 (use 2.7.3)
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends build-essential ca-certificates wget xz-utils; \
    wget -O /tmp/expat.tar.xz https://github.com/libexpat/libexpat/releases/download/R_2_7_3/expat-2.7.3.tar.xz; \
    tar -C /tmp -xf /tmp/expat.tar.xz; \
    cd /tmp/expat-2.7.3; \
    ./configure --prefix=/usr; \
    make -j"$(nproc)"; \
    make install; \
    ldconfig; \
    cd /; \
    rm -rf /tmp/expat-2.7.3 /tmp/expat.tar.xz; \
    apt-get purge -y build-essential; \
    apt-get autoremove -y; \
    rm -rf /var/lib/apt/lists/*

# must install python-is-python3 to avoid conflict with python3 because that image is based on
# ubuntu < 24.04 
RUN apt-get update && apt-get install -y python3 python3-pip python-is-python3 && \
    python3 -m pip install --upgrade pip setuptools

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends equivs; \
    mkdir -p /tmp/expat-equivs; \
    cd /tmp/expat-equivs; \
    equivs-control libexpat1; \
    sed -i 's/^Package: .*/Package: libexpat1/' libexpat1; \
    sed -i 's/^Version: .*/Version: 2.7.3-99/' libexpat1; \
    sed -i 's/^Architecture: .*/Architecture: amd64/' libexpat1; \
    sed -i 's/^Description: .*/Description: Metadata package marking Expat 2.7.3 present (built from source)/' libexpat1; \
    printf '\nProvides: libexpat1 (= 2.7.3-99)\nReplaces: libexpat1\n' >> libexpat1; \
    equivs-build libexpat1; \
    dpkg -i ./libexpat1_2.7.3-99_amd64.deb || apt-get -f install -y; \
    rm -rf /tmp/expat-equivs; \
    apt-get purge -y equivs; \
    apt-get autoremove -y; \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/s2s/

# Install system dependencies
# Installing ffmpeg should be fine, since its OSS.
# Install grpcurl inside the container to run health checks.
RUN set -eux; \
    echo "deb http://deb.debian.org/debian bookworm-backports main" > /etc/apt/sources.list.d/backports.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends wget git git-man -t bookworm-backports; \
    rm -rf /var/lib/apt/lists/*
RUN wget https://github.com/fullstorydev/grpcurl/releases/download/v1.9.1/grpcurl_1.9.1_linux_amd64.deb && \
    dpkg -i grpcurl_1.9.1_linux_amd64.deb

# Install poetry
RUN pip install poetry

# Configure poetry to install packages in system Python
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=0 \
    POETRY_VIRTUALENVS_CREATE=0 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Copy poetry configuration files
COPY pyproject.toml README.md /opt/s2s/

# Install dependencies using poetry
RUN poetry install --without dev --without gpu --no-root

# Copy proto files
COPY protos /opt/s2s/protos
RUN cd /opt/s2s/protos && chmod +x generate_protos.sh && ./generate_protos.sh

# Copy service code
COPY src/s2s_service/ /opt/s2s/s2s_service/
COPY src/common/ /opt/s2s/common/
COPY src/base_utils/ /opt/s2s/base_utils/

# Set Python path to include service code and generated protos
ENV PYTHONPATH=/opt/s2s/s2s_service:/opt/s2s/protos/generated:/opt/s2s/common:${PYTHONPATH}

# Copy entrypoint script and make it executable
COPY src/docker_entrypoints/s2s/entrypoint.sh /opt/s2s/entrypoint.sh
RUN chmod +x /opt/s2s/entrypoint.sh

# Security: build libtiff 4.7.1 from source to fix CVE-2026-4775
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends build-essential cmake ca-certificates wget libjpeg-dev zlib1g-dev; \
    wget -O /tmp/tiff-4.7.1.tar.gz https://download.osgeo.org/libtiff/tiff-4.7.1.tar.gz; \
    tar -C /tmp -xzf /tmp/tiff-4.7.1.tar.gz; \
    cd /tmp/tiff-4.7.1; \
    cmake -B build -DCMAKE_INSTALL_PREFIX=/usr -DBUILD_SHARED_LIBS=ON; \
    cmake --build build -j"$(nproc)"; \
    cmake --install build; \
    ldconfig; \
    cd /; \
    rm -rf /tmp/tiff-4.7.1 /tmp/tiff-4.7.1.tar.gz; \
    apt-get purge -y cmake; \
    apt-get autoremove -y; \
    rm -rf /var/lib/apt/lists/*

# Security: build libde265 1.0.17 from source to fix CVE-2026-33164
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends build-essential cmake ca-certificates wget; \
    wget -O /tmp/libde265-1.0.17.tar.gz https://github.com/strukturag/libde265/releases/download/v1.0.17/libde265-1.0.17.tar.gz; \
    tar -C /tmp -xzf /tmp/libde265-1.0.17.tar.gz; \
    cd /tmp/libde265-1.0.17; \
    cmake -B build -DCMAKE_INSTALL_PREFIX=/usr -DBUILD_SHARED_LIBS=ON; \
    cmake --build build -j"$(nproc)"; \
    cmake --install build; \
    ldconfig; \
    cd /; \
    rm -rf /tmp/libde265-*; \
    apt-get purge -y cmake; \
    apt-get autoremove -y; \
    rm -rf /var/lib/apt/lists/*

# Register source-built libraries in dpkg so scanners see the patched versions
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends equivs; \
    mkdir -p /tmp/equivs-pkgs; \
    cd /tmp/equivs-pkgs; \
    equivs-control libde265-0; \
    sed -i 's/^Package: .*/Package: libde265-0/' libde265-0; \
    sed -i 's/^Version: .*/Version: 1.0.17-99/' libde265-0; \
    sed -i 's/^Architecture: .*/Architecture: amd64/' libde265-0; \
    sed -i 's/^Description: .*/Description: Metadata package marking libde265 1.0.17 present (built from source)/' libde265-0; \
    printf '\nProvides: libde265-0 (= 1.0.17-99)\nReplaces: libde265-0\n' >> libde265-0; \
    equivs-build libde265-0; \
    dpkg -i ./libde265-0_1.0.17-99_amd64.deb || apt-get -f install -y; \
    equivs-control libtiff6; \
    sed -i 's/^Package: .*/Package: libtiff6/' libtiff6; \
    sed -i 's/^Version: .*/Version: 4.7.1-99/' libtiff6; \
    sed -i 's/^Architecture: .*/Architecture: amd64/' libtiff6; \
    sed -i 's/^Description: .*/Description: Metadata package marking libtiff 4.7.1 present (built from source)/' libtiff6; \
    printf '\nProvides: libtiff6 (= 4.7.1-99)\nReplaces: libtiff6\n' >> libtiff6; \
    equivs-build libtiff6; \
    dpkg -i ./libtiff6_4.7.1-99_amd64.deb || apt-get -f install -y; \
    rm -rf /tmp/equivs-pkgs; \
    apt-get purge -y equivs; \
    apt-get autoremove -y; \
    rm -rf /var/lib/apt/lists/*

# Cleaning up git and unused packages with known CVEs
RUN set -eux; \
    apt-get purge -y git git-man; \
    apt-get remove -y libjs-underscore || true; \
    apt-get autoremove -y; \
    rm -rf /var/lib/apt/lists/*

# Run the service
ENTRYPOINT ["/opt/s2s/entrypoint.sh"]
