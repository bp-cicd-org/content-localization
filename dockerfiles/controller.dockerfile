# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Security fixes: try exact versions from guidance, else install latest available
RUN set -eux; \
    apt-get update; \
    install_version() { pkg="$1"; ver="$2"; if apt-cache madison "$pkg" | awk '{print $3}' | grep -x "$ver" >/dev/null; then apt-get install -y --no-install-recommends --allow-downgrades "$pkg=$ver"; else apt-get install -y --no-install-recommends "$pkg"; fi }; \
    install_version libc-bin 2.36-9+deb12u11; \
    install_version libc-dev-bin 2.36-9+deb12u11; \
    install_version libc6 2.36-9+deb12u11; \
    install_version libgnutls30 3.7.9-2+deb12u5; \
    apt-get install -y --no-install-recommends --allow-downgrades \
    libgssapi-krb5-2=1.20.1-2+deb12u3 \
    libk5crypto3=1.20.1-2+deb12u3 \
    libkrb5-3=1.20.1-2+deb12u3 \
    libkrb5support0=1.20.1-2+deb12u3 \
    || apt-get install -y --no-install-recommends \
    libgssapi-krb5-2 libk5crypto3 libkrb5-3 libkrb5support0; \
    install_version libsqlite3-0 3.40.1-2+deb12u2; \
    install_version libssl3 3.0.19-1~deb12u2; \
    install_version openssl 3.0.19-1~deb12u2; \
    install_version libsystemd0 252.23-1~deb12u1; \
    install_version libudev1 252.23-1~deb12u1; \
    install_version perl 5.36.0-7+deb12u3; \
    install_version perl-base 5.36.0-7+deb12u3; \
    install_version perl-modules-5.36 5.36.0-7+deb12u3; \
    install_version python3-pkg-resources 66.1.1-1+deb12u2; \
    install_version python3-setuptools 66.1.1-1+deb12u2; \
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
WORKDIR /opt/controller

# Install system dependencies
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
COPY pyproject.toml README.md /opt/controller/

# Install dependencies using poetry
RUN poetry config virtualenvs.create false && \
    poetry install --without dev --without gpu --no-root

# Install debugpy for remote debugging
RUN pip install debugpy

# Security: upgrade setuptools to fix CVE GHSA-5rjg-fvgr-3xxf
RUN pip install "setuptools>=78.1.1"



# Copy proto files and generate Python code
COPY protos/ /opt/controller/protos/
RUN cd /opt/controller/protos && chmod +x generate_protos.sh && ./generate_protos.sh

# Copy service code
COPY src/controller_service/ /opt/controller/controller_service/
COPY src/common/ /opt/controller/common/
COPY src/profiler/ /opt/controller/profiler/
COPY src/base_utils/ /opt/controller/base_utils/

# Set Python path to include service code and generated protos
ENV PYTHONPATH=/opt/controller/controller_service:/opt/controller/protos/generated:/opt/controller/common:${PYTHONPATH}

# Copy entrypoint script and make it executable
COPY src/docker_entrypoints/controller/entrypoint.sh /opt/controller/entrypoint.sh
RUN chmod +x /opt/controller/entrypoint.sh


# Cleaning up git and unused packages with known CVEs
RUN set -eux; \
    apt-get purge -y git git-man; \
    apt-get remove -y --allow-remove-essential libjs-underscore || true; \
    apt-get autoremove -y; \
    rm -rf /var/lib/apt/lists/*

# Set the entrypoint
ENTRYPOINT ["/opt/controller/entrypoint.sh"]
