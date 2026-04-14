# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base utilities — re-exports vendored ai4m_base_utils for the project.

The vendored source at ``base_utils/ai4m_base_utils/`` is kept **unmodified** so
upstream updates can be dropped in without conflict.  A ``sys.path`` shim lets
the vendored package's internal imports (``from ai4m_base_utils.config import …``)
resolve as top-level, while consumers import everything through this wrapper::

    from base_utils import logger
    from base_utils import GRPCServiceBase
"""

import sys
from pathlib import Path

# Allow vendored ai4m_base_utils internal imports to resolve as top-level
sys.path.insert(0, str(Path(__file__).parent))

# --- Re-export public API -------------------------------------------------- #

from ai4m_base_utils.auth import Auth  # noqa: E402
from ai4m_base_utils.config import AI4M_DEFAULT_MESSAGE_SIZE  # noqa: E402
from ai4m_base_utils.error_utils import FileSizeError  # noqa: E402
from ai4m_base_utils.error_utils import ServiceConfigurationError  # noqa: E402
from ai4m_base_utils.error_utils import SSLConfigurationError  # noqa: E402
from ai4m_base_utils.file_utils import FileUtils  # noqa: E402
from ai4m_base_utils.grpc_service import GRPCServiceBase  # noqa: E402
from ai4m_base_utils.hooks import BaseHooks  # noqa: E402
from ai4m_base_utils.hooks import CleanupHooks  # noqa: E402
from ai4m_base_utils.hooks import MonitoringHooks  # noqa: E402
from ai4m_base_utils.logger import logger  # noqa: E402

__all__ = [
    "logger",
    "AI4M_DEFAULT_MESSAGE_SIZE",
    "GRPCServiceBase",
    "FileSizeError",
    "ServiceConfigurationError",
    "SSLConfigurationError",
    "Auth",
    "FileUtils",
    "BaseHooks",
    "CleanupHooks",
    "MonitoringHooks",
]
