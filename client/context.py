# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared context for standalone client execution."""


class LocalContext:
    """Minimal gRPC-compatible context for local client execution.

    Provides a ``context.abort()`` method that raises ``RuntimeError``
    instead of performing a real gRPC abort.  Used by all standalone
    client scripts that run outside a gRPC servicer.

    Examples:
        >>> ctx = LocalContext()
        >>> ctx.abort("INTERNAL", "something went wrong")
        Traceback (most recent call last):
            ...
        RuntimeError: Aborted: INTERNAL, something went wrong
    """

    def abort(self, code: object, msg: str) -> None:
        """Abort the current operation by raising a RuntimeError.

        Args:
            code (object): The gRPC status code (informational only).
            msg (str): The error message.

        Raises:
            RuntimeError: Always raised with the provided code and message.

        Examples:
            >>> ctx = LocalContext()
            >>> try:
            ...     ctx.abort("INTERNAL", "test error")
            ... except RuntimeError as e:
            ...     print(e)
            Aborted: INTERNAL, test error
        """
        raise RuntimeError(f"Aborted: {code}, {msg}")
