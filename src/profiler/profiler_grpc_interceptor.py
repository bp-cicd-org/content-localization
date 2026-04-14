# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
gRPC Profiler Interceptor Module

Provides automatic profiling of gRPC service calls.
Intercepts RPC methods and profiles their execution automatically.


"""

from collections.abc import Callable
from typing import Any

import grpc

from profiler.profiler_interface import ProfilerInterface
from profiler.yappi_profiler import YappiProfiler


class ProfileInterceptor(grpc.ServerInterceptor):
    """
    gRPC interceptor that automatically profiles RPC method calls.

    Wraps gRPC service methods to start/stop profiling for each call.
    Excludes health check calls from profiling.

    Attributes:
        profiler (ProfilerInterface): Profiler instance to use (defaults to YappiProfiler)
    """

    def __init__(self, profiler: ProfilerInterface | None = None):
        """
        Initialize the profiler interceptor.

        Args:
            profiler (ProfilerInterface, optional): Profiler to use. Defaults to YappiProfiler.
        """
        if profiler is None:
            profiler = YappiProfiler()
        self.profiler = profiler

    def intercept_service(
        self,
        continuation: Callable[[grpc.HandlerCallDetails], grpc.RpcMethodHandler],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """Intercept gRPC service calls to wrap them with profiling."""
        handler = continuation(handler_call_details)
        behavior, handler_factory = get_rpc_handler(handler)

        def _intercept(request_or_iterator, servicer_context):
            grpc_service_name, grpc_method_name = split_method_call(handler_call_details)

            return run_profiler(
                behavior=behavior,
                request_or_iterator=request_or_iterator,
                servicer_context=servicer_context,
                grpc_service_name=grpc_service_name,
                grpc_method_name=grpc_method_name,
                profiler=self.profiler,
            )

        return handler_factory(
            behavior=_intercept,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )


def split_method_call(handler_call_details: grpc.HandlerCallDetails) -> tuple[str, str]:
    """
    Extract service and method names from gRPC call details.

    Args:
        handler_call_details: gRPC handler call details

    Returns:
        Tuple of (service_name, method_name)
    """
    parts = handler_call_details.method.split("/")
    if len(parts) < 3:
        return "", ""
    grpc_service_name, grpc_method_name = parts[1:3]
    return grpc_service_name, grpc_method_name


def get_rpc_handler(handler: grpc.RpcMethodHandler):
    """
    Get the RPC behavior function and handler factory based on streaming type.

    Args:
        handler: gRPC RPC method handler

    Returns:
        Tuple of (behavior_function, handler_factory)
    """
    if handler is None:
        return None

    if handler.request_streaming and handler.response_streaming:
        behavior_fn = handler.stream_stream
        handler_factory = grpc.stream_stream_rpc_method_handler
    elif handler.request_streaming and not handler.response_streaming:
        behavior_fn = handler.stream_unary
        handler_factory = grpc.stream_unary_rpc_method_handler
    elif not handler.request_streaming and handler.response_streaming:
        behavior_fn = handler.unary_stream
        handler_factory = grpc.unary_stream_rpc_method_handler
    else:
        behavior_fn = handler.unary_unary
        handler_factory = grpc.unary_unary_rpc_method_handler

    return behavior_fn, handler_factory


def run_profiler(
    profiler: ProfilerInterface,
    behavior: Callable,
    request_or_iterator: Any,
    servicer_context: grpc.ServicerContext,
    grpc_service_name: str,
    grpc_method_name: str,
):
    """
    Run the RPC behavior with profiling enabled.

    Args:
        profiler: Profiler instance to use
        behavior: RPC behavior function to execute
        request_or_iterator: Request or request iterator
        servicer_context: gRPC servicer context
        grpc_service_name: Name of the gRPC service
        grpc_method_name: Name of the RPC method

    Returns:
        Response or response iterator from the RPC behavior

    Note:
        Health check calls are excluded from profiling.
    """
    if grpc_service_name == "grpc.health.v1.Health" and grpc_method_name == "Check":
        return behavior(request_or_iterator, servicer_context)
    profiler.start(func_name=f"{grpc_service_name}_{grpc_method_name}")
    try:
        response_or_iterator = behavior(
            request_or_iterator,
            servicer_context,
        )
    finally:
        profiler.stop()
    return response_or_iterator
