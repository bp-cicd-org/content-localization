# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client for the NVIDIA AI4M Lipsync service.

Thin entry-point that orchestrates configuration, channel setup, and the
request/response pipeline.  Business logic lives in the sibling modules
``encoding``, ``request_generators``, and ``response_writers``.
"""

import sys
import threading

from nvidia.ai4m.lipsync.v1 import lipsync_pb2

from client.context import LocalContext
from client.lipsync.args import argsfactory
from client.lipsync.config import LipSyncConfig
from client.lipsync.request_generators import generate_request_for_inference
from client.lipsync.response_writers import process_response_iter
from client.utils import create_channel_credentials
from common.buffers import Buffer
from common.buffers import RequestIteratorFromBuffer
from common.nims import LipsyncClient
from common.nims import LipsyncServer


def main():
    """Main entry point for the LipSync client.

    Handles:
    1. Argument parsing
    2. Configuration validation
    3. Channel setup (secure/insecure)
    4. Request processing
    """
    # Parse arguments and create configuration
    parser = argsfactory()
    args = parser.parse_args()
    lipsync_config = LipSyncConfig.from_args(args)

    # Validate configuration
    try:
        lipsync_config.validate_lipsync_config()
    except Exception as e:
        print(f"Invalid configuration: {e}")
        return 1

    print(lipsync_config)

    # Set up channel based on SSL mode
    try:
        channel_credentials = None
        if args.ssl_mode != "DISABLED":
            channel_credentials = create_channel_credentials(args)
        host, port = args.target.split(":", 1)
        server = LipsyncServer(
            host=host,
            port=int(port),
            channel_credentials=channel_credentials,
        )
        client = LipsyncClient(server=server)
        output_buffer: Buffer[lipsync_pb2.LipsyncResponse] = Buffer()

        def run_client() -> None:
            print(f"LipSync client running on thread: {threading.current_thread().name}")
            context = LocalContext()
            client(
                request_iterator=generate_request_for_inference(lipsync_config=lipsync_config),
                output_buffer=output_buffer,
                context=context,
                request_id="lipsync-client",
            )

        client_thread = threading.Thread(target=run_client, daemon=True)
        client_thread.start()

        response_iter = RequestIteratorFromBuffer(output_buffer, poll_timeout=0.1)
        process_response_iter(response_iter=response_iter, lipsync_config=lipsync_config)
        client_thread.join()
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
