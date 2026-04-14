# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package for the main S2S service.

Here is the flow of the S2S service:

.. code-block:: text

    Client
    |
    | 1. gRPC StreamSpeechToSpeech(request_iterator)
    v
    S2SServiceServicer (from service.py)
    |
    | 2. Extract request_id, wrap iterator
    | 3. Call: service.infer(request_iterator, context, request_id)
    v
    S2SService (abstract, implemented by S2SRIVATransactionalService,
    S2SRIVAStreamingService, or ELDubbingService)
    |
    |--[RIVA Path: threaded pipeline]------------------------------|
    |                                                             |
    | 4a. S2SRIVA*Service.infer()                                  |
    |    |                                                        |
    |    | 5a. self._check_riva_health()                           |
    |    | 6a. self._s2s_impl()                                    |
    |    |    |                                                    |
    |    |    | 7a. Extract config from first request              |
    |    |    |    (source_language, target_language, voice_name)  |
    |    |    |                                                    |
    |    |    |-- AST thread --------> Riva ASR gRPC -> text buffer |
    |    |    |-- Segmentizer thread -> segment buffer             |
    |    |    |-- TTS thread --------> Riva TTS gRPC -> audio buffer|
    |    |    `-- Main thread: yield SpeechToSpeechResponse(...)   |
    |    |                                                        |
    |<-------------------------------------------------------------|
    |                                                             |
    |--[ElevenLabs Path]------------------------------------------|
    |                                                             |
    | 4b. ELDubbingService.infer()                                |
    |    |                                                        |
    |    | 5b. Extract config from first request                   |
    |    |    (source_language, target_language)                   |
    |    |    v                                                   |
    |    | 6b. self.download_input_audio()                         |
    |    |    (collects all audio, writes temp WAV)                |
    |    |    v                                                   |
    |    | 7b. self._impl()                                        |
    |    |    |-- background thread: create_dub_from_file(),        |
    |    |    |   download MP3, enqueue chunks                     |
    |    |    `-- main thread: dequeue chunks, send keep-alive     |
    |    |    v                                                   |
    |    | 8b. yield SpeechToSpeechResponse(audio_data, ...)        |
    |    |<--------------------------------------------------------|
    |<-------------------------------------------------------------|
    |
    v
    Client receives streaming SpeechToSpeechResponse messages
"""

__version__ = "0.1.0"

# Modules available for import:
# - service: Abstract base class for S2S services
# - entrypoint: Main entry point for the S2S service
# - el_utils: ElevenLabs utilities and dubbing service
# - riva_utils: RIVA utilities (ASR, TTS, S2S)
# - segmentizer: Text segmentation utilities
