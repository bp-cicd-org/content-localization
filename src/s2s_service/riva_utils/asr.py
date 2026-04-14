# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ASR-related abstractions."""

import os
import time
import traceback
import wave
from abc import ABC
from collections.abc import Callable
from collections.abc import Iterator

import grpc
import numpy as np
import riva.client
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechRequest
from scipy import signal

from base_utils import logger
from common.buffers import Buffer
from common.clients import Client
from s2s_service.riva_utils.servers import RivaASRServer
from s2s_service.service import download_input_audio_file


def resample_wav_to_target_rate(
    input_path: str,
    target_sample_rate: int,
) -> str:
    """Resample a WAV file to a target sample rate using scipy.

    Assumes WAV header is already valid (n_frames is correct).
    RIVA ASR expects the audio to be at 16kHz.

    Args:
        input_path: Path to input WAV file.
        target_sample_rate: Target sample rate in Hz.

    Returns:
        str: Path to the resampled file (overwrites input_path).

    Raises:
        RuntimeError: If resampling fails.
    """
    # Check current sample rate
    with wave.open(input_path, "rb") as wav_check:
        original_sample_rate = wav_check.getframerate()
        channels = wav_check.getnchannels()
        sample_width = wav_check.getsampwidth()

    # Only resample if sample rates don't match
    if original_sample_rate != target_sample_rate:
        logger.debug(
            f"Resampling from {original_sample_rate}Hz to {target_sample_rate}Hz using scipy"
        )

        try:
            # Read the audio data
            with wave.open(input_path, "rb") as wav_in:
                pcm_data = wav_in.readframes(wav_in.getnframes())

            # Convert bytes to numpy array
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)

            # Calculate new number of samples
            num_samples_new = int(len(audio_array) * target_sample_rate / original_sample_rate)

            # Resample using scipy
            resampled = signal.resample(audio_array, num_samples_new)
            resampled_pcm = resampled.astype(np.int16).tobytes()

            # Write resampled audio back to the same file

            with wave.open(input_path, "wb") as wav_out:
                wav_out.setnchannels(channels)
                wav_out.setsampwidth(sample_width)
                wav_out.setframerate(target_sample_rate)
                wav_out.writeframes(resampled_pcm)

            logger.info(f"Resampled audio to {target_sample_rate}Hz ({num_samples_new} samples)")

        except Exception as e:
            logger.error(f"Resampling failed: {e}")
            raise RuntimeError(f"Failed to resample audio: {e}")
    else:
        logger.debug(
            f"Audio already at target sample rate {target_sample_rate}Hz, no resampling needed"
        )
    return input_path


class GRPCRIVAASTClient(Client[SpeechToSpeechRequest, str], ABC):
    """Abstract RIVA ASR client that streams transcripts into a buffer."""

    def __init__(
        self,
        server: RivaASRServer,
        sample_rate_hz: int = 16000,
        nchannels: int = 1,
    ) -> None:
        """Constructor for the GRPCASTClient.

        Note:
            - The ASR service returns the audio length in seconds, but for this calculation,
                we need the audio to be 16 bit.
            - Supports multichannel audio via nchannels argument.

        Args:
            server (RivaASRServer): A RIVA ASR server adapter.
            sample_rate_hz (int): The sample rate in Hz.
            nchannels (int): Number of audio channels (default 1).

        """
        super().__init__(server=server)
        self.sample_rate_hz = sample_rate_hz
        self.nchannels = nchannels

        if self.server.stub is None:
            self.server.create_server()
        self._ast_service = self.server.stub

    @classmethod
    def from_string(cls, url: str) -> "GRPCRIVAASTClient":
        """Create a GRPCRIVAASTClient from a string in the form of host:port.

        Args:
            url (str): The URL to create the GRPCASTClient from.

        Returns:
            GRPCRIVAASTClient: A GRPCRIVAASTClient instance.
        """
        return cls(RivaASRServer.from_string(url=url))

    def _bytes_to_seconds(self, num_bytes: int) -> float:
        """Convert number of bytes of 16-bit PCM audio to seconds using
        the sample rate and number of channels.

        Args:
            num_bytes (int): Number of audio bytes.

        Returns:
            float: Duration in seconds.
        """
        bytes_per_sample = 2  # 16-bit PCM = 2 bytes per sample
        num_samples = num_bytes / (bytes_per_sample * self.nchannels)
        return num_samples / self.sample_rate_hz


class GRPCRIVAStreamingASTClient(GRPCRIVAASTClient):
    """RIVA Streaming ASR client."""

    def _config_ast_stream(
        self,
        context: grpc.ServicerContext,
        request_id: str,
        source_language: str = "en-US",
        target_language: str = "es-US",
    ) -> None:
        """Configure the ASR stream.

        Class is called per-inference to configure the ASR streaming request.

        Args:
            context (grpc.ServicerContext): The context of the ASR stream.
            request_id (str): The request ID of the ASR stream.
            source_language (str): The source language. Defaults to "en-US".
            target_language (str): The target language. Defaults to "es-US".

        Raises:
            grpc.RpcError: If the ASR stream configuration fails.
        """
        # Initialize ASR stream configuration
        # TODO: Make this configurable through RPC parameters
        logger.debug(f"Configuring ASR stream for request {request_id}")
        try:
            _streaming_config = riva.client.StreamingRecognitionConfig(
                config=riva.client.RecognitionConfig(
                    encoding=riva.client.AudioEncoding.LINEAR_PCM,
                    max_alternatives=1,
                    enable_automatic_punctuation=True,
                    verbatim_transcripts=False,
                    sample_rate_hertz=self.sample_rate_hz,
                    language_code=source_language,
                ),
                interim_results=True,
            )
            riva.client.add_custom_configuration_to_config(
                config=_streaming_config,
                custom_configuration=f"target_language:{target_language},task:translate",
            )
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Exception in ast during configuration\n{tb}")
            context.abort(grpc.StatusCode.INTERNAL, f"{type(e).__name__}: {e}\n{tb}")
        return _streaming_config

    def _impl(
        self,
        request_iterator: Iterator[SpeechToSpeechRequest],
        output_buffer: Buffer[str],
        context: grpc.ServicerContext,
        request_id: str,
        update_audio_length_callback: Callable | None = None,
        source_language: str = "en-US",
        target_language: str = "es-US",
    ) -> None:
        """Stream ASR transcripts into the provided buffer.

        Args:
            request_iterator: Incoming audio requests.
            output_buffer: Destination buffer for transcript strings.
            context: gRPC servicer context.
            request_id: Correlation identifier.
            update_audio_length_callback: Callback to report processed audio length.
            source_language: Source language code.
            target_language: Target language code.

        Returns:
            None. Transcripts are written to ``output_buffer``.
        """
        logger.debug(f"Running ASR for request {request_id}")

        streaming_config = self._config_ast_stream(
            context=context,
            request_id=request_id,
            source_language=source_language,
            target_language=target_language,
        )

        # Generate audio bytes as a generator. This is needed since RIVA ASR
        # expects a generator of audio bytes.
        audio_bytes_length = 0

        def audio_bytes_generator() -> Iterator[bytes]:
            """Generator for audio bytes to track length of audio chunks.

            Returns:
                Iterator[bytes]: A generator of audio bytes.
            """
            try:
                chunk_count = 0
                logger.debug(f"Starting audio_bytes_generator for request {request_id}")
                logger.debug(f"Request iterator type: {type(request_iterator)}")

                for chunk in request_iterator:
                    chunk_count += 1
                    nonlocal audio_bytes_length

                    # Debug logging to see what we're getting
                    logger.debug(f"Processing chunk {chunk_count}, chunk type: {type(chunk)}")
                    if hasattr(chunk, "audio_data"):
                        logger.debug(
                            f"Chunk {chunk_count} has audio_data: "
                            f"{len(chunk.audio_data) if chunk.audio_data else 0} bytes"
                        )
                    else:
                        logger.debug(f"Chunk {chunk_count} has no audio_data field")

                    # len() will return the number of bytes in the audio chunk.
                    audio_bytes_length += len(chunk.audio_data)
                    logger.info(
                        f"ASR processing chunk {chunk_count} - audio_data length: "
                        f"{len(chunk.audio_data)}"
                    )
                    yield chunk.audio_data

                logger.debug(f"Finished audio_bytes_generator after {chunk_count} chunks")
            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"Exception in audio_bytes_generator: {e}")
                logger.error(f"Exception type: {type(e)}")
                logger.error(f"Exception traceback: {tb}")
                context.abort(grpc.StatusCode.INTERNAL, f"{type(e).__name__}: {e}\n{tb}")
                return

        # Stream to Riva ASR
        try:
            # TODO: Pass the request_id to the streaming response generator for tracking.
            response_generator = self._ast_service.streaming_response_generator(
                audio_chunks=audio_bytes_generator(),
                streaming_config=streaming_config,
            )
            _start_time_for_asr = time.time()
            for asr_response in response_generator:
                # Sometimes RIVA doesn't return any results, so we need to try again.
                if not asr_response.results:
                    continue

                # Select a final result that actually contains alternatives.
                final_result = next(
                    (
                        result
                        for result in asr_response.results
                        if result.is_final and result.alternatives
                    ),
                    None,
                )
                if final_result is None:
                    continue

                final_transcript = final_result.alternatives[0].transcript
                final_audio_processed = getattr(final_result, "audio_processed", None)
                audio_length_seconds = self._bytes_to_seconds(num_bytes=audio_bytes_length)
                # Reset the byte counter after each completed final transcript.
                audio_bytes_length = 0
                if update_audio_length_callback is not None:
                    update_audio_length_callback(audio_length_seconds)
                logger.debug(
                    f"Yielding transcript chunk: {final_transcript}, audio length of this "
                    f"chunk: {audio_length_seconds:.2f} seconds. Processing time: "
                    f"{time.time() - _start_time_for_asr:.2f} seconds."
                )
                if final_audio_processed is not None:
                    logger.debug(
                        "RIVA streaming cumulative audio_processed for request "
                        f"{request_id}: {float(final_audio_processed):.2f} seconds"
                    )
                _start_time_for_asr = time.time()
                output_buffer.put(final_transcript)
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Exception in _ast during streaming response generator\n{tb}")
            context.abort(grpc.StatusCode.INTERNAL, f"{type(e).__name__}: {e}\n{tb}")
            return


class GRPCRIVATransactionalASTClient(GRPCRIVAASTClient):
    """RIVA Transactional ASR client."""

    def _config_ast_transactional(
        self,
        context: grpc.ServicerContext,
        request_id: str,
        source_language: str = "en-US",
        target_language: str = "es-US",
    ) -> None:
        """Configure the ASR stream.

        Class is called per-inference to configure the ASR streaming request.

        Args:
            context (grpc.ServicerContext): The context of the ASR stream.
            request_id (str): The request ID of the ASR stream.
            source_language (str): The source language. Defaults to "en-US".
            target_language (str): The target language. Defaults to "es-US".

        Raises:
            grpc.RpcError: If the ASR stream configuration fails.
        """
        # Initialize ASR stream configuration
        # TODO: Make this configurable through RPC parameters
        logger.debug(f"Configuring ASR stream for request {request_id}")
        try:
            # Note: This is the same as the streaming config's internal config.
            # We don't wrap it with a riva.client.StreamingRecognitionConfig.
            # Creating a duplicate version here, to keep this customizable.
            _transactional_config = riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                max_alternatives=1,
                enable_automatic_punctuation=True,
                verbatim_transcripts=False,
                sample_rate_hertz=self.sample_rate_hz,
                language_code=source_language,
            )
            riva.client.add_custom_configuration_to_config(
                config=_transactional_config,
                custom_configuration=f"target_language:{target_language},task:translate",
            )
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Exception in ast during configuration\n{tb}")
            context.abort(grpc.StatusCode.INTERNAL, f"{type(e).__name__}: {e}\n{tb}")
        return _transactional_config

    def download_input_audio(
        self,
        request_iterator: Iterator[SpeechToSpeechRequest],
        context: grpc.ServicerContext,
        request_id: str,
    ) -> str:
        """Receives audio data from the gRPC stream and writes it to a temp file in /tmp.

        This method is duplicated in the EL Dubbing service, which is another transactional
        service. This could be a double download if we are running transactional RIVA.

        Returns the path to the temp file.

        Args:
            request_iterator (Iterator[SpeechToSpeechRequest]): The request iterator.
            context (grpc.ServicerContext): The gRPC context.
            request_id (str): The request ID.

        Returns:
            str: The path to the temp file.
        """
        try:
            input_path = download_input_audio_file(
                request_iterator=request_iterator,
                context=context,
                request_id=request_id,
            )

            # Resample to target sample rate (16kHz for Canary)
            input_path = resample_wav_to_target_rate(
                input_path=input_path,
                target_sample_rate=self.sample_rate_hz,
            )

        except Exception as e:
            if "input_path" in locals() and os.path.exists(input_path):
                os.remove(input_path)
            tb = traceback.format_exc()
            logger.error(f"Error collecting audio data in request id {request_id}: {e}\n{tb}")
            context.abort(grpc.StatusCode.INTERNAL, f"Collecting audio data failed: {e}\n{tb}")
        logger.debug(
            f"Input file streamed in for request id {request_id}: {input_path} of "
            f" size Input file size: {os.path.getsize(input_path)} bytes"
        )
        return input_path

    def _impl(
        self,
        request_iterator: Iterator[SpeechToSpeechRequest],
        output_buffer: Buffer[str],
        context: grpc.ServicerContext,
        request_id: str,
        update_audio_length_callback: Callable | None = None,
        source_language: str = "en-US",
        target_language: str = "es-US",
    ) -> None:
        """Run transactional ASR and write transcript chunks into the buffer.

        Args:
            request_iterator: Incoming audio requests.
            output_buffer: Destination buffer for transcript strings.
            context: gRPC servicer context.
            request_id: Correlation identifier.
            update_audio_length_callback: Callback to report processed audio length.
            source_language: Source language code.
            target_language: Target language code.

        Returns:
            None. Transcript chunks are written to ``output_buffer``.
        """
        logger.debug(f"Running ASR for request {request_id}")

        transactional_config = self._config_ast_transactional(
            context=context,
            request_id=request_id,
            source_language=source_language,
            target_language=target_language,
        )

        # Download input audio file.
        try:
            input_path = self.download_input_audio(
                request_iterator=request_iterator, context=context, request_id=request_id
            )
            logger.debug(f"Downloaded Input audio file path: {input_path}")
            logger.debug(f"Downloaded Input audio file size: {os.path.getsize(input_path)} bytes")
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Error in streaming inputs: {request_id}: {e}\n{tb}")
            context.abort(grpc.StatusCode.INTERNAL, f"Error in streaming inputs: {e}\n{tb}")

        if not os.path.exists(input_path):
            logger.error(f"Error in streaming inputs: {request_id}: {input_path}.")
            context.abort(grpc.StatusCode.INTERNAL, f"Error in streaming inputs: {input_path}.")

        # Call RIVA Transactionally
        try:
            with open(input_path, "rb") as fh:
                data = fh.read()
            logger.debug(
                f"Input audio file size for request {request_id}: "
                f"{os.path.getsize(input_path)} bytes"
            )

            response = self._ast_service.offline_recognize(
                audio_bytes=data,
                config=transactional_config,
            )

        except Exception as e:
            os.remove(input_path)
            tb = traceback.format_exc()
            logger.error(f"Exception in _ast during streaming response generator\n{tb}")
            context.abort(grpc.StatusCode.INTERNAL, f"{type(e).__name__}: {e}\n{tb}")

        # Emit transactional results chunk-by-chunk to match streaming-mode behavior.
        if not response.results:
            os.remove(input_path)
            context.abort(grpc.StatusCode.INTERNAL, "No transcription results from ASR service")

        emitted_transcript_chunks = 0
        has_audio_processed = False
        previous_audio_processed = 0.0
        for result in response.results:
            current_audio_processed: float | None = None
            audio_processed = getattr(result, "audio_processed", None)
            if audio_processed is not None:
                current_audio_processed = float(audio_processed)
                has_audio_processed = True
                logger.debug(
                    "Cumulative audio processed: %.2f seconds for request %s",
                    current_audio_processed,
                    request_id,
                )
            if not result.alternatives:
                continue
            chunk_text = result.alternatives[0].transcript.strip()
            if not chunk_text:
                continue
            emitted_transcript_chunks += 1
            logger.debug(
                "Yielding transactional transcript chunk %s for request %s: %s",
                emitted_transcript_chunks - 1,
                request_id,
                chunk_text,
            )
            if update_audio_length_callback is not None and current_audio_processed is not None:
                # RIVA exposes cumulative chunk durations; convert to per-chunk deltas.
                audio_length_seconds = max(current_audio_processed - previous_audio_processed, 0.0)
                previous_audio_processed = current_audio_processed
                update_audio_length_callback(audio_length_seconds)
            output_buffer.put(chunk_text)
        if emitted_transcript_chunks == 0:
            os.remove(input_path)
            context.abort(
                grpc.StatusCode.INTERNAL,
                "No transcription results from ASR service",
            )

        # Clean up temp file after successful processing
        os.remove(input_path)
        logger.debug(f"Cleaned up temp file: {input_path}")
        logger.debug(
            f"Transactional ASR emitted {emitted_transcript_chunks} transcript chunks for "
            f"request {request_id}"
        )

        if update_audio_length_callback is not None and not has_audio_processed:
            update_audio_length_callback(self._bytes_to_seconds(num_bytes=len(data)))
