# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RIVA-based Speech-to-Speech (S2S) gRPC service using AI4M base utilities."""

import argparse
import os
import tempfile
import threading
import traceback
import wave
from abc import abstractmethod
from collections.abc import Callable
from collections.abc import Iterator
from pathlib import Path

import grpc
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechRequest
from nvidia.ai4m.s2s.v1.s2s_pb2 import SpeechToSpeechResponse

from base_utils import AI4M_DEFAULT_MESSAGE_SIZE
from base_utils import logger
from common.audio_utils import create_wav_header
from common.buffers import Buffer
from common.buffers import RequestIteratorFromBuffer
from s2s_service.riva_utils.asr import GRPCRIVAASTClient
from s2s_service.riva_utils.asr import GRPCRIVAStreamingASTClient
from s2s_service.riva_utils.asr import GRPCRIVATransactionalASTClient
from s2s_service.riva_utils.servers import RivaASRServer
from s2s_service.riva_utils.servers import RivaTTSServer
from s2s_service.riva_utils.tts import GRPCRIVATTSClient

# Import components
from s2s_service.segmentizer import length_segmentizer
from s2s_service.service import S2SService

# Seconds to wait for ASR/segmentizer/TTS sub-pipeline threads during cleanup
S2S_CLEANUP_TIMEOUT: float = float(os.environ.get("S2S_CLEANUP_TIMEOUT", "1.0"))


class S2SRIVAAbstractService(S2SService):
    """Speech-to-Speech service using RIVA ASR and TTS.

    This class implements the main S2S service, handling initialization,
    client connections, and service lifecycle. It manages connections to
    the Riva ASR and TTS services and provides them to the servicer.

    The RIVA path runs a threaded pipeline: ASR, segmentizer, and TTS each
    execute in background threads and communicate through buffers. The main
    thread streams responses from the TTS output buffer.

    .. code-block:: text

        S2SServiceServicer (from service.py)
          |
          | 1. Extract request_id, wrap iterator
          | 2. Call: service.infer(request_iterator, context, request_id)
          v
        S2SService (abstract, implemented by S2SRIVATransactionalService or
        S2SRIVAStreamingService)
          |
          | (RIVA Path)
          |
          | 3. self._check_riva_health()
          | 4. self._s2s_impl()
          |    |
          |    |-- AST thread -------------------------------------------.
          |    |   request_iterator -> ASR gRPC -> text_output_buffer    |
          |    |                                                       |
          |    |-- Segmentizer thread -------------------------------.  |
          |    |   text_output_buffer -> segmentizer -> segment_buffer | |
          |    |                                                     | | |
          |    |-- TTS thread --------------------------------------+-|-+-.
          |    |   segment_buffer -> TTS gRPC -> tts_output_buffer   | |  |
          |    |                                                     | |  |
          |    `-- Main thread: read tts_output_buffer -> yield responses--'
          |
          v

    """

    @abstractmethod
    def ast_service(self) -> GRPCRIVAASTClient:
        """AST service."""

    # Subclasses must implement these lists.
    supported_voice_names = {"en-US": []}
    supported_source_languages = ["en-US"]
    supported_target_languages = ["en-US"]
    use_auto_zero_shot = False

    @property
    def voice_name(self) -> str:
        """Voice name.

        Args:
            value (str): The voice name.

        Returns:
            str: The voice name.
        """
        return self._voice_name

    def validate_voice_name(self, value: str, target_language: str) -> bool:
        """Validate the voice name.

        Args:
            value (str): The voice name.
            target_language (str): The target language to validate against.

        Returns:
            bool: True if the voice name is supported, False otherwise.
        """
        if value is None:
            return True  # None is valid (will use default or zero-shot)
        return value in self.supported_voice_names.get(target_language, [])

    def validate_audio_format(self, value: str) -> bool:
        """Supported audio formats: LINEAR_PCM, WAV.

        Args:
            value (str): The audio format.

        Returns:
            bool: True if the audio format is supported, False otherwise.
        """
        return value in ["WAV"]

    def __init__(
        self,
        ast_server: RivaASRServer,
        tts_server: RivaTTSServer,
        message_size: int = AI4M_DEFAULT_MESSAGE_SIZE,
        sample_rate_hz: int = 16000,
        default_source_language: str = "en-US",
        default_target_language: str = "es-US",
        default_voice_name: str | None = None,
        segmentizer: Callable = length_segmentizer,
        nchannels: int = 1,
        audio_format: str = "WAV",
    ) -> None:
        """Initialize the S2S service.

        Sets up the service with:
        - Hooks for initialization and execution
        - Riva ASR and TTS clients
        - Logging configuration

        Args:
            ast_server (RivaASRServer): Address of the ASR server
            tts_server (RivaTTSServer): Address of the TTS server
            message_size (int): Maximum size of gRPC messages in bytes
            sample_rate_hz (int): Sample rate in Hz
            default_source_language (str): Default source language
            default_target_language (str): Default target language
            default_voice_name (str): Default voice name
            segmentizer (Callable): Segmentizer function
            nchannels (int): Number of audio channels (default 1).
            audio_format (str): Audio format (default WAV).
        """
        super().__init__(
            message_size=message_size,
            sample_rate_hz=sample_rate_hz,
            default_source_language=default_source_language,
            default_target_language=default_target_language,
            nchannels=nchannels,
            audio_format=audio_format,
            supported_source_languages=self.supported_source_languages,
            supported_target_languages=self.supported_target_languages,
        )
        self.segmentizer = segmentizer
        self.default_voice_name = default_voice_name

        self.ast_client = self.ast_service(
            server=ast_server,
            sample_rate_hz=self.sample_rate_hz,
            nchannels=nchannels,
        )
        self.tts_client = GRPCRIVATTSClient(
            server=tts_server,
            sample_rate_hz=self.sample_rate_hz,
        )

        logger.debug(f"Initialized ASR client: {self.ast_client}")
        logger.debug(f"Initialized TTS client: {self.tts_client}")
        logger.debug(f"Default voice name: {self.default_voice_name}")
        logger.debug(f"Default source language: {self.default_source_language}")
        logger.debug(f"Default target language: {self.default_target_language}")
        logger.debug(f"Sample rate: {self.sample_rate_hz}")
        logger.debug(f"Number of channels: {self.nchannels}")
        logger.debug(f"Audio format: {self.audio_format}")
        logger.debug(f"Supported source languages: {self.supported_source_languages}")
        logger.debug(f"Supported target languages: {self.supported_target_languages}")

    def _check_riva_health(self) -> None:
        """Check the health of the Riva services.

        This method checks the health of the Riva services by calling their
        HTTP health endpoints. It verifies both ASR and TTS services are running.

        Raises:
            ConnectionError: If services are not accessible
        """
        self.ast_client.is_healthy()
        self.tts_client.is_healthy()

    def _extract_config_from_request(
        self, config_request: SpeechToSpeechRequest
    ) -> tuple[str, str, str | None]:
        """Extract configuration from the first request.

        Args:
            config_request (SpeechToSpeechRequest): The first request containing config.

        Returns:
            tuple[str, str, str | None]: (source_language, target_language, voice_name)
        """
        voice_name = None

        if config_request.HasField("config"):
            config = config_request.config
            source_language = (
                config.source_language
                if config.HasField("source_language")
                else self.default_source_language
            )
            target_language = (
                config.target_language
                if config.HasField("target_language")
                else self.default_target_language
            )
            voice_name = (
                config.voice_name if config.HasField("voice_name") else self.default_voice_name
            )
        else:
            source_language = self.default_source_language
            target_language = self.default_target_language
            voice_name = self.default_voice_name

        logger.debug(f"Using source language: {source_language}")
        logger.debug(f"Using target language: {target_language}")
        logger.debug(f"Using voice name: {'None' if voice_name is None else voice_name}")

        return source_language, target_language, voice_name

    def _validate_request_config(
        self,
        source_language: str,
        target_language: str,
        voice_name: str | None,
        context: grpc.ServicerContext,
    ) -> None:
        """Validate the request configuration.

        Args:
            source_language (str): The source language.
            target_language (str): The target language.
            voice_name (str | None): The voice name.
            context (grpc.ServicerContext): The gRPC context.

        Raises:
            grpc.RpcError: If validation fails.
        """
        if not self.validate_source_language(source_language):
            logger.error(f"Invalid source language: {source_language}")
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, f"Invalid source language: {source_language}"
            )
        if not self.validate_target_language(target_language):
            logger.error(f"Invalid target language: {target_language}")
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, f"Invalid target language: {target_language}"
            )
        if voice_name is not None and not self.validate_voice_name(voice_name, target_language):
            logger.error(f"Invalid voice name: {voice_name}")
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"Invalid voice name: {voice_name}")

    def _create_audio_iterator_from_file(
        self, audio_file_path: str
    ) -> Iterator[SpeechToSpeechRequest]:
        """Create an iterator that yields SpeechToSpeechRequest from a WAV file.

        Similar to AudioSourceSimulator.simulated_audio_chunk_generator, this creates
        properly formatted SpeechToSpeechRequest objects with audio metadata.

        The first chunk includes a WAV header so that when the iterator is consumed
        by the ASR client and written to a file, it becomes a valid WAV file.

        Args:
            audio_file_path (str): Path to the WAV file.

        Returns:
            Iterator[SpeechToSpeechRequest]: Iterator yielding audio chunks with metadata.

        Raises:
            Exception: If there's an error reading the WAV file.
        """
        try:
            with wave.open(audio_file_path, "rb") as wav:
                sample_rate_from_file = wav.getframerate()
                n_channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                n_frames = wav.getnframes()
                chunk_size = sample_rate_from_file  # 1 second chunks

                # Create WAV header and yield it as the first chunk
                wav_header = create_wav_header(
                    n_channels=n_channels,
                    sample_width=sample_width,
                    frame_rate=sample_rate_from_file,
                    n_frames=n_frames,
                )

                header_request = SpeechToSpeechRequest(
                    audio_data=wav_header,
                    audio_sample_rate=sample_rate_from_file,
                    audio_num_channels=n_channels,
                    audio_format="LINEAR_PCM",
                )
                yield header_request

                # Then yield the audio frames
                while True:
                    frames = wav.readframes(chunk_size)
                    if not frames:
                        break

                    # Create request with metadata similar to AudioSourceSimulator
                    request = SpeechToSpeechRequest(
                        audio_data=frames,
                        audio_sample_rate=sample_rate_from_file,
                        audio_num_channels=n_channels,
                        audio_format="LINEAR_PCM",
                    )
                    yield request

        except Exception as e:
            logger.error(f"Error reading WAV file for iterator: {e}")
            raise

    def setup_zero_shot_audio(
        self,
        request_iterator: Iterator[SpeechToSpeechRequest],
        target_language: str,
        context: grpc.ServicerContext,
        request_id: str,
    ) -> tuple[
        dict | None,
        Iterator[SpeechToSpeechRequest],
        Path | None,
        str | None,
    ]:
        """Setup zero-shot audio reference by downloading and processing input audio.

        This implements the following logic:
        1. Download the entire file and empty the request iterator
        2. Create a duplicate file clipped to first ten seconds (for zero-shot)
        3. Create an iterator from the downloaded file for AST request consumption
        4. Return zero-shot data to be sent to TTS along with AST output

        Args:
            request_iterator (Iterator[SpeechToSpeechRequest]): The audio request iterator.
            target_language (str): The target language for fallback voice selection.
            context (grpc.ServicerContext): The gRPC context.
            request_id (str): The request ID.

        Returns:
            tuple: (zero_shot_data, new_request_iterator, zero_shot_ref_file, full_audio_path)
        """
        zero_shot_ref_file = None
        full_audio_path = None
        zero_shot_data = None

        try:
            # Step 1: Download the entire file and empty the request iterator
            full_audio_path = self.download_input_audio(request_iterator, context, request_id)

            # Step 2: Create a duplicate file clipped to first ten seconds for zero-shot
            zero_shot_ref_file, ref_sample_rate = self.extract_zeroshot_reference_audio(
                source_audio_file=full_audio_path, max_duration=10.0
            )

            # Prepare zero-shot config data with file path
            zero_shot_data = {
                "audio_prompt_file": zero_shot_ref_file,
                "sample_rate_hz": ref_sample_rate,
                "quality": 20,
                "encoding": "LINEAR_PCM",
            }
            logger.debug("Using auto-extracted zero-shot reference")

            # Step 3: Create an iterator from the downloaded file for AST request consumption
            request_iterator = self._create_audio_iterator_from_file(full_audio_path)

        except Exception as e:
            logger.error(f"Failed to extract automatic zero-shot reference: {e}")
            # Fall back to default voice if extraction fails
            voice_name = self.supported_voice_names.get(target_language, [""])[0]
            logger.warning(f"Falling back to voice_name: {voice_name}")
            zero_shot_data = None

        # Step 4: zero_shot_data will be sent to TTS along with AST output iterator
        return zero_shot_data, request_iterator, zero_shot_ref_file, full_audio_path

    def _determine_voice_config(
        self, zero_shot_data: dict | None, voice_name: str | None, target_language: str
    ) -> str | None:
        """Determine the final voice configuration to use.

        Args:
            zero_shot_data (dict | None): Zero-shot data if available.
            voice_name (str | None): Requested voice name.
            target_language (str): Target language.

        Returns:
            str | None: Final voice name to use (None for pure zero-shot).
        """
        if zero_shot_data:
            # Pure zero-shot: voice cloned from audio prompt only (voice_name not needed)
            logger.info("Using pure zero-shot TTS (voice cloned from audio prompt)")
            return None
        else:
            # Regular TTS mode: need voice_name
            if voice_name is None:
                voice_name = self.supported_voice_names[target_language][0]
                logger.info(f"Using default voice name for {target_language}: {voice_name}")
            else:
                logger.info(f"Using voice name: {voice_name}")
            return voice_name

    def _cleanup_temp_files(
        self,
        zero_shot_ref_file: Path | None,
        full_audio_path: str | None,
    ) -> None:
        """Clean up temporary files created during processing.

        Args:
            zero_shot_ref_file (Path | None): Path to zero-shot reference file.
            full_audio_path (str | None): Path to full audio temp file.
        """
        if zero_shot_ref_file and zero_shot_ref_file.exists():
            try:
                zero_shot_ref_file.unlink()
                logger.debug(f"Cleaned up zero-shot reference file: {zero_shot_ref_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up zero-shot reference file: {e}")

        if full_audio_path and Path(full_audio_path).exists():
            try:
                Path(full_audio_path).unlink()
                logger.debug(f"Cleaned up full audio file: {full_audio_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up full audio file: {e}")

    def _bytes_to_seconds(self, num_bytes: int) -> float:
        """Convert number of bytes of 16-bit PCM audio to seconds.

        Using the sample rate and number of channels

        Args:
            num_bytes (int): Number of audio bytes.

        Returns:
            float: Duration in seconds.
        """
        bytes_per_sample = 2  # 16-bit PCM = 2 bytes per sample
        num_samples = num_bytes / (bytes_per_sample * self.nchannels)
        return num_samples / self.sample_rate_hz

    def extract_zeroshot_reference_audio(
        self, source_audio_file: str, max_duration: float = 10.0
    ) -> tuple[Path, int]:
        """Extract first N seconds of audio from WAV file for zero-shot reference.

        Args:
            source_audio_file (str): Path to WAV audio file (written by write_wav_iterator_to_file).
            max_duration (float): Maximum duration to extract in seconds.

        Returns:
            tuple[Path, int]: (path to extracted audio file, sample_rate)
        """
        try:
            # Read source audio file
            with wave.open(source_audio_file, "rb") as wav:
                sample_rate = wav.getframerate()
                nchannels = wav.getnchannels()
                sampwidth = wav.getsampwidth()

                # Calculate frames for desired duration
                max_frames = int(sample_rate * max_duration)
                total_frames = wav.getnframes()
                frames_to_extract = min(max_frames, total_frames)

                # Extract audio frames (raw PCM data)
                audio_frames = wav.readframes(frames_to_extract)

                duration_extracted = frames_to_extract / sample_rate
                logger.info(f"Extracted {duration_extracted:.2f}s of audio for zero-shot reference")

            # Write extracted audio as a proper WAV file for TTS (with headers)
            temp_ref = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="/tmp")
            temp_ref.close()

            with wave.open(temp_ref.name, "wb") as wav_out:
                wav_out.setnchannels(nchannels)
                wav_out.setsampwidth(sampwidth)
                wav_out.setframerate(sample_rate)
                wav_out.writeframes(audio_frames)

            logger.debug(f"Created zero-shot reference WAV: {temp_ref.name}")

            return Path(temp_ref.name), sample_rate

        except Exception as e:
            logger.error(f"Failed to extract reference audio from {source_audio_file}: {e}")
            raise

    def _s2s_impl(
        self,
        request_iterator: SpeechToSpeechRequest,
        context: grpc.ServicerContext,
        request_id: str,
    ) -> Iterator[SpeechToSpeechResponse]:
        """Translate audio to target language.

        Args:
            request_iterator (SpeechToSpeechRequest): The audio to translate.
            context (grpc.ServicerContext): The context of the request.
            request_id (str): The id of the request.

        Returns:
            SpeechToSpeechResponse: The target audio.

        Raises:
            grpc.RpcError: If there's an error in processing the stream.
        """
        logger.info(f"Running ASR for request id {request_id}")

        if self.ast_client is None or self.tts_client is None:
            yield SpeechToSpeechResponse()
            return

        # AST client will produce text in chunks and also produces a length of the input audio
        # in seconds that is processed so far in this invocation.
        # This audio length should be used at a later time for AV sync.
        current_input_audio_length: float = 0.0
        current_output_audio_length: float = 0.0
        length_lock = threading.Lock()

        def update_input_audio_length(length: float) -> None:
            """Callback to track audio length processed by ASR."""
            nonlocal current_input_audio_length
            with length_lock:
                current_input_audio_length += length
                logger.debug(f"Input audio length processed: {current_input_audio_length} seconds.")

        # Get the source and target language from the first request.
        # If not provided, use the default values.
        config_request = next(request_iterator)
        if config_request.audio_data:
            # Preserve audio data when the first packet contains both config and audio.
            original_request_iterator = request_iterator

            def replay_request_iterator() -> Iterator[SpeechToSpeechRequest]:
                yield config_request
                yield from original_request_iterator

            request_iterator = replay_request_iterator()
        source_language, target_language, voice_name = self._extract_config_from_request(
            config_request
        )

        # Validate configuration
        # This is needed cause the client proto sends empty string for voice name if not provided.
        if voice_name == "":
            logger.debug("Voice name is empty string, setting to None")
            voice_name = None
        self._validate_request_config(source_language, target_language, voice_name, context)

        # Handle automatic zero-shot extraction if enabled
        zero_shot_data = None
        zero_shot_ref_file = None
        full_audio_path = None

        if self.use_auto_zero_shot:
            # Override the request iterator using a new one with zero-shot data.
            zero_shot_data, request_iterator, zero_shot_ref_file, full_audio_path = (
                self.setup_zero_shot_audio(request_iterator, target_language, context, request_id)
            )

        # Log final configuration
        logger.info(f"Using source language: {source_language}")
        logger.info(f"Using target language: {target_language}")

        # Determine final voice configuration
        voice_name = self._determine_voice_config(zero_shot_data, voice_name, target_language)

        text_output_buffer: Buffer[str] = Buffer()
        ast_exceptions: list[Exception] = []
        segmentizer_exceptions: list[Exception] = []
        ast_thread: threading.Thread | None = None
        segmentizer_thread: threading.Thread | None = None
        tts_thread: threading.Thread | None = None

        def run_ast() -> None:
            try:
                self.ast_client(
                    request_iterator=request_iterator,
                    output_buffer=text_output_buffer,
                    context=context,
                    request_id=request_id,
                    update_audio_length_callback=update_input_audio_length,
                    source_language=source_language,
                    target_language=target_language,
                )
            except Exception as exc:
                ast_exceptions.append(exc)

        ast_thread = threading.Thread(target=run_ast, daemon=True)
        ast_thread.start()
        text_chunk_generator = RequestIteratorFromBuffer(text_output_buffer)

        # Segmentizer to prepare for TTS.
        # All segmentizers should take an iterator and return an iterator.
        segmentizer_output_buffer: Buffer[str] = Buffer()

        def run_segmentizer() -> None:
            try:
                for segment in self.segmentizer(text_chunk_generator):
                    segmentizer_output_buffer.put(segment)
            except Exception as exc:
                segmentizer_exceptions.append(exc)
            finally:
                segmentizer_output_buffer.done = True

        segmentizer_thread = threading.Thread(target=run_segmentizer, daemon=True)
        segmentizer_thread.start()
        text_chunk_generator_segmented = RequestIteratorFromBuffer(segmentizer_output_buffer)

        # Run TTS inference
        # Step 4: Send zero_shot_data (duplicate trimmed audio) to TTS along with AST output
        tts_output_buffer: Buffer[object] = Buffer()
        tts_exceptions: list[Exception] = []

        def run_tts() -> None:
            try:
                self.tts_client(
                    request_iterator=text_chunk_generator_segmented,
                    output_buffer=tts_output_buffer,
                    context=context,
                    request_id=request_id,
                    language_code=target_language,
                    voice_name=voice_name,
                    zero_shot_data=zero_shot_data,  # Contains the 10-second reference audio
                )
            except Exception as exc:
                tts_exceptions.append(exc)

        tts_thread = threading.Thread(target=run_tts, daemon=True)
        tts_thread.start()

        audio_output_chunk_generator = RequestIteratorFromBuffer(tts_output_buffer)

        # Produce output responses
        try:
            for chunk_count, chunk in enumerate(audio_output_chunk_generator):
                logger.debug(
                    f"Producing output audio chunk {chunk_count} of format: {self.audio_format}"
                )

                _s2s_response_generator = SpeechToSpeechResponse(
                    audio_data=chunk.audio,
                    audio_sample_rate=self.tts_client.sample_rate_hz,
                    audio_num_channels=self.nchannels,
                    audio_format=self.audio_format,
                )

                # Calculate and log audio lengths for AV sync tracking
                current_output_audio_length += self._bytes_to_seconds(num_bytes=len(chunk.audio))
                with length_lock:
                    input_length_snapshot = current_input_audio_length
                logger.debug(
                    f"Input: {input_length_snapshot:.2f}s, "
                    f"Output: {current_output_audio_length:.2f}s, "
                    f"Delta: {current_output_audio_length - input_length_snapshot:.2f}s"
                )

                yield _s2s_response_generator
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Exception in TTS inference during streaming response generator\n{tb}")
            context.abort(grpc.StatusCode.INTERNAL, f"{type(e).__name__}: {e}\n{tb}")
        finally:
            if ast_thread is not None:
                ast_thread.join(timeout=S2S_CLEANUP_TIMEOUT)
            if segmentizer_thread is not None:
                segmentizer_thread.join(timeout=S2S_CLEANUP_TIMEOUT)
            if tts_thread is not None:
                tts_thread.join(timeout=S2S_CLEANUP_TIMEOUT)
            if ast_exceptions:
                logger.error(f"AST thread errors: {[str(e) for e in ast_exceptions]}")
            if segmentizer_exceptions:
                logger.error(
                    f"Segmentizer thread errors: {[str(e) for e in segmentizer_exceptions]}"
                )
            if tts_exceptions:
                logger.error(f"TTS thread errors: {[str(e) for e in tts_exceptions]}")
            # Clean up temp files created for zero-shot
            self._cleanup_temp_files(zero_shot_ref_file, full_audio_path)

    def infer(
        self,
        request_iterator: Iterator[SpeechToSpeechRequest],
        context: grpc.ServicerContext,
        request_id: str,
    ) -> Iterator[SpeechToSpeechResponse]:
        """Infer the S2S service.

        Args:
            request_iterator (Iterator[SpeechToSpeechRequest]): The audio to translate.
            context (grpc.ServicerContext): The context of the request.
            request_id (str): The id of the request.

        Returns:
            Iterator[SpeechToSpeechResponse]: The target audio.

        Raises:
            grpc.RpcError: If there's an error in processing the stream.
        """
        logger.debug(f"Infer request: for request id {request_id}")
        self._check_riva_health()

        response_generator = self._s2s_impl(
            request_iterator=request_iterator, context=context, request_id=request_id
        )
        yield from response_generator

    @staticmethod
    def argsfactory(
        parser: argparse.ArgumentParser | None = None,
    ) -> argparse.ArgumentParser:
        """Parser for command line arguments.

        Args:
            parser (argparse.ArgumentParser | None): Optional existing parser to extend.

        Returns:
            argparse.ArgumentParser: Unparsed command line arguments
        """
        if parser is None:
            parser = argparse.ArgumentParser(description="RIVA Speech-to-Speech Service")
        parser = S2SService.argsfactory(parser=parser)

        # Service dependencies
        parser.add_argument(
            "--ast-server",
            type=str,
            default="localhost:50051",
            help="ASR server address (default: localhost:50051)",
        )
        parser.add_argument(
            "--tts-server",
            type=str,
            default="localhost:50052",
            help="TTS server address (default: localhost:50052)",
        )
        # Health dependencies
        parser.add_argument(
            "--ast-health-server",
            type=str,
            default="localhost:8000",
            help="ASR server address (default: localhost:8000)",
        )
        parser.add_argument(
            "--tts-health-server",
            type=str,
            default="localhost:9003",
            help="TTS server address (default: localhost:9003)",
        )
        parser.add_argument(
            "--default-voice-name",
            type=str,
            default="Magpie-Multilingual.EN-US.Sofia",
            help="Voice name (default: Magpie-Multilingual.EN-US.Sofia)",
        )
        return parser


class S2SRIVATransactionalService(S2SRIVAAbstractService):
    """RIVA Transactional Speech-to-Speech service.

    This class currently serves only RIVA Magpie ZeroShot NIM for TTS.
    """

    supported_source_languages = [
        "en-US",
        "es-ES",
        "ar-AR",
        "es-US",
        "pt-BR",
        "fr-FR",
        "de-DE",
        "it-IT",
        "ja-JP",
        "ko-KR",
        "ru-RU",
        "hi-IN",
    ]
    supported_target_languages = ["en-US"]
    use_auto_zero_shot = True
    supported_voice_names = {
        "en-US": [
            "Magpie-ZeroShot.Female-1",
            "Magpie-ZeroShot.Female-Neutral",
            "Magpie-ZeroShot.Female-Angry",
            "Magpie-ZeroShot.Female-Fearful",
            "Magpie-ZeroShot.Female-Calm",
            "Magpie-ZeroShot.Female-Happy",
            "Magpie-ZeroShot.Male-1",
            "Magpie-ZeroShot.Male-Calm",
            "Magpie-ZeroShot.Male-Neutral",
            "Magpie-ZeroShot.Male-Angry",
            "Magpie-ZeroShot.Male-Fearful",
        ],
    }

    @property
    def ast_service(self) -> GRPCRIVAASTClient:
        """AST service assignment to transactional AST client."""
        return GRPCRIVATransactionalASTClient


class S2SRIVAStreamingService(S2SRIVAAbstractService):
    """RIVA Streaming Speech-to-Speech service.

    This class currently serves only RIVA Magpie Multilingual NIM for TTS.
    The Magpie Multilingual TTS model only supports en-US as an output
    (target) language.
    """

    supported_source_languages = [
        "en-US",
        "es-US",
        "fr-FR",
    ]
    # Magpie Multilingual TTS only supports en-US as output language.
    supported_target_languages = [
        "en-US",
    ]
    use_auto_zero_shot = False
    supported_voice_names = {
        "en-US": [
            "Magpie-Multilingual.EN-US.Sofia",
            "Magpie-Multilingual.EN-US.Ray",
            "Magpie-Multilingual.EN-US.Sofia.Calm",
            "Magpie-Multilingual.EN-US.Sofia.Fearful",
            "Magpie-Multilingual.EN-US.Sofia.Happy",
            "Magpie-Multilingual.EN-US.Sofia.Angry",
            "Magpie-Multilingual.EN-US.Sofia.Neutral",
            "Magpie-Multilingual.EN-US.Ray.Calm",
            "Magpie-Multilingual.EN-US.Ray.Fearful",
            "Magpie-Multilingual.EN-US.Ray.Happy",
            "Magpie-Multilingual.EN-US.Ray.Neutral",
            "Magpie-Multilingual.EN-US.Ray.Angry",
            "Magpie-Multilingual.EN-US.Ray.Disgusted",
            "Magpie-Multilingual.EN-US.Ray",
        ],
        "es-US": [
            "Magpie-Multilingual.ES-US.Isabela",
            "Magpie-Multilingual.ES-US.Isabela.Neutral",
            "Magpie-Multilingual.ES-US.Isabela.Angry",
            "Magpie-Multilingual.ES-US.Isabela.Happy",
            "Magpie-Multilingual.ES-US.Isabela.Calm",
            "Magpie-Multilingual.ES-US.Isabela.Pleasant_Surprise",
            "Magpie-Multilingual.ES-US.Isabela.Sad",
            "Magpie-Multilingual.ES-US.Diego",
            "Magpie-Multilingual.ES-US.Diego.Neutral",
            "Magpie-Multilingual.ES-US.Diego.Angry",
            "Magpie-Multilingual.ES-US.Diego.Happy",
            "Magpie-Multilingual.ES-US.Diego.Calm",
            "Magpie-Multilingual.ES-US.Diego.Pleasant_Surprise",
            "Magpie-Multilingual.ES-US.Diego.Sad",
            "Magpie-Multilingual.ES-US.Diego.Disgust",
        ],
        "fr-FR": [
            "Magpie-Multilingual.FR-FR.Louise",
            "Magpie-Multilingual.FR-FR.Louise.Angry",
            "Magpie-Multilingual.FR-FR.Louise.Calm",
            "Magpie-Multilingual.FR-FR.Louise.Disgust",
            "Magpie-Multilingual.FR-FR.Louise.Sad",
            "Magpie-Multilingual.FR-FR.Louise.Happy",
            "Magpie-Multilingual.FR-FR.Louise.Fearful",
            "Magpie-Multilingual.FR-FR.Louise.Neutral",
            "Magpie-Multilingual.FR-FR.Pascal",
            "Magpie-Multilingual.FR-FR.Pascal.Neutral",
            "Magpie-Multilingual.FR-FR.Pascal.Angry",
            "Magpie-Multilingual.FR-FR.Pascal.Calm",
            "Magpie-Multilingual.FR-FR.Pascal.Sad",
        ],
    }

    @property
    def ast_service(self) -> GRPCRIVAASTClient:
        """AST service assignment to streaming AST client."""
        return GRPCRIVAStreamingASTClient
