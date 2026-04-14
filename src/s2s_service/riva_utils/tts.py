# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RIVA TTS client."""

import time
import traceback
from collections.abc import Iterator

import grpc
import riva.client
import riva.client.proto.riva_tts_pb2 as rtts

from base_utils import logger
from common.buffers import Buffer
from common.clients import Client
from s2s_service.riva_utils.servers import RivaTTSServer


class GRPCRIVATTSClient(Client[str, object]):
    """RIVA TTS client that writes audio chunks into an output buffer."""

    def __init__(
        self,
        server: RivaTTSServer,
        sample_rate_hz: int = 16000,
    ) -> None:
        """Initialize the TTS client.

        Args:
            server (RivaTTSServer): The server to connect to.
            sample_rate_hz (int, optional): The sample rate in Hz. Defaults to 16000.
        """
        super().__init__(server=server)  # type: ignore[arg-type]
        self.sample_rate_hz = sample_rate_hz
        if self.server.stub is None:
            self.server.create_server()
        self._tts_service = self.server.stub

    @classmethod
    def from_string(cls, url: str) -> "GRPCRIVATTSClient":
        """Create a TTS client from a string.

        Args:
            url (str): The URL to create the TTS client from.

        Returns:
            GRPCRIVATTSClient: The TTS client.
        """
        return cls(RivaTTSServer.from_string(url))

    def _impl(
        self,
        request_iterator: Iterator[str],
        output_buffer: Buffer[object],
        context: grpc.ServicerContext,
        request_id: str,
        language_code: str = "es-US",
        voice_name: str | None = None,
        zero_shot_data: dict | None = None,
    ) -> None:
        """Run TTS inference on the text stream and write responses to output_buffer.

        Args:
            request_iterator: Text chunks to synthesize.
            output_buffer: Destination buffer for audio responses.
            context: gRPC servicer context.
            request_id: Correlation identifier.
            language_code: Language code for synthesis.
            voice_name: Optional voice name.
            zero_shot_data: Optional zero-shot config with prompt audio and metadata.

        Returns:
            None. Audio chunks are written to ``output_buffer``.
        """
        logger.info(f"Running TTS for request id {request_id}")

        if zero_shot_data:
            logger.info("TTS using zero-shot mode")
        else:
            logger.info(f"TTS using voice_name={voice_name}")

        logger.debug(f"TTS language_code={language_code}")
        _start_time_for_tts = time.time()

        # Map encoding string to riva.client.AudioEncoding enum
        encoding_map = {
            "LINEAR_PCM": riva.client.AudioEncoding.LINEAR_PCM,
            "OGGOPUS": riva.client.AudioEncoding.OGGOPUS,
        }

        try:
            for text_chunk in request_iterator:
                logger.debug(f"Synthesizing text chunk: {text_chunk[:50]}...")

                # Build proto request directly to avoid voice_name truncation issues
                # The riva.client.synthesize_online() convenience method may parse
                # voice_name incorrectly
                req = rtts.SynthesizeSpeechRequest(
                    text=text_chunk,
                    language_code=language_code,
                    sample_rate_hz=self.sample_rate_hz,
                    encoding=riva.client.AudioEncoding.LINEAR_PCM,
                )

                if zero_shot_data:
                    # Zero-shot mode: use proto directly to properly set sample_rate_hz
                    # Do NOT set voice_name for pure zero-shot (voice_name=None works!)

                    # Set zero-shot data with sample_rate_hz (CRITICAL for duration calculation)
                    # Read as WAV and extract only PCM data (no headers)
                    import wave

                    with wave.open(str(zero_shot_data["audio_prompt_file"]), "rb") as wav_file:
                        wav_sr = wav_file.getframerate()
                        audio_prompt_data = wav_file.readframes(wav_file.getnframes())

                    req.zero_shot_data.audio_prompt = audio_prompt_data
                    # Use actual sample rate from WAV file to ensure consistency
                    req.zero_shot_data.sample_rate_hz = wav_sr
                    req.zero_shot_data.encoding = encoding_map.get(
                        zero_shot_data["encoding"], riva.client.AudioEncoding.LINEAR_PCM
                    )
                    req.zero_shot_data.quality = zero_shot_data["quality"]
                else:
                    # Regular TTS with voice name
                    logger.debug(f"Using regular TTS with voice_name={voice_name}")
                    # Set voice_name in the proto request directly to preserve full name with dots
                    if voice_name is not None:
                        req.voice_name = voice_name

                # Call proto directly to avoid any voice_name parsing issues
                tts_response = self._tts_service.stub.SynthesizeOnline(
                    req, metadata=self._tts_service.auth.get_auth_metadata()
                )

                logger.debug(
                    f"Time taken for text chunk: {text_chunk} is "
                    f"{time.time() - _start_time_for_tts:.2f} seconds."
                )
                _start_time_for_tts = time.time()
                for audio_chunk in tts_response:
                    output_buffer.put(audio_chunk)

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Exception in TTS:\n{tb}")
            context.abort(grpc.StatusCode.INTERNAL, f"{type(e).__name__}: {e}\n{tb}")
