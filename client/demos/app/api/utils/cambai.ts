/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Camb AI transcription API client for diarization.
 *
 * Three-step flow: POST /transcribe → poll GET /transcribe/{task_id}
 * → GET /transcription-result/{run_id}?word_level_timestamps=true
 */

import fs from "fs";
import path from "path";
import logger from "../../utils/logger";

const CAMB_API_BASE_URL = "https://client.camb.ai/apis";
const CAMB_API_KEY = process.env.CAMB_API_KEY || "";

const POLL_INTERVAL_MS = 10_000;
const MAX_POLL_ATTEMPTS = 120;

/**
 * Submit a transcription request to Camb AI.
 * @param audioFilePath - Path to the audio file to transcribe
 * @param languageId - Camb AI numeric language ID (default: 1 for English)
 * @returns The task ID for polling
 */
async function submitTranscription(audioFilePath: string, languageId: number = 1): Promise<string> {
  const fileBuffer = fs.readFileSync(audioFilePath);
  const fileName = path.basename(audioFilePath);

  const formData = new FormData();
  formData.append("media_file", new Blob([fileBuffer]), fileName);
  formData.append("language", String(languageId));

  const response = await fetch(`${CAMB_API_BASE_URL}/transcribe`, {
    method: "POST",
    headers: { "x-api-key": CAMB_API_KEY },
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Camb AI /transcribe failed (${response.status}): ${text}`);
  }

  const data = await response.json();
  const taskId = data.task_id;
  if (!taskId) {
    throw new Error(`Camb AI /transcribe response missing task_id: ${JSON.stringify(data)}`);
  }
  return String(taskId);
}

/**
 * Poll Camb AI transcription status until SUCCESS.
 * @param taskId - Task ID from submitTranscription
 * @returns The run_id for fetching results
 */
async function waitForTranscription(taskId: string): Promise<number> {
  for (let attempt = 0; attempt < MAX_POLL_ATTEMPTS; attempt++) {
    const response = await fetch(`${CAMB_API_BASE_URL}/transcribe/${taskId}`, {
      headers: { "x-api-key": CAMB_API_KEY },
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Camb AI status check failed (${response.status}): ${text}`);
    }

    const data = await response.json();
    const status = String(data.status ?? "").toUpperCase();

    if (status === "SUCCESS") {
      const runId = data.run_id;
      if (typeof runId !== "number") {
        throw new Error(`Camb AI status missing run_id on SUCCESS: ${JSON.stringify(data)}`);
      }
      return runId;
    }

    if (["ERROR", "TIMEOUT", "PAYMENT_REQUIRED"].includes(status)) {
      throw new Error(`Camb AI transcription failed: status=${status}, message=${data.message}`);
    }

    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  }

  throw new Error(`Camb AI transcription timed out after ${MAX_POLL_ATTEMPTS} attempts`);
}

/**
 * Fetch the transcription result with word-level timestamps.
 * @param runId - Run ID from waitForTranscription
 * @returns The transcription result JSON (array of segments)
 */
async function getTranscriptionResult(runId: number): Promise<any> {
  const url = `${CAMB_API_BASE_URL}/transcription-result/${runId}?word_level_timestamps=true`;
  const response = await fetch(url, {
    headers: { "x-api-key": CAMB_API_KEY },
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Camb AI /transcription-result failed (${response.status}): ${text}`);
  }

  return response.json();
}

/**
 * Perform Camb AI diarization: submit → poll → fetch result.
 * Returns null if CAMB_API_KEY is not set (graceful skip).
 * @param audioFilePath - Path to the audio file
 * @param streamId - Unique stream identifier for logging
 * @param languageId - Camb AI numeric language ID (default: 1)
 * @returns Transcription result or null if skipped
 */
export async function performCambAiDiarization(
  audioFilePath: string,
  streamId: string,
  languageId: number = 1,
): Promise<any | null> {
  if (!CAMB_API_KEY) {
    logger.warn("CAMB_API_KEY not set. Skipping Camb AI diarization.");
    return null;
  }

  if (!fs.existsSync(audioFilePath)) {
    throw new Error(`Audio file not found for diarization: ${audioFilePath}`);
  }

  logger.info(`Starting Camb AI diarization for: ${audioFilePath} (stream: ${streamId})`);

  try {
    const taskId = await submitTranscription(audioFilePath, languageId);
    logger.info(`Camb AI transcription submitted: taskId=${taskId}`);

    const runId = await waitForTranscription(taskId);
    logger.info(`Camb AI transcription completed: runId=${runId}`);

    const result = await getTranscriptionResult(runId);
    logger.info(`Camb AI diarization completed for stream: ${streamId}`);

    return result;
  } catch (error) {
    logger.error(`Error during Camb AI diarization: ${error}`);
    throw error;
  }
}

/**
 * Save Camb AI diarization data to a JSON file.
 * @param streamId - Unique stream identifier
 * @param data - Diarization response data
 * @param outputDir - Directory to save the file
 * @returns Path to the saved file
 */
export async function saveCambAiDiarizationFile(streamId: string, data: any, outputDir: string): Promise<string> {
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  const filePath = path.join(outputDir, `${streamId}.json`);
  await fs.promises.writeFile(filePath, JSON.stringify(data, null, 2), "utf-8");
  logger.info(`Camb AI diarization saved to ${filePath}`);
  return filePath;
}
