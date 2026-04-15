/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { NextResponse } from "next/server";
import { getEnablePreprocessingFromEnv } from "../../utils/audioProcessing";

const S2S_SERVICE = process.env.S2S_SERVICE;

let SUPPORTED_SOURCE_LANGUAGES: { code: string; label: string }[] = [];
let SUPPORTED_TARGET_LANGUAGES: { code: string; label: string }[] = [];
let DEFAULT_TARGET_LANGUAGE = "";
let DEFAULT_SOURCE_LANGUAGE = "";

const isLanguageSupported = (languages: { code: string; label: string }[], language: string): boolean => {
  return languages.some((l) => l.code === language);
};

if (S2S_SERVICE === "RIVA_TRANSACTIONAL") {
  SUPPORTED_SOURCE_LANGUAGES = [
    { code: "ar-AR", label: "Arabic (ar-AR)" },
    { code: "en-US", label: "English (en-US)" },
    { code: "fr-FR", label: "French (fr-FR)" },
    { code: "de-DE", label: "German (de-DE)" },
    { code: "hi-IN", label: "Hindi (hi-IN)" },
    { code: "it-IT", label: "Italian (it-IT)" },
    { code: "ja-JP", label: "Japanese (ja-JP)" },
    { code: "ko-KR", label: "Korean (ko-KR)" },
    { code: "pt-BR", label: "Portuguese (pt-BR)" },
    { code: "ru-RU", label: "Russian (ru-RU)" },
    { code: "es-ES", label: "Spanish (es-ES)" },
    { code: "es-US", label: "Spanish (es-US)" },
  ];

  SUPPORTED_TARGET_LANGUAGES = [{ code: "en-US", label: "English (en-US)" }];

  DEFAULT_SOURCE_LANGUAGE = "es-US";
  DEFAULT_TARGET_LANGUAGE = "en-US";
} else if (S2S_SERVICE === "RIVA_STREAMING") {
  // NOTE: The RIVA Magpie Multilingual TTS model only supports en-US as an
  // output (target) language.
  SUPPORTED_SOURCE_LANGUAGES = [
    { code: "en-US", label: "English (en-US)" },
    { code: "fr-FR", label: "French (fr-FR)" },
    { code: "es-US", label: "Spanish (es-US)" },
  ];

  SUPPORTED_TARGET_LANGUAGES = [{ code: "en-US", label: "English (en-US)" }];

  DEFAULT_SOURCE_LANGUAGE = "es-US";
  DEFAULT_TARGET_LANGUAGE = "en-US";
} else if (S2S_SERVICE === "EL_DUBBING") {
  SUPPORTED_SOURCE_LANGUAGES = [
    { code: "auto", label: "Auto detect" },
    { code: "nl", label: "Dutch (nl-NL)" },
    { code: "en", label: "English (en-US)" },
    { code: "fr", label: "French (fr-FR)" },
    { code: "de", label: "German (de-DE)" },
    { code: "es", label: "Spanish (es-ES)" },
  ];

  SUPPORTED_TARGET_LANGUAGES = [
    { code: "nl", label: "Dutch (nl-NL)" },
    { code: "en", label: "English (en-US)" },
    { code: "fr", label: "French (fr-FR)" },
    { code: "de", label: "German (de-DE)" },
    { code: "es", label: "Spanish (es-ES)" },
  ];

  DEFAULT_SOURCE_LANGUAGE = "auto";
  DEFAULT_TARGET_LANGUAGE = "de";
} else if (S2S_SERVICE === "CAMB_DUBBING") {
  // CambAI uses numeric language IDs as strings.
  // Scoped to the same languages as ElevenLabs for now.
  SUPPORTED_SOURCE_LANGUAGES = [
    { code: "22", label: "Dutch (nl-NL)" },
    { code: "1", label: "English (en-US)" },
    { code: "25", label: "French (fr-FR)" },
    { code: "26", label: "German (de-DE)" },
    { code: "54", label: "Spanish (es-ES)" },
  ];

  SUPPORTED_TARGET_LANGUAGES = [...SUPPORTED_SOURCE_LANGUAGES];

  DEFAULT_SOURCE_LANGUAGE = "1";
  DEFAULT_TARGET_LANGUAGE = "26";
}

export async function GET() {
  try {
    const enablePreprocessing = getEnablePreprocessingFromEnv();

    const configMapping = {
      target_language: process.env.TARGET_LANGUAGE_LABEL,
      voice_name: process.env.VOICE_NAME,
      enable_preprocessing: enablePreprocessing,
      supported_source_languages: SUPPORTED_SOURCE_LANGUAGES,
      supported_target_languages: SUPPORTED_TARGET_LANGUAGES,
      default_source_language: isLanguageSupported(
        SUPPORTED_SOURCE_LANGUAGES,
        process.env.DEFAULT_SOURCE_LANGUAGE || DEFAULT_SOURCE_LANGUAGE,
      )
        ? process.env.DEFAULT_SOURCE_LANGUAGE
        : DEFAULT_SOURCE_LANGUAGE,
      default_target_language: isLanguageSupported(
        SUPPORTED_TARGET_LANGUAGES,
        process.env.DEFAULT_TARGET_LANGUAGE || DEFAULT_TARGET_LANGUAGE,
      )
        ? process.env.DEFAULT_TARGET_LANGUAGE
        : DEFAULT_TARGET_LANGUAGE,
    };

    return NextResponse.json(configMapping);
  } catch (error) {
    return NextResponse.json({ error: "Failed to load config" }, { status: 500 });
  }
}
