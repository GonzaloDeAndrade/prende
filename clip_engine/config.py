"""Configuración central del pipeline, cargada desde variables de entorno (.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import truststore

# Usa el almacén de certificados del sistema operativo (Windows) en vez de la
# lista interna de certifi. Antivirus con inspección HTTPS (Avast, etc.)
# inyectan su propio certificado raíz en el almacén de Windows, que certifi
# no conoce — esto evita el CERTIFICATE_VERIFY_FAILED sin tener que
# desactivar el antivirus cada vez.
truststore.inject_into_ssl()

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
CLIPS_DIR = DATA_DIR / "clips"
TMP_DIR = DATA_DIR / "tmp"

for d in (INPUT_DIR, TRANSCRIPTS_DIR, CLIPS_DIR, TMP_DIR):
    d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    # Transcripción (faster-whisper, local)
    whisper_model_size: str = os.getenv("WHISPER_MODEL_SIZE", "large-v3")
    whisper_device: str = os.getenv("WHISPER_DEVICE", "auto")  # auto | cuda | cpu
    whisper_language: str = os.getenv("WHISPER_LANGUAGE", "es")

    # Análisis de clips (OpenAI)
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Reglas de selección de clips
    min_clip_seconds: int = int(os.getenv("MIN_CLIP_SECONDS", "10"))
    max_clip_seconds: int = int(os.getenv("MAX_CLIP_SECONDS", "30"))
    min_clips: int = int(os.getenv("MIN_CLIPS", "5"))
    max_clips: int = int(os.getenv("MAX_CLIPS", "10"))

    # Formato de salida
    output_width: int = int(os.getenv("OUTPUT_WIDTH", "1080"))
    output_height: int = int(os.getenv("OUTPUT_HEIGHT", "1920"))
    subtitle_font: str = os.getenv("SUBTITLE_FONT", "Arial")
    subtitle_font_size: int = int(os.getenv("SUBTITLE_FONT_SIZE", "64"))
    words_per_caption: int = int(os.getenv("WORDS_PER_CAPTION", "4"))


settings = Settings()
