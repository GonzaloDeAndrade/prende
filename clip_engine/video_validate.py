"""Valida que un archivo de video sirva ANTES de arrancar a procesarlo.

Sin esto, un archivo corrupto o sin pista de video recién se descubre varios
minutos después (a mitad de transcripción o de corte), después de haber
gastado tiempo de cómputo real. Un chequeo con ffprobe tarda menos de un
segundo, así que vale la pena hacerlo primero siempre.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

MIN_DURATION_SECONDS = 3.0


def validate_video(path: Path) -> tuple[bool, str]:
    """Devuelve (ok, mensaje). Si ok es False, el mensaje explica por qué el
    archivo no sirve para procesar."""
    path = Path(path)
    if not path.exists():
        return False, "el archivo no existe"
    if path.stat().st_size == 0:
        return False, "el archivo está vacío (0 bytes) — la subida o descarga puede haberse cortado"

    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or "").strip().splitlines()
        last_line = detail[-1] if detail else "ffprobe no pudo leer el archivo"
        return False, f"archivo de video inválido o corrupto ({last_line})"

    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False, "ffprobe no devolvió metadata legible para este archivo"

    streams = info.get("streams", [])
    if not any(s.get("codec_type") == "video" for s in streams):
        return False, "el archivo no tiene ninguna pista de video"

    duration_raw = info.get("format", {}).get("duration")
    try:
        duration = float(duration_raw)
    except (TypeError, ValueError):
        return False, "no se pudo determinar la duración del video"

    if duration < MIN_DURATION_SECONDS:
        return False, f"el video dura {duration:.1f}s, muy corto para procesar (mínimo {MIN_DURATION_SECONDS:.0f}s)"

    return True, f"ok ({duration:.0f}s)"


def get_duration_seconds(path: Path) -> float | None:
    """Duración del video en segundos, o None si ffprobe no puede leerla.
    Usado para estimar cuánto puede tardar el análisis (no para validar)."""
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json", "-show_format", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    try:
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
