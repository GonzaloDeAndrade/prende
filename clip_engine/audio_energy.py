"""Detección de risas/reacciones fuertes por energía de audio.

No es un clasificador de risa entrenado — es una heurística de volumen: busca
tramos donde el audio sube muy por encima del nivel normal de charla del
video (carcajadas, gritos de emoción, reacciones grupales) y los marca como
señal extra para el LLM de selección, que ya evalúa el texto. No reemplaza
ese análisis, lo complementa: corre 100% local (ffmpeg + numpy), sin costo
de API.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from .config import TRANSCRIPTS_DIR

SAMPLE_RATE = 16000
WINDOW_SECONDS = 0.5
MIN_Z = 2.2  # desvíos (robustos) por encima de la mediana para contar como pico


def _extract_pcm(video_path: Path) -> np.ndarray:
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(video_path),
        "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg no pudo extraer audio para detección de energía: "
            f"{result.stderr[-2000:].decode(errors='ignore')}"
        )
    return np.frombuffer(result.stdout, dtype=np.float32)


def _rms_windows(pcm: np.ndarray, sr: int, window_seconds: float) -> np.ndarray:
    window = max(1, int(sr * window_seconds))
    n_windows = len(pcm) // window
    if n_windows == 0:
        return np.array([])
    trimmed = pcm[: n_windows * window]
    reshaped = trimmed.reshape(n_windows, window)
    return np.sqrt(np.mean(reshaped ** 2, axis=1))


def _find_spikes(rms: np.ndarray) -> list[tuple[float, float, float]]:
    if len(rms) < 10:
        return []

    # Mediana/MAD en vez de media/desvío estándar: son más robustos a que el
    # video tenga tramos largos de silencio o un pico único gigante que
    # arrastre el promedio y esconda el resto de las reacciones.
    median = np.median(rms)
    mad = np.median(np.abs(rms - median)) or 1e-9
    z = (rms - median) / (mad * 1.4826)  # normaliza MAD a escala de desvío estándar

    spikes: list[tuple[float, float, float]] = []
    in_spike = False
    spike_start = 0.0
    spike_max_z = 0.0
    for i, zi in enumerate(z):
        t = i * WINDOW_SECONDS
        if zi >= MIN_Z:
            if not in_spike:
                in_spike = True
                spike_start = t
                spike_max_z = zi
            else:
                spike_max_z = max(spike_max_z, zi)
        elif in_spike:
            spikes.append((spike_start, t, round(float(spike_max_z), 2)))
            in_spike = False
    if in_spike:
        spikes.append((spike_start, len(z) * WINDOW_SECONDS, round(float(spike_max_z), 2)))

    # Fusionamos picos separados por menos de 1s: suele ser la misma reacción
    # con un bache breve en el medio (alguien toma aire en la carcajada, etc).
    merged: list[tuple[float, float, float]] = []
    for s, e, zi in spikes:
        if merged and s - merged[-1][1] < 1.0:
            merged[-1] = (merged[-1][0], e, max(merged[-1][2], zi))
        else:
            merged.append((s, e, zi))
    return merged


def detect_energy_spikes(video_path: Path, video_stem: str, force: bool = False) -> list[tuple[float, float, float]]:
    """Devuelve tramos (start, end, intensidad) donde el volumen sube muy por
    encima del nivel normal de charla del video. `intensidad` es un z-score
    robusto: más alto = pico más marcado. Cachea el resultado igual que la
    transcripción, para no re-decodificar el audio en corridas siguientes.
    """
    out_path = TRANSCRIPTS_DIR / f"{video_stem}.energy.json"
    if out_path.exists() and not force:
        return [tuple(x) for x in json.loads(out_path.read_text(encoding="utf-8"))]

    pcm = _extract_pcm(video_path)
    rms = _rms_windows(pcm, SAMPLE_RATE, WINDOW_SECONDS)
    spikes = _find_spikes(rms)

    out_path.write_text(json.dumps(spikes), encoding="utf-8")
    return spikes
