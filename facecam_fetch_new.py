"""Reintento automático (con backoff) para descargar los 3 videos nuevos que
el usuario pasó puntualmente para el experimento de facecam, y en cuanto cada
uno baja, correr el Enfoque 1 (el que ya sabemos que funciona: full_range,
confianza 0.15, intervalo 12s) sobre ese video en particular.

A propósito separado de research_batch.py: estos 3 videos NO son parte del
corpus general, son material específico que el usuario pasó para este
experimento puntual. No se agregan al manifest.json del corpus.

Uso: python facecam_fetch_new.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from yt_dlp_helper import YT_DLP_FIX_ARGS

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "data" / "tmp" / "facecam_experiment"

VIDEOS = ["BWYfOLg7xdU", "OR525SXMvVM", "BbH64OVHlho"]

_COOLDOWNS = [5 * 60, 10 * 60, 20 * 60, 30 * 60, 45 * 60, 60 * 60]  # segundos, escalando
_MAX_ATTEMPTS = 12


def _download(video_id: str) -> bool:
    out_path = OUT_DIR / f"{video_id}.mp4"
    if out_path.exists():
        return True
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        sys.executable, "-c",
        "import truststore; truststore.inject_into_ssl(); from yt_dlp import main; main()",
        *YT_DLP_FIX_ARGS,
        "-f", "bv*[height<=1080]+ba/b[height<=1080]",
        "-o", str(OUT_DIR / f"{video_id}.%(ext)s"),
        "--merge-output-format", "mp4",
        url,
    ]
    print(f"  intentando bajar {video_id}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    ok = result.returncode == 0 and out_path.exists()
    if not ok:
        tail = (result.stderr or result.stdout or "").strip().splitlines()
        last_line = tail[-1] if tail else "(sin salida)"
        print(f"    falló: {last_line}")
    return ok


def _analyze(video_id: str) -> None:
    video_path = OUT_DIR / f"{video_id}.mp4"
    print(f"  {video_id} descargado, corriendo Enfoque 1...")
    cmd = [
        sys.executable, str(ROOT / "facecam_experiment.py"), str(video_path),
        "--approach-name", "enfoque1_full_range",
        "--model", "full_range", "--confidence", "0.15", "--interval", "12",
        "--box-color", "yellow",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"    [error análisis] {result.stderr[-2000:]}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pending = list(VIDEOS)
    attempt = 0

    while pending and attempt < _MAX_ATTEMPTS:
        attempt += 1
        print(f"\n=== intento {attempt}/{_MAX_ATTEMPTS} — pendientes: {pending} ===")
        still_pending = []
        for video_id in pending:
            if _download(video_id):
                _analyze(video_id)
            else:
                still_pending.append(video_id)
        pending = still_pending

        if pending:
            cooldown = _COOLDOWNS[min(attempt - 1, len(_COOLDOWNS) - 1)]
            print(f"  {len(pending)} pendientes, esperando {cooldown // 60} min antes de reintentar...")
            time.sleep(cooldown)

    if pending:
        print(f"\nNo se pudieron bajar (bloqueo persistente de YouTube): {pending}")
        print("Estos van a necesitar descarga manual cuando el usuario esté disponible.")
    else:
        print("\nLos 3 videos se bajaron y analizaron correctamente.")


if __name__ == "__main__":
    main()
