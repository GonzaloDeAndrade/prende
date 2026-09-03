"""Enfoque 3 del experimento de facecam: detección por patrón de movimiento,
sin depender de reconocer una cara. Hipótesis: la zona de la facecam tiene
un comportamiento de movimiento distinto al del gameplay alrededor — no
necesita verse bien la cara (funciona de perfil, tapada, mal iluminada),
solo que esa región se mueva "distinto" al resto de la pantalla.

Método:
1. Sampleamos frames DENSOS (cada ~0.5-1s — a diferencia de la detección de
   cara, acá necesitamos pares de frames consecutivos cercanos en el tiempo
   para medir movimiento real, no solo fotos sueltas).
2. Dividimos cada frame en una grilla (por defecto 16x9 bloques).
3. Por cada par de frames consecutivos, medimos la diferencia de píxeles
   promedio en cada bloque (qué tanto cambió esa zona).
4. Sobre toda la ventana de tiempo analizada, para cada bloque calculamos
   la media y el desvío del movimiento — la hipótesis es que la facecam
   tiene movimiento MODERADO y relativamente ESTABLE (una persona hablando
   frente a un fondo fijo), distinto de: zonas casi estáticas (UI fija,
   z-score de movimiento ~0) o zonas muy caóticas (gameplay con acción,
   cambios de cámara, HUD parpadeando).
5. Elegimos el bloque (o grupo de bloques contiguos) que mejor calza con
   ese perfil, priorizando además las esquinas (donde casi siempre va un
   facecam de verdad).

Separado del enfoque de cara a propósito, para poder compararlos.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "data" / "tmp" / "facecam_experiment"
_FRAME_SCALE = 640  # mas chico que el de cara: acá analizamos MUCHOS mas frames, hay que ir liviano
_GRID_COLS, _GRID_ROWS = 16, 9

# Perfil esperado de la zona facecam: ni casi-estatica (UI/fondo fijo) ni
# caotica (accion de gameplay). Estos umbrales son un punto de partida, no
# una verdad revelada — se calibran mirando los resultados reales.
_MIN_MOTION = 1.5    # por debajo de esto, consideramos "casi estatico" (no es facecam, es UI/fondo)
_MAX_MOTION = 25.0   # por encima de esto, consideramos "caotico" (accion de gameplay, no facecam)
_MAX_VARIANCE_RATIO = 1.5  # motion muy inconsistente (picos y valles grandes) tampoco encaja con "persona hablando"

# Bonus para bloques cerca de una esquina — un facecam real casi siempre
# esta en un corner, es una señal barata y fuerte que no cuesta nada usar.
_CORNER_BONUS = 0.3
_CORNER_MARGIN = 0.35  # fraccion del ancho/alto que cuenta como "cerca de la esquina"


def _get_duration(video_path: Path) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def _extract_frames_dense(video_path: Path, interval: float, max_frames: int, out_dir: Path) -> list[Path]:
    """Extrae una racha de frames consecutivos densos — usamos el filtro fps
    de ffmpeg en una sola pasada en vez de un -ss por frame (mucho más rápido
    para este volumen)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "f_%05d.jpg")
    fps = 1.0 / interval
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", str(video_path),
        "-vf", f"fps={fps:.4f},scale={_FRAME_SCALE}:-1",
        "-frames:v", str(max_frames), "-q:v", "4", pattern,
    ]
    subprocess.run(cmd, capture_output=True)
    return sorted(out_dir.glob("f_*.jpg"))


def _load_gray(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32)


def analyze_video_motion(video_path: Path, interval: float, window_seconds: float, out_dir: Path) -> dict:
    print(f"\n=== {video_path.name} ===")
    duration = _get_duration(video_path)
    max_frames = min(int(duration / interval), int(window_seconds / interval))
    raw_dir = out_dir / "_raw_frames"
    frames = _extract_frames_dense(video_path, interval, max_frames, raw_dir)
    print(f"  {len(frames)} frames densos extraídos (cada {interval:.1f}s, ventana de {window_seconds:.0f}s)")
    if len(frames) < 5:
        print("  muy pocos frames, salteo este video")
        return {"video": video_path.name, "verdict": "sin suficientes frames"}

    first = _load_gray(frames[0])
    h, w = first.shape
    block_h, block_w = h // _GRID_ROWS, w // _GRID_COLS

    # matriz de movimiento por bloque a lo largo del tiempo
    motion_series = np.zeros((_GRID_ROWS, _GRID_COLS, len(frames) - 1), dtype=np.float32)
    prev = first
    for i in range(1, len(frames)):
        cur = _load_gray(frames[i])
        diff = np.abs(cur - prev)
        for r in range(_GRID_ROWS):
            for c in range(_GRID_COLS):
                block = diff[r * block_h:(r + 1) * block_h, c * block_w:(c + 1) * block_w]
                motion_series[r, c, i - 1] = block.mean()
        prev = cur

    means = motion_series.mean(axis=2)
    stds = motion_series.std(axis=2)
    variance_ratio = np.divide(stds, means, out=np.full_like(means, 999.0), where=means > 0.01)

    # puntaje por bloque: encaja con el perfil "persona hablando" (movimiento
    # moderado, no demasiado errático) + bonus si está cerca de una esquina
    scores = np.full((_GRID_ROWS, _GRID_COLS), -1.0)
    for r in range(_GRID_ROWS):
        for c in range(_GRID_COLS):
            m, v = means[r, c], variance_ratio[r, c]
            if not (_MIN_MOTION <= m <= _MAX_MOTION):
                continue
            if v > _MAX_VARIANCE_RATIO:
                continue
            fr, fc = r / _GRID_ROWS, c / _GRID_COLS
            near_corner = (
                (fr < _CORNER_MARGIN or fr > 1 - _CORNER_MARGIN)
                and (fc < _CORNER_MARGIN or fc > 1 - _CORNER_MARGIN)
            )
            score = 1.0 / (1.0 + v)  # mas estable (v chico) = mejor score base
            if near_corner:
                score += _CORNER_BONUS
            scores[r, c] = score

    best_r, best_c = np.unravel_index(np.argmax(scores), scores.shape)
    best_score = scores[best_r, best_c]

    # Dibujamos la grilla completa con el bloque ganador resaltado, para
    # poder confirmar a ojo — no solo confiar en el número.
    sample_frame = frames[len(frames) // 2]
    annotated = out_dir / f"{video_path.stem}_grid.jpg"
    _draw_grid(sample_frame, means, best_r, best_c, block_w, block_h, annotated)

    bbox = (best_c * block_w, best_r * block_h, block_w, block_h)
    summary = {
        "video": video_path.name, "duration": duration, "n_frames": len(frames),
        "best_block": (int(best_r), int(best_c)), "best_score": round(float(best_score), 2),
        "best_block_mean_motion": round(float(means[best_r, best_c]), 2),
        "bbox_xywh": bbox, "annotated_frame": str(annotated),
    }
    print(f"  mejor bloque: fila={best_r} col={best_c} score={best_score:.2f} movimiento_promedio={means[best_r, best_c]:.2f}")
    print(f"  frame anotado: {annotated}")
    for path in frames:
        path.unlink()
    return summary


def _draw_grid(frame_path: Path, means: np.ndarray, best_r: int, best_c: int, block_w: int, block_h: int, out_path: Path) -> None:
    img = Image.open(frame_path).convert("RGB")
    import PIL.ImageDraw as ImageDraw
    draw = ImageDraw.Draw(img)
    rows, cols = means.shape
    max_m = means.max() or 1.0
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * block_w, r * block_h
            x1, y1 = x0 + block_w, y0 + block_h
            intensity = int(255 * min(1.0, means[r, c] / max_m))
            draw.rectangle([x0, y0, x1, y1], outline=(intensity, 60, 60), width=1)
    bx0, by0 = best_c * block_w, best_r * block_h
    draw.rectangle([bx0, by0, bx0 + block_w, by0 + block_h], outline=(0, 255, 0), width=5)
    img.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--interval", type=float, default=0.7, help="segundos entre frames densos")
    parser.add_argument("--window", type=float, default=90.0, help="segundos de video a analizar (ventana densa)")
    parser.add_argument("--approach-name", default="enfoque3_movimiento")
    args = parser.parse_args()

    approach_dir = OUT_DIR / args.approach_name
    summaries = [
        analyze_video_motion(Path(p), args.interval, args.window, approach_dir / Path(p).stem)
        for p in args.videos
    ]

    print(f"\n=== RESUMEN {args.approach_name} ===")
    for s in summaries:
        print(f"  {s}")


if __name__ == "__main__":
    main()
