"""Experimento standalone: ¿se puede detectar de forma confiable, automática,
la posición de un facecam superpuesto sobre gameplay? Separado del pipeline
principal a propósito — no toca clip_engine/ ni el server, es solo para
decidir si vale la pena construir la integración completa después.

Soporta comparar enfoques distintos sobre los mismos videos:
- Detección de cara (MediaPipe short_range o full_range) + filtro de
  consistencia de posición a lo largo del tiempo.
- (más adelante) detección por patrón de movimiento, sin depender de cara.

Salida: por cada video/enfoque, guarda los frames sampleados con un recuadro
dibujado donde se detectó algo, en una carpeta separada por enfoque — para
poder comparar a simple vista, no solo confiar en los números.

Uso:
  python facecam_experiment.py <video1.mp4> [video2.mp4] [...] \
      --approach-name enfoque1_full_range --model full_range \
      --confidence 0.15 --interval 12
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "data" / "tmp" / "facecam_experiment"
MODEL_PATHS = {
    "short_range": OUT_DIR / "blaze_face_short_range.tflite",
    "full_range": OUT_DIR / "blaze_face_full_range.tflite",
}

# Qué tan cerca (en fracción del ancho del frame) tienen que estar dos
# detecciones para contar como "la misma zona" al medir consistencia.
_POSITION_TOLERANCE = 0.08
_FRAME_SCALE = 1280


def _get_duration(video_path: Path) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def _extract_frame(video_path: Path, t: float, out_path: Path, scale: int = _FRAME_SCALE) -> bool:
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video_path),
        "-vf", f"scale={scale}:-1", "-frames:v", "1", "-q:v", "3", str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0 and out_path.exists()


def _draw_box(image_path: Path, bbox: tuple[int, int, int, int] | None, out_path: Path, color: str = "lime") -> None:
    if bbox is None:
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(image_path), str(out_path)]
    else:
        x, y, w, h = bbox
        cmd = [
            "ffmpeg", "-y", "-v", "error", "-i", str(image_path),
            "-vf", f"drawbox=x={x}:y={y}:w={w}:h={h}:color={color}@0.9:thickness=4",
            str(out_path),
        ]
    subprocess.run(cmd, capture_output=True)


def _cluster_and_score(detections: list[tuple], total_samples: int) -> tuple[float, tuple | None]:
    """Agrupa detecciones por cercanía de posición; el grupo más grande es el
    candidato a "zona real". Devuelve (consistencia_pct, zona_promedio)."""
    found = [d for d in detections if d[1] is not None]
    if not found:
        return 0.0, None
    clusters: list[list[tuple]] = []
    for item in found:
        b = item[1]
        cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
        placed = False
        for cluster in clusters:
            ref = cluster[0][1]
            rcx, rcy = ref[0] + ref[2] / 2, ref[1] + ref[3] / 2
            if abs(cx - rcx) < _POSITION_TOLERANCE * _FRAME_SCALE and abs(cy - rcy) < _POSITION_TOLERANCE * _FRAME_SCALE:
                cluster.append(item)
                placed = True
                break
        if not placed:
            clusters.append([item])
    dominant = max(clusters, key=len)
    consistency_pct = len(dominant) / total_samples
    avg_box = tuple(round(sum(d[1][i] for d in dominant) / len(dominant)) for i in range(4))
    return consistency_pct, avg_box


def analyze_video_face(detector, video_path: Path, frame_interval: float, out_dir: Path, box_color: str) -> dict:
    print(f"\n=== {video_path.name} ===")
    duration = _get_duration(video_path)
    n_samples = max(1, int(duration // frame_interval))
    print(f"duración {duration:.0f}s, {n_samples} frames a samplear cada {frame_interval:.1f}s")

    out_dir.mkdir(parents=True, exist_ok=True)

    detections = []  # (t, bbox o None, score)
    for i in range(n_samples):
        t = i * frame_interval
        raw_path = out_dir / f"raw_{i:04d}.jpg"
        if not _extract_frame(video_path, t, raw_path):
            continue
        image = mp.Image.create_from_file(str(raw_path))
        result = detector.detect(image)
        if result.detections:
            best = max(result.detections, key=lambda d: d.categories[0].score)
            bbox = best.bounding_box
            box = (bbox.origin_x, bbox.origin_y, bbox.width, bbox.height)
            detections.append((t, box, best.categories[0].score))
        else:
            detections.append((t, None, 0.0))

        annotated_path = out_dir / f"box_{i:04d}_t{int(t)}s.jpg"
        _draw_box(raw_path, detections[-1][1], annotated_path, box_color)
        raw_path.unlink()

    hit_rate = sum(1 for d in detections if d[1] is not None) / len(detections) if detections else 0.0
    consistency_pct, dominant_zone = _cluster_and_score(detections, len(detections))

    summary = {
        "video": video_path.name, "duration": duration, "n_samples": len(detections),
        "hit_rate": round(hit_rate, 2), "consistency_pct": round(consistency_pct, 2),
        "dominant_zone_xywh": dominant_zone, "output_dir": str(out_dir),
    }
    print(f"  detección: {hit_rate:.0%} | consistencia zona: {consistency_pct:.0%} | zona: {dominant_zone}")
    print(f"  frames guardados en: {out_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--approach-name", default="experimento")
    parser.add_argument("--model", choices=["short_range", "full_range"], default="short_range")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--interval", type=float, default=12.0)
    parser.add_argument("--box-color", default="lime")
    args = parser.parse_args()

    model_path = MODEL_PATHS[args.model]
    if not model_path.exists():
        raise RuntimeError(f"falta el modelo en {model_path}")

    print(f"=== {args.approach_name} === modelo={args.model} confianza={args.confidence} intervalo={args.interval}s")
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=args.confidence)
    detector = mp_vision.FaceDetector.create_from_options(options)

    approach_dir = OUT_DIR / args.approach_name
    summaries = [
        analyze_video_face(detector, Path(p), args.interval, approach_dir / Path(p).stem, args.box_color)
        for p in args.videos
    ]

    print(f"\n=== RESUMEN {args.approach_name} ===")
    for s in summaries:
        veredicto = "PROBABLE FACECAM" if s["consistency_pct"] >= 0.5 else "no concluyente / sin facecam claro"
        print(f"  {s['video']}: {veredicto} (detección {s['hit_rate']:.0%}, consistencia {s['consistency_pct']:.0%})")


if __name__ == "__main__":
    main()
