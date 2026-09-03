"""Prototipo de render: a partir de la posición de facecam ya detectada
(Enfoque 1), genera un clip corto de muestra en el formato "cámara arriba,
juego abajo" que el usuario describió — para ver un resultado concreto,
no solo números de detección.

Standalone y separado del pipeline principal a propósito, igual que el
resto del experimento: esto es un prototipo para decidir si vale la pena
integrarlo, no la integración en sí.

Uso: python facecam_render_demo.py <video.mp4> <x> <y> <w> <h> <start_seconds> [--duration 15]

x,y,w,h son la zona detectada por facecam_experiment.py, en píxeles, sobre
un frame escalado a 1280 de ancho (_FRAME_SCALE en ese script) — este
script reescala esas coordenadas a la resolución real del video.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

_DETECTION_SCALE = 1280  # tiene que coincidir con _FRAME_SCALE de facecam_experiment.py

OUTPUT_W, OUTPUT_H = 1080, 1920
TOP_H = int(OUTPUT_H * 0.38) // 2 * 2  # cámara arriba, un poco más chica que la mitad; par (h264 lo exige)
BOTTOM_H = OUTPUT_H - TOP_H


def _get_video_width(video_path: Path) -> int:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width", "-of", "csv=p=0", str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return int(result.stdout.strip())


def render_demo(video_path: Path, bbox: tuple[int, int, int, int], start: float, duration: float, out_path: Path) -> None:
    real_width = _get_video_width(video_path)
    scale_factor = real_width / _DETECTION_SCALE
    x, y, w, h = (round(v * scale_factor) for v in bbox)

    # Le sumamos un margen alrededor de la cara detectada (no queremos el
    # recorte pegado al borde de la cara, se ve mejor con un poco de aire)
    # y forzamos relación de aspecto acorde al panel de arriba.
    margin = round(w * 0.7)
    cx, cy = x + w // 2, y + h // 2
    crop_w = w + margin * 2
    crop_h = round(crop_w * TOP_H / OUTPUT_W)
    crop_x = max(0, cx - crop_w // 2)
    crop_y = max(0, cy - crop_h // 2)

    filter_complex = (
        f"[0:v]trim=start={start}:duration={duration},setpts=PTS-STARTPTS,"
        f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale={OUTPUT_W}:{TOP_H}[top];"
        f"[0:v]trim=start={start}:duration={duration},setpts=PTS-STARTPTS,"
        f"scale={OUTPUT_W}:{BOTTOM_H}:force_original_aspect_ratio=increase,"
        f"crop={OUTPUT_W}:{BOTTOM_H}[bottom];"
        f"[top][bottom]vstack=inputs=2[outv];"
        f"[0:a]atrim=start={start}:duration={duration},asetpts=PTS-STARTPTS[outa]"
    )

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-map_metadata", "-1",  # sin esto, el título/canal del video original
                                 # queda pegado y el celular muestra SU PROPIO
                                 # panel de "reproduciendo ahora" (con ese
                                 # título y una miniatura) tapando el player
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac", "-b:a", "128k",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg fallo:\n{result.stderr[-3000:]}")
    print(f"listo: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("x", type=int)
    parser.add_argument("y", type=int)
    parser.add_argument("w", type=int)
    parser.add_argument("h", type=int)
    parser.add_argument("start", type=float)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    video_path = Path(args.video)
    out_path = Path(args.out) if args.out else video_path.parent / f"{video_path.stem}_demo_vertical.mp4"
    render_demo(video_path, (args.x, args.y, args.w, args.h), args.start, args.duration, out_path)


if __name__ == "__main__":
    main()
