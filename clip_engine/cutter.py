"""Corte de clips + transformación a formato vertical + quemado de subtítulos, todo con ffmpeg.

Un clip puede tener una o dos "partes" (rangos de tiempo no contiguos del video
original) que se pegan con un corte seco. Cada parte se abre como un input de
ffmpeg independiente (con su propio seek rápido) para no tener que decodificar
el tramo muerto entre partes lejanas, y se concatenan con el filtro `concat`
antes de aplicar el formato vertical, el zoom y los subtítulos.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .config import settings


def _escape_for_filter(path: Path) -> str:
    """Escapa una ruta de Windows para usarla dentro de un filtro de ffmpeg (subtitles=...)."""
    p = str(path).replace("\\", "/")
    p = p.replace(":", "\\:")
    return p


OUTPUT_FPS = 30
SEEK_MARGIN = 5.0

# Zoom-in estilo Ken Burns: tope 20%, llega al tope en ~8s. Verificado por
# diff de píxeles que un ritmo más lento (0.0002, tope 15%) sí se aplicaba
# pero era imperceptible mirando el video normal — esto tiene que notarse
# a simple vista, no solo en un análisis de píxeles.
_ZOOM_EXPR = "min(zoom+0.0008,1.20)"


def _build_filter_complex(parts: list[tuple[float, float]], seeks: list[float], w: int, h: int, ass_path: str) -> str:
    trims = []
    concat_inputs = []
    for i, ((start, end), seek) in enumerate(zip(parts, seeks)):
        rel_start = start - seek
        rel_end = end - seek
        trims.append(
            f"[{i}:v]trim=start={rel_start:.3f}:end={rel_end:.3f},setpts=PTS-STARTPTS[v{i}];"
            f"[{i}:a]atrim=start={rel_start:.3f}:end={rel_end:.3f},asetpts=PTS-STARTPTS[a{i}];"
        )
        concat_inputs.append(f"[v{i}][a{i}]")

    concat = ""
    if len(parts) > 1:
        concat = f"{''.join(concat_inputs)}concat=n={len(parts)}:v=1:a=1[catv][cata];"
        src_v, src_a = "[catv]", "[cata]"
    else:
        src_v, src_a = "[v0]", "[a0]"

    return (
        f"{''.join(trims)}"
        f"{concat}"
        f"{src_v}split=2[base1][base2];"
        f"[base1]scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h},boxblur=20:5[bg];"
        f"[base2]scale={w}:{h}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,fps={OUTPUT_FPS},"
        f"zoompan=z='{_ZOOM_EXPR}':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"s={w}x{h}:fps={OUTPUT_FPS}[zoomed];"
        f"[zoomed]subtitles='{ass_path}'[outv];"
        f"{src_a}anull[outa]"
    )


def _encode_cmd(
    source_video: Path,
    seeks: list[float],
    filter_complex: str,
    out_path: Path,
    use_gpu: bool,
) -> list[str]:
    if use_gpu:
        video_codec = ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "20", "-b:v", "0"]
    else:
        video_codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]

    cmd = ["ffmpeg", "-y"]
    for seek in seeks:
        cmd += ["-ss", f"{seek:.3f}", "-i", str(source_video)]
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "[outa]",
        *video_codec,
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    return cmd


_PREVIEW_WIDTH = 480
_PREVIEW_HEIGHT = 854


def _build_preview_filter_complex(parts: list[tuple[float, float]], seeks: list[float], w: int, h: int, ass_path: str) -> str:
    trims = []
    concat_inputs = []
    for i, ((start, end), seek) in enumerate(zip(parts, seeks)):
        rel_start = start - seek
        rel_end = end - seek
        trims.append(
            f"[{i}:v]trim=start={rel_start:.3f}:end={rel_end:.3f},setpts=PTS-STARTPTS[v{i}];"
            f"[{i}:a]atrim=start={rel_start:.3f}:end={rel_end:.3f},asetpts=PTS-STARTPTS[a{i}];"
        )
        concat_inputs.append(f"[v{i}][a{i}]")

    concat = ""
    if len(parts) > 1:
        concat = f"{''.join(concat_inputs)}concat=n={len(parts)}:v=1:a=1[catv][cata];"
        src_v, src_a = "[catv]", "[cata]"
    else:
        src_v, src_a = "[v0]", "[a0]"

    # Mismo recorte con fondo difuminado que la versión final (para que la
    # decisión sea representativa de cómo va a quedar), pero SIN zoompan —
    # es la parte más cara de codificar y la menos relevante para elegir.
    return (
        f"{''.join(trims)}"
        f"{concat}"
        f"{src_v}split=2[base1][base2];"
        f"[base1]scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h},boxblur=12:3[bg];"
        f"[base2]scale={w}:{h}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,subtitles='{ass_path}'[outv];"
        f"{src_a}anull[outa]"
    )


def render_preview(
    source_video: Path,
    parts: list[tuple[float, float]],
    subtitles_ass: Path,
    out_path: Path,
) -> Path:
    """Versión rápida y liviana del clip (sin blur de fondo ni zoom, baja
    resolución, preset ultrafast) solo para poder mirarlo y decidir — no para
    publicar. render_clip() hace la versión final con el tratamiento completo.
    """
    seeks = [max(0.0, start - SEEK_MARGIN) for start, _ in parts]
    ass_path = _escape_for_filter(subtitles_ass)
    filter_complex = _build_preview_filter_complex(parts, seeks, _PREVIEW_WIDTH, _PREVIEW_HEIGHT, ass_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y"]
    for seek in seeks:
        cmd += ["-ss", f"{seek:.3f}", "-i", str(source_video)]
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
        "-c:a", "aac", "-b:a", "96k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg falló generando preview de {out_path.name}:\n{result.stderr[-3000:]}")
    return out_path


def render_clip(
    source_video: Path,
    parts: list[tuple[float, float]],
    subtitles_ass: Path,
    out_path: Path,
) -> Path:
    """Corta una o dos partes de source_video, las pega con un corte seco si
    son dos, lo pasa a vertical 9:16 (con fondo difuminado, sin recortar el
    contenido original), le aplica un zoom-in sutil y quema los subtítulos.
    Usa NVENC (GPU) si está disponible; si falla, cae a libx264 por CPU sin
    cambiar la calidad visual del resultado.
    """
    # Cada parte abre su propio input con seek rápido (por keyframe), así no
    # hay que decodificar el tramo muerto entre partes lejanas en el tiempo.
    seeks = [max(0.0, start - SEEK_MARGIN) for start, _ in parts]

    w, h = settings.output_width, settings.output_height
    ass_path = _escape_for_filter(subtitles_ass)
    filter_complex = _build_filter_complex(parts, seeks, w, h, ass_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = _encode_cmd(source_video, seeks, filter_complex, out_path, use_gpu=True)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[cutter] NVENC falló para {out_path.name}, reintentando por CPU...")
        cmd = _encode_cmd(source_video, seeks, filter_complex, out_path, use_gpu=False)
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg falló en {out_path.name}:\n{result.stderr[-3000:]}")
    return out_path
