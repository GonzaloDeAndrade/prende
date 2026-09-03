"""Generación de subtítulos quemados (.ass) a partir de timestamps por palabra."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import settings

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.601

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline},1,2,60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _fmt_ts(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _sanitize_words(seg: dict[str, Any]) -> list[dict[str, Any]]:
    """Corrige timestamps de palabras que se salen del segmento o van fuera de orden.

    faster-whisper puede devolver timestamps por palabra desalineados (sobre todo
    después de tramos filtrados por VAD), lo que hace que el subtítulo se adelante
    o atrase respecto al audio real. Si detectamos eso, repartimos el tiempo del
    segmento (que sí es confiable) proporcionalmente al largo de cada palabra.
    """
    words = seg.get("words", [])
    if not words:
        return words

    seg_start, seg_end = seg["start"], seg["end"]
    degenerate = False
    prev_end = seg_start
    for w in words:
        if (
            w["end"] <= w["start"]
            or w["start"] < seg_start - 0.05
            or w["end"] > seg_end + 0.05
            or w["start"] < prev_end - 0.2
        ):
            degenerate = True
            break
        prev_end = w["end"]

    if not degenerate:
        return words

    span = max(0.05, seg_end - seg_start)
    total_chars = sum(len(w["word"]) for w in words) or len(words)
    fixed: list[dict[str, Any]] = []
    t = seg_start
    for w in words:
        share = len(w["word"]) / total_chars if total_chars else 1 / len(words)
        dur = span * share
        fixed.append({"start": round(t, 3), "end": round(t + dur, 3), "word": w["word"]})
        t += dur
    return fixed


def _words_in_window(segments: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for seg in segments:
        for w in _sanitize_words(seg):
            if w["end"] <= start or w["start"] >= end:
                continue
            words.append(w)
    words.sort(key=lambda w: w["start"])
    return words


_MAX_GAP_SECONDS = 0.7  # una pausa más larga que esto corta el grupo de subtítulo


def _group_words(words: list[dict[str, Any]], per_caption: int) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for w in words:
        if current and w["start"] - current[-1]["end"] > _MAX_GAP_SECONDS:
            groups.append(current)
            current = []
        current.append(w)
        ends_sentence = w["word"].strip().endswith((".", "!", "?"))
        if len(current) >= per_caption or ends_sentence:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


_HIGHLIGHT_COLOR = "&H00FFFF&"  # amarillo (formato ASS &HBBGGRR&) — default si no se elige otro

# Paleta chica a propósito (no un selector de color libre): son las opciones
# que se exponen en la interfaz. "naranja" es el color de marca de Prende.
SUBTITLE_COLOR_PRESETS = {
    "amarillo": "&H00FFFF&",
    "naranja": "&H1F5AFF&",
    "blanco": "&HFFFFFF&",
    "verde": "&H4DE87A&",
}


def build_clip_subtitles(
    segments: list[dict[str, Any]],
    parts: list[tuple[float, float]],
    out_path: Path,
    video_width: int | None = None,
    video_height: int | None = None,
    highlight_color: str = _HIGHLIGHT_COLOR,
) -> Path:
    """Genera un .ass con los subtítulos del clip a partir de una o dos partes
    (rangos no contiguos del video original que se pegan con un corte seco).
    Cada parte se agrupa por separado (nunca se mezclan palabras de una parte
    con la siguiente en un mismo subtítulo) y se reubica en el tiempo según su
    posición en el clip final ya concatenado.

    Estilo "karaoke": dentro de cada grupo de palabras se ve el grupo entero
    fijo, pero la palabra que se está diciendo en ese instante queda resaltada
    — un Dialogue por palabra (no uno por grupo), todos con el mismo texto de
    grupo pero cambiando cuál palabra lleva el color de resaltado.
    """
    width = video_width or settings.output_width
    height = video_height or settings.output_height
    margin_v = int(height * 0.12)

    header = ASS_HEADER.format(
        width=width,
        height=height,
        font=settings.subtitle_font,
        font_size=settings.subtitle_font_size,
        outline=max(2, settings.subtitle_font_size // 20),
        margin_v=margin_v,
    )

    lines = [header]
    offset = 0.0
    for part_start, part_end in parts:
        words = _words_in_window(segments, part_start, part_end)
        groups = _group_words(words, settings.words_per_caption)
        for group in groups:
            tokens = [w["word"].strip().upper() for w in group]
            n = len(group)
            for i, w in enumerate(group):
                w_start = offset + max(0.0, w["start"] - part_start)
                if i + 1 < n:
                    # Sin hueco entre palabras: el resaltado de la próxima
                    # arranca justo donde termina esta, no en su timestamp
                    # real (que suele tener micro-gaps que generan parpadeo).
                    w_end = offset + max(w_start - offset + 0.02, group[i + 1]["start"] - part_start)
                else:
                    w_end = offset + max(w_start - offset + 0.05, w["end"] - part_start)

                rendered = [
                    f"{{\\c{highlight_color}}}{t}{{\\r}}" if j == i else t
                    for j, t in enumerate(tokens)
                ]
                text = " ".join(rendered)
                lines.append(
                    f"Dialogue: 0,{_fmt_ts(w_start)},{_fmt_ts(w_end)},Default,,0,0,0,,{text}\n"
                )
        offset += part_end - part_start

    out_path.write_text("".join(lines), encoding="utf-8")
    return out_path
