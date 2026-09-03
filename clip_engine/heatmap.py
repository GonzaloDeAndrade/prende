"""Heatmap de "más repetido" de YouTube como señal de selección.

Encontrado por investigación de competencia (2026-08-21): Opus Clip usa
exactamente esta señal como mecanismo PRINCIPAL de selección — "el agente
lee el heatmap de más repetido de YouTube y corta justo los picos que
muestra... la señal de audiencia decide qué se corta, el puntaje de IA
solo desempata cuando hay superposición". No es una idea nuestra, es
metodología ya probada por el líder del mercado.

También validado en vivo esta misma noche: dos momentos que el pipeline se
perdió por completo (uno marcado a mano por el usuario, otro encontrado
cruzando datos) resultaron ser, medido con este mismo heatmap, de los
tramos más repetidos de sus videos — la señal es real, no teórica.

El video_stem de algo bajado de YouTube ES el video ID (yt-dlp lo nombra
así) — no hace falta guardar la URL original aparte, se reconstruye.
Para archivos subidos a mano (no vienen de YouTube), el fetch simplemente
falla y devuelve None — degradación silenciosa, no rompe nada.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import TRANSCRIPTS_DIR


def _cache_path(video_stem: str) -> Path:
    return TRANSCRIPTS_DIR / f"{video_stem}.heatmap.json"


def get_heatmap(video_stem: str, force: bool = False) -> list[dict[str, Any]] | None:
    """Devuelve los puntos del heatmap de "más repetido" para un video_stem
    que sea un ID de YouTube válido, o None si no aplica (archivo subido a
    mano, video sin suficientes vistas para tener heatmap, o cualquier
    error de red) — nunca lanza excepción, esto es una señal opcional, no
    algo de lo que el pipeline dependa para funcionar.
    """
    import json

    cache = _cache_path(video_stem)
    if cache.exists() and not force:
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            return data if data else None
        except (json.JSONDecodeError, OSError):
            pass

    try:
        import truststore
        truststore.inject_into_ssl()
        import yt_dlp
        from yt_dlp_helper import NODE22_PATH

        # Mismo fix que usa el resto del proyecto (yt_dlp_helper.YT_DLP_FIX_ARGS),
        # pero armado como dict de Python en vez de argv de CLI — acá se llama a
        # la librería directo, no por subprocess.
        ydl_opts = {
            "js_runtimes": {"node": {"path": str(NODE22_PATH)}},
            "extractor_args": {"youtube": {"player_client": ["android", "tv"]}},
            "quiet": True,
            "skip_download": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_stem}", download=False)
            hm = info.get("heatmap")
    except Exception:  # noqa: BLE001 — señal opcional, cualquier fallo degrada a None
        hm = None

    cache.write_text(json.dumps(hm or [], ensure_ascii=False, indent=2), encoding="utf-8")
    return hm or None
