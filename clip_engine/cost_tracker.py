"""Trackea el costo real en USD de la API de OpenAI gastado por video
procesado (selección de clips, ranking, análisis visual) — para tener el
número real de cuánto cuesta procesar cada uno, no una estimación a ojo.

Se guarda junto a los demás datos del video (data/transcripts/<stem>.cost.json)
así queda asociado al mismo video y sobrevive entre corridas.

La transcripción (faster-whisper) corre local — no tiene costo de API, se
reporta en $0 explícitamente para que quede claro que no es un olvido.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import TRANSCRIPTS_DIR

# USD por 1M tokens (precios de lista de OpenAI). Un modelo no listado acá
# no se puede costear con precisión — se guardan los tokens crudos igual,
# pero el total en $ para esa llamada puntual queda excluido en vez de
# inventar un número con un precio que podría estar desactualizado.
_PRICING_PER_1M_TOKENS = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
}


def _cost_path(video_stem: str) -> Path:
    return TRANSCRIPTS_DIR / f"{video_stem}.cost.json"


def record_usage(video_stem: str, label: str, model: str, usage: Any) -> None:
    """Agrega una llamada al historial de costo de este video. `usage` es el
    objeto `.usage` que devuelve la respuesta de la API de OpenAI."""
    if usage is None:
        return
    path = _cost_path(video_stem)
    entries = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    entries.append({
        "label": label,
        "model": model,
        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
    })
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def cost_summary(video_stem: str) -> dict[str, Any]:
    """Total en USD gastado en llamadas a OpenAI para este video, desglosado
    por etapa. `all_priced=False` si hubo llamadas con un modelo sin precio
    conocido (el total sigue siendo válido, solo que parcial)."""
    path = _cost_path(video_stem)
    entries = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    total_usd = 0.0
    all_priced = True
    by_stage: dict[str, float] = {}
    for e in entries:
        pricing = _PRICING_PER_1M_TOKENS.get(e["model"])
        if pricing is None:
            all_priced = False
            continue
        cost = (e["input_tokens"] / 1e6) * pricing["input"] + (e["output_tokens"] / 1e6) * pricing["output"]
        total_usd += cost
        by_stage[e["label"]] = by_stage.get(e["label"], 0.0) + cost

    return {
        "total_usd": round(total_usd, 4),
        "by_stage_usd": {k: round(v, 4) for k, v in by_stage.items()},
        "api_calls": len(entries),
        "all_priced": all_priced,
        "transcription_usd": 0.0,
    }
