"""Base de datos estructurada (SQLite) de features por clip — la "Parte 0"
de las mejoras. Separado a propósito de research_batch.py: solo LEE los
JSON que el daemon ya genera (transcripción, energía, visual, heatmap) y
los organiza en una tabla consultable. No toca el daemon ni sus archivos —
si algo acá tiene un bug, en el peor caso se arregla y se vuelve a
sincronizar desde los JSON crudos, que son la fuente de verdad real y no
se tocan.

Uso: python research_db.py [--sync] [--stats]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from statistics import mean
from typing import Any

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
RESEARCH_DIR = ROOT / "data" / "research"
TRANSCRIPTS_DIR = RESEARCH_DIR / "transcripts"
MANIFEST_PATH = RESEARCH_DIR / "manifest.json"
DB_PATH = RESEARCH_DIR / "features.db"

_SENTENCE_END = (".", "!", "?", "...", "¡", "¿")

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,                      -- 'research_clip' | 'research_largo'
    source TEXT,                             -- 'Clipeados', 'original', etc.
    url TEXT,
    title TEXT,
    category TEXT,
    duration REAL,
    view_count INTEGER,
    like_count INTEGER,
    channel TEXT,

    -- señal de audio (picos de energía/volumen)
    audio_peak_z REAL,                       -- z-score del pico mas fuerte detectado
    audio_peak_position REAL,                -- posicion relativa (0-1) del pico mas fuerte dentro del video
    audio_spike_count INTEGER,

    -- señal visual (momentos notables)
    visual_notable_count INTEGER,
    visual_top_hook_type TEXT,               -- hook_type mas frecuente entre los momentos notables
    visual_top_score INTEGER,
    visual_top_position REAL,                -- posicion relativa (0-1) del momento visual mas fuerte

    -- heatmap real de YouTube ("mas reproducido"), si esta disponible
    heatmap_peak_position REAL,              -- 0-1
    heatmap_peak_value REAL,

    -- texto
    word_count INTEGER,
    ends_with_question_or_exclaim INTEGER,   -- bool 0/1, ultima linea del video

    -- estructura interna (solo tiene sentido para kind='research_clip': un
    -- clip YA CORTADO y publicado, no un video largo) — mismas metricas que
    -- el analisis ad-hoc del 2026-08-20 (duracion, arranque, cierre, ritmo),
    -- ahora reproducibles y desglosables por hook_type en vez de un numero
    -- agregado suelto.
    silence_before_first_word REAL,          -- segundos de aire antes de la primera palabra
    median_internal_gap REAL,                -- mediana de huecos entre palabras consecutivas
    p75_internal_gap REAL,
    opening_text TEXT,                       -- primeras ~8 palabras, para inspeccion manual del patron de arranque
    closing_text TEXT,                       -- ultimas ~8 palabras, patron de cierre

    -- clasificado por LLM (gpt-4o-mini) contra la misma taxonomia de 6
    -- hook_types que usa el pipeline real (RANKING_SYSTEM_PROMPT /
    -- SYSTEM_PROMPT en clip_engine/prompts.py) — permite comparar patrones
    -- estructurales POR TIPO DE GANCHO, no solo agregados, que es lo que
    -- el mandato del loop pide explicitamente.
    hook_type TEXT,

    -- resultado, solo aplica cuando esto se use sobre candidatos de la app
    -- (no sobre el corpus de referencia externo, que no tiene "elegido/descartado")
    app_status TEXT,
    app_score INTEGER,

    updated_at TEXT
);
"""


_NEW_COLUMNS = {
    "silence_before_first_word": "REAL",
    "median_internal_gap": "REAL",
    "p75_internal_gap": "REAL",
    "opening_text": "TEXT",
    "closing_text": "TEXT",
    "hook_type": "TEXT",
}


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    # `CREATE TABLE IF NOT EXISTS` no agrega columnas nuevas a una tabla ya
    # existente de una version anterior del schema — migracion manual simple,
    # se puede correr las veces que haga falta (ALTER TABLE ... ADD COLUMN
    # es un no-op seguro de reintentar via el try/except de abajo).
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(items)")}
    for col, col_type in _NEW_COLUMNS.items():
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE items ADD COLUMN {col} {col_type}")
    return conn


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _text_features(transcript: dict | None) -> dict[str, Any]:
    if not transcript or not transcript.get("segments"):
        return {"word_count": None, "ends_with_question_or_exclaim": None}
    segments = transcript["segments"]
    word_count = sum(len(seg.get("text", "").split()) for seg in segments)
    last_text = segments[-1].get("text", "").strip()
    ends_qe = int(last_text.endswith(("?", "!")))
    return {"word_count": word_count, "ends_with_question_or_exclaim": ends_qe}


def _structure_features(transcript: dict | None, kind: str) -> dict[str, Any]:
    """Silencio antes de arrancar, ritmo interno (huecos entre palabras) y
    texto de apertura/cierre — solo tiene sentido medirlo sobre clips YA
    CORTADOS (kind='research_clip'): en un video largo el "silencio antes de
    la primera palabra" no significa nada (depende de donde arranca la
    grabacion, no de una decision de edicion)."""
    empty = {
        "silence_before_first_word": None, "median_internal_gap": None,
        "p75_internal_gap": None, "opening_text": None, "closing_text": None,
    }
    if kind != "research_clip" or not transcript or not transcript.get("segments"):
        return empty

    words = [w for seg in transcript["segments"] for w in seg.get("words", [])]
    if len(words) < 2:
        return empty

    gaps = sorted(words[i]["start"] - words[i - 1]["end"] for i in range(1, len(words)))
    gaps = [g for g in gaps if g > 0]
    median_gap = gaps[len(gaps) // 2] if gaps else None
    p75_gap = gaps[int(len(gaps) * 0.75)] if gaps else None

    opening = "".join(w["word"] for w in words[:8]).strip()
    closing = "".join(w["word"] for w in words[-8:]).strip()

    return {
        "silence_before_first_word": words[0]["start"],
        "median_internal_gap": median_gap,
        "p75_internal_gap": p75_gap,
        "opening_text": opening,
        "closing_text": closing,
    }


def _audio_features(spikes: list | None, duration: float | None) -> dict[str, Any]:
    if not spikes:
        return {"audio_peak_z": None, "audio_peak_position": None, "audio_spike_count": 0}
    # cada spike es [start, end, z]
    top = max(spikes, key=lambda s: s[2])
    mid = (top[0] + top[1]) / 2
    pos = (mid / duration) if duration else None
    return {"audio_peak_z": top[2], "audio_peak_position": pos, "audio_spike_count": len(spikes)}


def _visual_features(moments: list | None, duration: float | None) -> dict[str, Any]:
    if not moments:
        return {
            "visual_notable_count": 0, "visual_top_hook_type": None,
            "visual_top_score": None, "visual_top_position": None,
        }
    top = max(moments, key=lambda m: m.get("score", 0))
    mid = (top["start"] + top["end"]) / 2
    pos = (mid / duration) if duration else None
    hook_types = [m.get("hook_type") for m in moments if m.get("hook_type")]
    top_hook = max(set(hook_types), key=hook_types.count) if hook_types else None
    return {
        "visual_notable_count": len(moments), "visual_top_hook_type": top_hook,
        "visual_top_score": top.get("score"), "visual_top_position": pos,
    }


def _heatmap_features(heatmap: list | None, duration: float | None) -> dict[str, Any]:
    if not heatmap:
        return {"heatmap_peak_position": None, "heatmap_peak_value": None}
    top = max(heatmap, key=lambda h: h.get("value", 0))
    mid = (top["start_time"] + top["end_time"]) / 2
    pos = (mid / duration) if duration else None
    return {"heatmap_peak_position": pos, "heatmap_peak_value": top.get("value")}


def sync() -> tuple[int, int]:
    """Lee el manifest + los JSON de cada item 'listo' y los vuelca a la DB.
    Devuelve (sincronizados, saltados_sin_datos)."""
    import time

    manifest = _load_json(MANIFEST_PATH) or {"clips": [], "largos": []}
    conn = _connect()
    synced = skipped = 0

    for bucket, kind in (("clips", "research_clip"), ("largos", "research_largo")):
        for item in manifest.get(bucket, []):
            if item.get("status") != "listo":
                continue
            item_id = item["id"]
            duration = item.get("duration")

            transcript = _load_json(TRANSCRIPTS_DIR / f"{item_id}.json")
            energy = _load_json(TRANSCRIPTS_DIR / f"{item_id}.energy.json")
            visual = _load_json(TRANSCRIPTS_DIR / f"{item_id}.visual.json")
            heatmap = item.get("heatmap")

            if transcript is None:
                skipped += 1
                continue

            row = {
                "id": item_id, "kind": kind, "source": item.get("source", "original"),
                "url": item.get("url"), "title": item.get("title"), "category": item.get("category"),
                "duration": duration, "view_count": item.get("view_count"),
                "like_count": item.get("like_count"), "channel": item.get("channel"),
                "app_status": None, "app_score": None,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            row.update(_text_features(transcript))
            row.update(_audio_features(energy, duration))
            row.update(_visual_features(visual, duration))
            row.update(_heatmap_features(heatmap, duration))
            row.update(_structure_features(transcript, kind))

            # No pisar un hook_type ya clasificado por --classify en una
            # corrida anterior: sync() se re-corre seguido (el daemon sigue
            # bajando videos nuevos) y no deberia tirar trabajo de LLM ya
            # pagado solo porque se volvio a sincronizar.
            cols = [k for k in row if k != "hook_type"]
            cols_sql = ", ".join(cols)
            placeholders = ", ".join(f":{k}" for k in cols)
            updates = ", ".join(f"{k}=excluded.{k}" for k in cols if k != "id")
            conn.execute(
                f"INSERT INTO items ({cols_sql}) VALUES ({placeholders}) "
                f"ON CONFLICT(id) DO UPDATE SET {updates}",
                row,
            )
            synced += 1

    conn.commit()
    conn.close()
    return synced, skipped


_HOOK_TYPES = [
    "hook_fuerte", "dato_sorprendente", "emocional",
    "gracioso", "consejo_practico", "debate_polemica",
]

_CLASSIFY_SYSTEM_PROMPT = """Clasificás la transcripción de un clip corto ya \
publicado (TikTok/Reels/Shorts, de habla hispana LatAm) en UNA sola categoría \
de gancho principal, la misma taxonomía que usa nuestro pipeline de selección:

- hook_fuerte: tensión, confrontación, algo picante o polémico que engancha por sí solo.
- dato_sorprendente: un número, hecho o revelación que sorprende (y de verdad \
llega a sorprender, no solo se nombra).
- emocional: vulnerabilidad, historia personal fuerte, enojo genuino, orgullo.
- gracioso: momento cómico, chiste, frase muy citable/quotable.
- consejo_practico: un consejo o explicación práctica y accionable.
- debate_polemica: discusión, ida y vuelta, opiniones enfrentadas.

Devolvé SOLO un JSON: {"hook_type": "<una de las 6 categorías exactas>"}."""


def classify_hook_types(limit: int | None = None) -> tuple[int, float]:
    """Clasifica con gpt-4o-mini los clips (kind='research_clip') que todavía
    no tienen hook_type. Barato (texto corto, modelo mini) — ~1500 clips
    salen por menos de 10 centavos de dólar al precio de lista. Devuelve
    (clasificados, costo_usd_total)."""
    import json as _json

    from openai import OpenAI

    from clip_engine.config import settings
    from clip_engine.cost_tracker import _PRICING_PER_1M_TOKENS
    from clip_engine.openai_retry import call_with_backoff

    client = OpenAI(api_key=settings.openai_api_key)
    model = settings.openai_model

    conn = _connect()
    query = "SELECT id FROM items WHERE kind='research_clip' AND hook_type IS NULL"
    if limit:
        query += f" LIMIT {int(limit)}"
    ids = [row[0] for row in conn.execute(query).fetchall()]

    classified = 0
    cost_usd = 0.0
    pricing = _PRICING_PER_1M_TOKENS.get(model)

    for item_id in ids:
        transcript = _load_json(TRANSCRIPTS_DIR / f"{item_id}.json")
        if not transcript or not transcript.get("segments"):
            continue
        text = " ".join(seg.get("text", "") for seg in transcript["segments"]).strip()
        if not text:
            continue

        def _call():
            return client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _CLASSIFY_SYSTEM_PROMPT},
                    {"role": "user", "content": text[:2000]},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )

        try:
            resp = call_with_backoff(_call, label=f"classify:{item_id}")
        except RuntimeError as exc:
            # Un solo item con fallo persistente (rate-limit sostenido, p.ej.
            # compitiendo con el daemon corriendo en paralelo) no debería
            # tirar abajo TODO el batch y perder el progreso ya commiteado
            # del resto — se saltea este item puntual (queda hook_type=NULL,
            # se reintenta solo en la proxima corrida) y se sigue con los demas.
            print(f"  {item_id}: {exc} — salteado, sigue el resto")
            continue
        try:
            hook_type = _json.loads(resp.choices[0].message.content).get("hook_type")
        except (ValueError, AttributeError):
            hook_type = None
        if hook_type not in _HOOK_TYPES:
            hook_type = None

        conn.execute(
            "UPDATE items SET hook_type=? WHERE id=?", (hook_type, item_id)
        )
        classified += 1
        if pricing and resp.usage:
            cost_usd += (resp.usage.prompt_tokens / 1e6) * pricing["input"]
            cost_usd += (resp.usage.completion_tokens / 1e6) * pricing["output"]

        if classified % 100 == 0:
            conn.commit()
            print(f"  ...{classified}/{len(ids)} clasificados (${cost_usd:.4f})")

    conn.commit()
    conn.close()
    return classified, cost_usd


_MAX_PLAUSIBLE_CLIP_DURATION = 180
# El bucket "clips" del manifest (contenido ya cortado y publicado) tiene
# ~11-16 items (de 1577) que en realidad son streams completos colados sin
# querer, no clips editados — encontrado en vivo (2026-08-22) al ver que la
# categoria "emocional" promediaba 91.9s de duracion, mas de 4 veces
# cualquier otra categoria: el caso mas extremo era un item de 19580s (5.4
# horas, titulo generico "(nuevo) <id>", sin metadata real de titulo — señal
# clara de que no paso por el proceso normal de un clip editado). Un solo
# outlier de horas alcanza para arruinar el promedio de una categoria con
# pocos cientos de items. Se excluyen de este desglose estructural (no de la
# DB entera — siguen ahi por si sirven para otra cosa) los que superan un
# techo generoso para contenido de clip real.
def breakdown_by_hook_type() -> None:
    """El desglose que pide el mandato: no un numero agregado del corpus
    entero, sino duracion/arranque/ritmo separados POR TIPO DE GANCHO —
    porque un remate gracioso y uno con un dato sorprendente no se
    estructuran igual."""
    conn = _connect()
    cur = conn.execute(
        f"""SELECT hook_type, COUNT(*),
               AVG(duration), AVG(silence_before_first_word),
               AVG(median_internal_gap), AVG(p75_internal_gap),
               SUM(ends_with_question_or_exclaim) * 1.0 / COUNT(*)
           FROM items
           WHERE kind='research_clip' AND hook_type IS NOT NULL
               AND duration < {_MAX_PLAUSIBLE_CLIP_DURATION}
           GROUP BY hook_type
           ORDER BY COUNT(*) DESC"""
    )
    rows = cur.fetchall()
    if not rows:
        print("Todavia no hay clips clasificados — corré con --classify primero.")
        conn.close()
        return

    print(f"{'hook_type':<20} {'n':>5} {'dur_avg':>8} {'silencio_ini':>13} {'gap_mediana':>12} {'gap_p75':>9} {'%cierra_?!':>11}")
    for hook_type, n, dur, silence, med_gap, p75_gap, pct_qe in rows:
        print(
            f"{hook_type:<20} {n:>5} {dur or 0:>8.1f} {silence or 0:>13.2f} "
            f"{med_gap or 0:>12.2f} {p75_gap or 0:>9.2f} {(pct_qe or 0) * 100:>10.1f}%"
        )

    # Ejemplos reales de apertura/cierre por tipo, para inspeccionar el
    # patron a mano (los promedios solos no dicen COMO arranca/cierra).
    print("\nEjemplos de apertura por hook_type:")
    for hook_type, *_ in rows:
        cur = conn.execute(
            "SELECT opening_text FROM items WHERE hook_type=? AND opening_text IS NOT NULL LIMIT 3",
            (hook_type,),
        )
        examples = [r[0] for r in cur.fetchall()]
        print(f"  {hook_type}:")
        for ex in examples:
            print(f"    - \"{ex}...\"")

    conn.close()


def stats() -> None:
    conn = _connect()
    cur = conn.execute("SELECT COUNT(*), source FROM items GROUP BY source")
    print("Items en la base por fuente:")
    for count, source in cur.fetchall():
        print(f"  {source or '(sin fuente)'}: {count}")

    cur = conn.execute("SELECT COUNT(*) FROM items WHERE heatmap_peak_position IS NOT NULL")
    print(f"\nCon heatmap real de YouTube: {cur.fetchone()[0]}")

    cur = conn.execute("SELECT COUNT(*) FROM items WHERE audio_peak_position IS NOT NULL")
    print(f"Con pico de audio detectado: {cur.fetchone()[0]}")

    cur = conn.execute(
        "SELECT audio_peak_position, heatmap_peak_position FROM items "
        "WHERE audio_peak_position IS NOT NULL AND heatmap_peak_position IS NOT NULL"
    )
    pairs = cur.fetchall()
    if pairs:
        diffs = [abs(a - h) for a, h in pairs]
        print(f"\n{len(pairs)} items con AMBOS audio+heatmap — diferencia promedio de posición: {mean(diffs):.2f} (0=coinciden exacto, 1=en extremos opuestos)")
    else:
        print("\nTodavía no hay items con audio+heatmap juntos para comparar.")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--classify", action="store_true", help="clasifica hook_type de clips sin clasificar (gasta API, barato)")
    parser.add_argument("--classify-limit", type=int, default=None)
    parser.add_argument("--breakdown", action="store_true", help="patrones estructurales agrupados por hook_type")
    args = parser.parse_args()

    ran_something = args.sync or args.stats or args.classify or args.breakdown

    if args.sync or not ran_something:
        synced, skipped = sync()
        print(f"Sincronizados: {synced}, saltados (sin transcript): {skipped}")
    if args.classify:
        n, cost = classify_hook_types(limit=args.classify_limit)
        print(f"Clasificados: {n}, costo real: ${cost:.4f}")
    if args.stats:
        stats()
    if args.breakdown:
        breakdown_by_hook_type()
