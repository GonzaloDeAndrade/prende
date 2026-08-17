"""Selección de los mejores clips a partir de la transcripción, vía LLM (OpenAI).

El LLM elige uno o dos rangos de líneas ("parts") que arman una idea coherente
y autocontenida — es una tarea de criterio narrativo que resuelve bien. Lo que
NO hace es aritmética de duración (pedirle que sume segundos hasta llegar a un
rango de X-Y segundos es poco confiable incluso con modelos grandes). Por eso
este módulo solo ajusta esos rangos para que caigan dentro de los límites de
duración configurados, y arma el corte no-continuo (dos partes con un salto
seco en el medio) cuando el LLM decide que una idea se retoma después de una
interrupción, sin tocar el criterio de dónde arranca y dónde cierra cada parte.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from openai import OpenAI

from .audio_energy import detect_energy_spikes
from .config import TRANSCRIPTS_DIR, settings
from .visual_analyze import analyze_visual
from .prompts import (
    CATEGORY_ADDENDUMS,
    CLIP_JSON_SCHEMA,
    RANKING_JSON_SCHEMA,
    RANKING_SYSTEM_PROMPT,
    RANKING_USER_TEMPLATE,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)


_MAX_STITCH_GAP_SECONDS = 240.0  # distancia máxima entre partes de un mismo clip


def _energy_tag(seg: dict[str, Any], energy_spikes: list[tuple[float, float, float]]) -> str:
    for start, end, z in energy_spikes:
        if seg["start"] < end and seg["end"] > start:
            return " (🔊 pico de volumen/risa fuerte acá)" if z < 4 else " (🔊🔊 pico de volumen/risa MUY fuerte acá)"
    return ""


def _numbered_transcript(
    segments: list[dict[str, Any]], energy_spikes: list[tuple[float, float, float]] | None = None,
) -> str:
    energy_spikes = energy_spikes or []
    lines = [
        f"[{i} | {seg['start']:.1f}s -> {seg['end']:.1f}s]{_energy_tag(seg, energy_spikes)} {seg['text']}"
        for i, seg in enumerate(segments)
    ]
    return "\n".join(lines)


_SENTENCE_END = (".", "!", "?", "...")
_SNAP_GAP_BREAK = 1.2  # una pausa más larga que esto probablemente es cambio de tema
_SNAP_MAX_EXTRA = 15.0  # segundos máximos que se extiende buscando cerrar la oración


def _ends_sentence(text: str) -> bool:
    return text.strip().endswith(_SENTENCE_END)


_DANGLING_REF_WORDS = {
    "esto", "eso", "esos", "estos", "esta", "estas", "este", "ese", "esa",
    "dicho", "dicha", "dichos", "dichas", "tal", "tales", "aquello", "aquellos", "aquellas",
}


def _starts_with_dangling_ref(text: str) -> bool:
    """Detecta si una línea arranca dependiendo de un pronombre/referencia sin
    nombrar el sustantivo (p. ej. "estos objetos", "eso", "dicho experimento").
    """
    words = [w.strip(".,;:¿?¡!()\"'").lower() for w in text.split()[:4]]
    return any(w in _DANGLING_REF_WORDS for w in words)


def _extend_lo_while(
    segments: list[dict[str, Any]], lo: int, should_extend,
) -> int:
    extra = 0.0
    while lo > 0 and should_extend(lo):
        gap = segments[lo]["start"] - segments[lo - 1]["end"]
        added = segments[lo]["start"] - segments[lo - 1]["start"]
        if gap > _SNAP_GAP_BREAK or extra + added > _SNAP_MAX_EXTRA:
            break
        lo -= 1
        extra += added
    return lo


def _snap_part(segments: list[dict[str, Any]], start_index: int, end_index: int) -> tuple[int, int]:
    """Clampea un tramo a los límites válidos y estira sus bordes hasta el
    cierre real de una oración (.!?...), sin cruzar pausas largas (probable
    cambio de tema) ni pasarse de un extra razonable de segundos. También
    retrocede el arranque si depende de un pronombre/referencia sin nombrar
    (el LLM tiene esta regla en el prompt pero a veces igual se la salta).
    """
    n = len(segments)
    lo, hi = sorted((start_index, end_index))
    lo = max(0, min(lo, n - 1))
    hi = max(0, min(hi, n - 1))

    extra = 0.0
    while hi < n - 1 and not _ends_sentence(segments[hi]["text"]):
        gap = segments[hi + 1]["start"] - segments[hi]["end"]
        added = segments[hi + 1]["end"] - segments[hi]["end"]
        if gap > _SNAP_GAP_BREAK or extra + added > _SNAP_MAX_EXTRA:
            break
        hi += 1
        extra += added

    # Cierre de oración del arranque, después referencia colgada, y volvemos a
    # alinear al cierre de oración por si el paso anterior lo desalineó.
    lo = _extend_lo_while(segments, lo, lambda l: not _ends_sentence(segments[l - 1]["text"]))
    lo = _extend_lo_while(segments, lo, lambda l: _starts_with_dangling_ref(segments[l]["text"]))
    lo = _extend_lo_while(segments, lo, lambda l: not _ends_sentence(segments[l - 1]["text"]))

    return lo, hi


def _fit_clip(
    segments: list[dict[str, Any]],
    raw_parts: list[dict[str, int]],
    min_dur: float,
    max_dur: float,
    duration: float,
) -> list[tuple[float, float]] | None:
    """Ajusta las partes (1 o 2) de un clip a los límites de duración
    configurados. Devuelve una lista ordenada de ventanas (start, end) en
    segundos, listas para cortar y concatenar.
    """
    n = len(segments)
    if n == 0 or not raw_parts:
        return None

    parts = [list(_snap_part(segments, p["start_index"], p["end_index"])) for p in raw_parts]
    parts.sort(key=lambda lohi: lohi[0])

    # Si el ajuste a oración hizo que dos partes queden pegadas o superpuestas,
    # las fusionamos en una sola (ya no hay salto que valga la pena marcar).
    merged: list[list[int]] = [parts[0]]
    for lo, hi in parts[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    parts = merged

    def total_dur() -> float:
        return sum(segments[hi]["end"] - segments[lo]["start"] for lo, hi in parts)

    # Si el conjunto quedó corto, estiramos los bordes externos (antes de la
    # primera parte, después de la última), priorizando contexto previo.
    step = 0
    while total_dur() < min_dur and (parts[0][0] > 0 or parts[-1][1] < n - 1):
        want_back = step % 3 != 2
        if want_back and parts[0][0] > 0:
            parts[0][0] -= 1
        elif parts[-1][1] < n - 1:
            parts[-1][1] += 1
        elif parts[0][0] > 0:
            parts[0][0] -= 1
        else:
            break
        step += 1

    # Si se pasa del techo duro, recortamos desde los bordes externos hacia
    # adentro (última parte primero), sin vaciar ninguna parte por completo.
    hard_max = max_dur * 1.25
    trimmed_hard_max = False
    while total_dur() > hard_max:
        last_lo, last_hi = parts[-1]
        if last_hi > last_lo:
            parts[-1][1] -= 1
            trimmed_hard_max = True
            continue
        first_lo, first_hi = parts[0]
        if first_hi > first_lo:
            parts[0][0] += 1
            trimmed_hard_max = True
            continue
        break

    # Última red: el recorte de arriba no sabe de puntuación, solo cuenta
    # segmentos — puede deshacer justo el cierre de oración que el ajuste
    # anterior había logrado. Solo tiene sentido correr esto si el recorte
    # de arriba de verdad se ejecutó: si no, el borde ya es el que dejó el
    # snap original, y retroceder acá buscando puntuación es peligroso —
    # en transcripciones sin puntuación confiable (diálogo rápido/coloquial,
    # típico de streams/IRL) ningún segmento cierra con punto, y este bucle
    # se come el clip entero hasta casi no dejar nada.
    if trimmed_hard_max:
        last_lo, last_hi = parts[-1]
        while last_hi > last_lo and not _ends_sentence(segments[last_hi]["text"]):
            last_hi -= 1
        parts[-1][1] = last_hi

        first_lo, first_hi = parts[0]
        while first_lo < first_hi and first_lo > 0 and not _ends_sentence(segments[first_lo - 1]["text"]):
            first_lo += 1
        parts[0][0] = first_lo

    windows = [(segments[lo]["start"], segments[hi]["end"]) for lo, hi in parts]

    total = sum(e - s for s, e in windows)
    if total < min_dur:
        deficit = min_dur - total
        if len(windows) == 1:
            s0, e0 = windows[0]
            windows[0] = (max(0.0, s0 - deficit / 2), min(duration, e0 + deficit / 2))
        else:
            s0, e0 = windows[0]
            s1, e1 = windows[-1]
            windows[0] = (max(0.0, s0 - deficit / 2), e0)
            windows[-1] = (s1, min(duration, e1 + deficit / 2))

    return [(round(s, 3), round(e, 3)) for s, e in windows]


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _quote_matches(segment_text: str, quote: str) -> bool:
    norm_seg = _normalize_text(segment_text)
    norm_quote = _normalize_text(quote)
    if not norm_quote or not norm_seg:
        return False
    return norm_quote in norm_seg or norm_seg in norm_quote


def _resolve_index(segments: list[dict[str, Any]], claimed_index: int, quote: str) -> int | None:
    """Contar índices en una transcripción de miles de líneas es justo el tipo
    de tarea donde el LLM se equivoca (visto en la práctica: apunta a un
    índice que no tiene nada que ver con el texto que él mismo describió).
    Por eso el LLM copia el texto exacto de la línea junto con el índice, y
    acá lo verificamos: si el índice reclamado no contiene ese texto, lo
    buscamos en el resto de la transcripción y corregimos. Si no aparece en
    ningún lado, es una alucinación y descartamos el candidato en vez de
    confiar en un índice que puede estar apuntando a cualquier cosa.
    """
    n = len(segments)
    if n == 0:
        return None
    claimed_index = max(0, min(claimed_index, n - 1))
    if _quote_matches(segments[claimed_index]["text"], quote):
        return claimed_index

    for radius in (5, 30, 120):
        lo = max(0, claimed_index - radius)
        hi = min(n, claimed_index + radius + 1)
        for i in range(lo, hi):
            if _quote_matches(segments[i]["text"], quote):
                return i

    for i, seg in enumerate(segments):
        if _quote_matches(seg["text"], quote):
            return i

    return None


def _bridge_is_weak(bridge: str) -> bool:
    """No confiamos ciegamente en que el LLM haya respetado la instrucción de
    solo usar 2 partes cuando corresponde, pero verificar la conexión semántica
    con un heurístico de texto es frágil (el `bridge` es un parafraseo del
    modelo, no una cita textual de la parte A — casi nunca va a compartir
    palabras exactas aunque la conexión sea válida). Acá solo filtramos los
    casos obvios de placeholder/vacío; el juicio semántico real queda para la
    segunda pasada de ranking comparativo, que sí lee el texto completo ya
    concatenado y puede detectar si de verdad no cierra.
    """
    bridge = (bridge or "").strip().lower()
    return len(bridge) < 15 or bridge in {"n/a", "na", "no aplica"}


def _parts_overlap(parts_a: list[tuple[float, float]], parts_b: list[tuple[float, float]]) -> bool:
    return any(not (a_end <= b_start or a_start >= b_end) for a_start, a_end in parts_a for b_start, b_end in parts_b)


def _dedupe_by_score(clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    for c in sorted(clips, key=lambda c: -c["score"]):
        if any(_parts_overlap(c["parts"], a["parts"]) for a in accepted):
            continue
        accepted.append(c)
    accepted.sort(key=lambda c: c["parts"][0][0])
    return accepted


def _clip_text(segments: list[dict[str, Any]], clip: dict[str, Any]) -> str:
    parts_text = []
    for s, e in clip["parts"]:
        text = " ".join(seg["text"] for seg in segments if seg["start"] >= s - 0.5 and seg["end"] <= e + 0.5)
        parts_text.append(text)
    return " [...salto...] ".join(parts_text)


def _validate_ranking(
    client: OpenAI, segments: list[dict[str, Any]], clips: list[dict[str, Any]], category: str = "general",
) -> list[dict[str, Any]]:
    """Segunda pasada: compara los candidatos ENTRE SÍ (la primera pasada
    evalúa cada uno aislado) y filtra los que, puestos al lado de los demás,
    son claramente más flojos — aunque técnicamente encajen en una categoría.

    Recibe el mismo `category` que la primera pasada: si no se le suma el
    addendum acá también, esta segunda pasada juzga con el criterio genérico
    (le exige "desarrollo claro", "remate", etc.) aunque la primera haya sido
    más permisiva para el género — eso deshace en la validación lo que el
    addendum logró en la selección.
    """
    has_multi_part = any(len(c["parts"]) > 1 for c in clips)
    if len(clips) <= settings.min_clips and not has_multi_part:
        # Con pocos candidatos y ninguno de 2 partes no hay nada que comparar
        # ni ningún empalme que verificar — nos salteamos la llamada extra.
        return clips

    listing = "\n\n".join(
        f"[{i}] {c['title']} | {c['hook_type']} | score inicial {c['score']}\n{_clip_text(segments, c)}"
        for i, c in enumerate(clips)
    )
    user = RANKING_USER_TEMPLATE.format(candidates=listing)
    system = RANKING_SYSTEM_PROMPT + CATEGORY_ADDENDUMS.get(category, "")

    print(f"[analyze] validando {len(clips)} candidatos entre sí...")
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_schema", "json_schema": RANKING_JSON_SCHEMA},
        temperature=0.3,
    )
    raw_ranking = json.loads(response.choices[0].message.content).get("ranking", [])
    by_index = {r["index"]: r for r in raw_ranking if 0 <= r["index"] < len(clips)}

    ranked: list[dict[str, Any]] = []
    for i, c in enumerate(clips):
        r = by_index.get(i)
        if r is None:
            ranked.append(c)
            continue
        ranked.append({**c, "score": r["final_score"], "keep": r["keep"], "verdict": r["verdict"]})
    ranked.sort(key=lambda c: -c["score"])

    for c in ranked:
        estado = "OK " if c.get("keep", True) else "OUT"
        if "verdict" in c:
            print(f"  [{estado}] score {c['score']} - {c['title']}: {c['verdict']}")

    # Devolvemos TODOS los candidatos rankeados (con su score/keep/verdict),
    # no solo los que pasaron el filtro — quien llama decide si se queda con
    # los `keep=true` (modo automático) o le muestra la lista completa a un
    # humano para elegir (modo revisión).
    return ranked


def _merge_visual_segments(
    segments: list[dict[str, Any]], visual_moments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Inserta los momentos visuales notables como pseudo-líneas dentro de la
    transcripción, para que el mismo pase de selección vea texto y contexto
    visual juntos. No pisa líneas con diálogo real: si un momento visual cae
    encima de un tramo donde ya hay texto, no hace falta agregarlo aparte
    (esa línea ya entra en juego por su propio contenido).
    """
    if not visual_moments:
        return segments

    merged = list(segments)
    for m in visual_moments:
        overlaps_real_text = any(seg["start"] < m["end"] and seg["end"] > m["start"] for seg in segments)
        if overlaps_real_text:
            continue
        merged.append({
            "start": m["start"],
            "end": m["end"],
            "text": f"(visual, sin diálogo) {m['description'].rstrip('.')}.",
            "words": [],
        })
    merged.sort(key=lambda seg: seg["start"])
    return merged


def _nearest_segment_index(segments: list[dict[str, Any]], t: float) -> int:
    best_i, best_dist = 0, float("inf")
    for i, seg in enumerate(segments):
        dist = abs((seg["start"] + seg["end"]) / 2 - t)
        if dist < best_dist:
            best_i, best_dist = i, dist
    return best_i


_SPIKE_GROUP_GAP = 20.0  # picos separados por menos que esto son la misma reacción


def _group_nearby_spikes(
    energy_spikes: list[tuple[float, float, float]], min_z: float,
) -> list[tuple[float, float, float]]:
    """Una risa o reacción larga puede partirse en varios picos separados
    (baches de silencio en el medio de la carcajada) — sin agrupar, cada uno
    termina generando su propio candidato casi idéntico al de al lado.
    Agrupamos los que están a menos de _SPIKE_GROUP_GAP entre sí en uno solo.
    """
    strong = sorted((s for s in energy_spikes if s[2] >= min_z), key=lambda s: s[0])
    if not strong:
        return []

    groups: list[list[float]] = [[strong[0][0], strong[0][1], strong[0][2]]]
    for start, end, z in strong[1:]:
        if start - groups[-1][1] < _SPIKE_GROUP_GAP:
            groups[-1][1] = max(groups[-1][1], end)
            groups[-1][2] = max(groups[-1][2], z)
        else:
            groups.append([start, end, z])
    return [tuple(g) for g in groups]


def _candidates_from_energy_spikes(
    segments: list[dict[str, Any]],
    energy_spikes: list[tuple[float, float, float]],
    existing_windows: list[list[tuple[float, float]]],
    min_z: float = 2.8,
) -> list[dict[str, Any]]:
    """El primer pase del LLM solo nomina un puñado de candidatos de todo el \
    video (limitado por `max_clips`) — en un video largo con muchos picos de \
    energía/risa, la mayoría ni siquiera llega a evaluarse porque el modelo \
    no los prioriza dentro de un transcript enorme. En vez de depender de que \
    el LLM repare en cada marca, forzamos un candidato directo por cada \
    reacción fuerte no cubierta ya por un candidato existente — igual pasa \
    por el mismo filtro de calidad (fit_clip + ranking comparativo) después, \
    esto solo garantiza que entre a la cancha.
    """
    existing = [w for windows in existing_windows for w in windows]
    extra: list[dict[str, Any]] = []
    for start, end, z in _group_nearby_spikes(energy_spikes, min_z):
        mid = (start + end) / 2
        if any(ws - _SPIKE_GROUP_GAP <= mid <= we + _SPIKE_GROUP_GAP for ws, we in existing):
            continue
        idx = _nearest_segment_index(segments, mid)
        snippet = segments[idx]["text"].strip()[:40]
        extra.append({
            "parts": [{"start_index": idx, "end_index": idx}],
            "title": f"(pico de audio) {snippet}",
            "bridge": "N/A",
            "bridge_type": "n_a",
            "hook_type": "gracioso",
            "score": 6,
            "reason": "Candidato agregado por un pico fuerte de volumen/risa en el audio, no vino del análisis de texto.",
        })
        # Lo sumamos a `existing` ya mismo para que el próximo pico cercano
        # de este mismo bloque no genere otro candidato pegado a este.
        existing.append((start, end))
    return extra


def analyze(
    transcript: dict[str, Any],
    video_stem: str,
    video_path: Path | None = None,
    use_visual: bool = False,
    review: bool = False,
    category: str = "general",
    force: bool = False,
) -> list[dict[str, Any]]:
    """Le pide al LLM los mejores momentos y devuelve una lista de clips con partes ya calculadas.

    Si `review=True`, devuelve TODOS los candidatos evaluados (no solo los
    que pasaron el filtro), para que un humano elija en vez de confiar en el
    filtro automático — pensado para la interfaz de revisión.

    `category` ajusta el criterio de selección al género del video (p. ej.
    "streaming" para contenido de streaming/gaming/IRL, que no tiene la
    misma estructura narrativa que una entrevista y no debería juzgarse con
    la misma vara). "general" usa el criterio por defecto (entrevista/podcast).
    """
    cache_suffix = "candidates" if review else "clips"
    category_tag = f".{category}" if category != "general" else ""
    out_path = TRANSCRIPTS_DIR / f"{video_stem}{category_tag}.{cache_suffix}.json"
    if out_path.exists() and not force:
        print(f"[analyze] usando selección cacheada: {out_path}")
        return json.loads(out_path.read_text(encoding="utf-8"))

    if not settings.openai_api_key:
        raise RuntimeError("Falta OPENAI_API_KEY (definila en tu archivo .env)")

    client = OpenAI(api_key=settings.openai_api_key)
    segments = transcript["segments"]

    energy_spikes: list[tuple[float, float, float]] = []
    if video_path is not None:
        print("[analyze] detectando picos de energía/risas en el audio...")
        energy_spikes = detect_energy_spikes(video_path, video_stem, force=force)
        print(f"[analyze] {len(energy_spikes)} picos de energía detectados")

    if use_visual and video_path is not None:
        visual_moments = analyze_visual(video_path, video_stem, force=force)
        segments = _merge_visual_segments(segments, visual_moments)

    # Pedimos un pool más amplio que el objetivo final: la segunda pasada
    # (validación comparativa) necesita candidatos de sobra para poder
    # descartar los más flojos en vez de quedarse sin opciones.
    pool_max = max(settings.max_clips + 3, settings.max_clips * 2)
    system = SYSTEM_PROMPT.format(min_clips=settings.min_clips, max_clips=pool_max)
    system += CATEGORY_ADDENDUMS.get(category, "")
    user = USER_PROMPT_TEMPLATE.format(numbered_transcript=_numbered_transcript(segments, energy_spikes))

    print(f"[analyze] llamando a {settings.openai_model}...")
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_schema", "json_schema": CLIP_JSON_SCHEMA},
        temperature=0.4,
    )

    raw_clips = json.loads(response.choices[0].message.content).get("clips", [])

    candidates: list[dict[str, Any]] = []
    for c in raw_clips:
        resolved_parts = []
        for p in c["parts"]:
            start_i = _resolve_index(segments, p["start_index"], p.get("start_quote", ""))
            end_i = _resolve_index(segments, p["end_index"], p.get("end_quote", ""))
            if start_i is None or end_i is None:
                print(f"  [analyze] descarto \"{c['title']}\": no pude verificar el "
                      f"índice citado contra el texto real (posible alucinación)")
                resolved_parts = None
                break
            if start_i != p["start_index"] or end_i != p["end_index"]:
                print(f"  [analyze] corrijo índice de \"{c['title']}\": "
                      f"{p['start_index']}->{start_i}, {p['end_index']}->{end_i}")
            resolved_parts.append({"start_index": min(start_i, end_i), "end_index": max(start_i, end_i)})
        if resolved_parts is None:
            continue

        raw_parts = resolved_parts
        bridge_type = c.get("bridge_type", "n_a")

        if len(raw_parts) == 2:
            a_end = segments[max(0, min(len(segments) - 1, raw_parts[0]["end_index"]))]["end"]
            b_start = segments[max(0, min(len(segments) - 1, raw_parts[1]["start_index"]))]["start"]
            gap = b_start - a_end
            if gap > _MAX_STITCH_GAP_SECONDS:
                # Ni el LLM (bridge) ni la pasada de ranking comparativo
                # detectaron con confiabilidad empalmes de partes muy
                # separadas en el tiempo que en realidad son temas distintos
                # (lo vimos repetirse en la práctica). Una idea que
                # genuinamente "se retoma" pasa cerca en el tiempo, no a
                # minutos de distancia en otro tema — es un límite mecánico,
                # no depende de que el modelo se autoevalúe bien.
                print(f"  [analyze] descarto 2da parte de \"{c['title']}\": "
                      f"{gap:.0f}s de distancia entre partes, demasiado lejos")
                raw_parts = [raw_parts[0]]
            elif bridge_type == "retoma_idea" and _bridge_is_weak(c.get("bridge", "")):
                # Para "recorte_pregunta" no hay fallback razonable a 1 sola
                # parte: la parte A es a propósito solo el núcleo de la
                # pregunta, sin la respuesta no sirve de nada. Acá solo
                # filtramos placeholders obvios; el resto del juicio semántico
                # lo hace la pasada de ranking.
                print(f"  [analyze] descarto 2da parte de \"{c['title']}\": "
                      f"bridge vacío o placeholder (bridge={c.get('bridge')!r})")
                raw_parts = [raw_parts[0]]

        windows = _fit_clip(
            segments,
            raw_parts,
            settings.min_clip_seconds,
            settings.max_clip_seconds,
            transcript["duration"],
        )
        if not windows:
            continue
        candidates.append({**c, "parts": windows})

    if energy_spikes:
        existing_windows = [c["parts"] for c in candidates]
        forced = _candidates_from_energy_spikes(segments, energy_spikes, existing_windows)
        if forced:
            print(f"  [analyze] {len(forced)} candidatos forzados por picos de energía no cubiertos por el LLM")
        for c in forced:
            windows = _fit_clip(
                segments, c["parts"], settings.min_clip_seconds, settings.max_clip_seconds, transcript["duration"],
            )
            if not windows:
                continue
            candidates.append({**c, "parts": windows})

    clips = _dedupe_by_score(candidates)

    if not clips:
        raise RuntimeError("El LLM no devolvió clips válidos. Revisá el transcript o el prompt.")

    ranked = _validate_ranking(client, segments, clips, category=category)

    if review:
        # Modo revisión: devolvemos TODOS los candidatos (con keep/score/
        # verdict), ordenados por tiempo, para que un humano elija en vez de
        # confiar ciegamente en el filtro automático.
        result = sorted(ranked, key=lambda c: c["parts"][0][0])
    else:
        result = sorted([c for c in ranked if c.get("keep", True)], key=lambda c: c["parts"][0][0])

    print(f"[analyze] {len(result)} clips {'candidatos (revisión)' if review else 'seleccionados'}:")
    for c in result:
        total = sum(e - s for s, e in c["parts"])
        partes = ", ".join(f"{s:.1f}s-{e:.1f}s" for s, e in c["parts"])
        salto = f" [{len(c['parts'])} partes]" if len(c["parts"]) > 1 else ""
        print(f"  {partes}{salto} ({c['hook_type']}, score {c['score']}, {total:.1f}s) {c['title']}")

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
