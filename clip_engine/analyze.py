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

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from openai import OpenAI

from .audio_energy import detect_energy_spikes
from .config import TRANSCRIPTS_DIR, settings
from .cost_tracker import record_usage
from .heatmap import get_heatmap
from .openai_retry import call_with_backoff
from .visual_analyze import analyze_visual
from .prompts import (
    CATEGORY_ADDENDUMS,
    CLIP_JSON_SCHEMA,
    RANKING_JSON_SCHEMA,
    RANKING_SYSTEM_PROMPT,
    RANKING_USER_TEMPLATE,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    custom_instruction_block,
)


_MAX_STITCH_GAP_SECONDS = 240.0  # distancia máxima entre partes de un mismo clip


def _energy_tag(seg: dict[str, Any], energy_spikes: list[tuple[float, float, float]]) -> str:
    for start, end, z in energy_spikes:
        if seg["start"] < end and seg["end"] > start:
            return " (🔊 pico de volumen/risa fuerte acá)" if z < 4 else " (🔊🔊 pico de volumen/risa MUY fuerte acá)"
    return ""


_RHYTHM_PAUSE_SECONDS = 2.0  # pausa que suele preceder a un remate/frase de impacto (timing cómico/dramático)
_RHYTHM_MAX_WORDS = 4  # una frase CORTA después de la pausa es lo que la hace sentir un remate, no una divagación cualquiera
# Calibrado contra transcripciones reales: con umbrales más laxos (1.0s/8 palabras)
# marcaba ~20% de las líneas en contenido conversacional normal — demasiado como
# para ser una señal distintiva. Con estos valores da ~1-2% en entrevista/streaming
# (contenido "de comer" tipo desafíos gastronómicos es una excepción esperada, tiene
# pausas naturales de masticar que suben la tasa, no es un bug).


def _rhythm_tag(prev_seg: dict[str, Any] | None, seg: dict[str, Any]) -> str:
    """Señal de ritmo del habla — no del contenido: una pausa larga seguida \
    de una frase corta suele ser timing de remate (cómico o dramático), \
    igual que el pico de volumen es una señal aparte del contenido semántico. \
    Se calcula con los timestamps que ya tenemos, no necesita nada nuevo."""
    if prev_seg is None:
        return ""
    gap = seg["start"] - prev_seg["end"]
    if gap < _RHYTHM_PAUSE_SECONDS or len(seg["text"].split()) > _RHYTHM_MAX_WORDS:
        return ""
    return " (⏸ pausa larga antes de esta frase corta — posible remate/frase de impacto)"


def _numbered_transcript(
    segments: list[dict[str, Any]], energy_spikes: list[tuple[float, float, float]] | None = None,
) -> str:
    energy_spikes = energy_spikes or []
    lines = []
    prev_seg = None
    for i, seg in enumerate(segments):
        tag = _energy_tag(seg, energy_spikes) + _rhythm_tag(prev_seg, seg)
        lines.append(f"[{i} | {seg['start']:.1f}s -> {seg['end']:.1f}s]{tag} {seg['text']}")
        prev_seg = seg
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


_FILLER_MIN_REPEAT = 2  # una sola palabra repetida esta cantidad de veces o más es puro relleno


def _is_filler_segment(text: str) -> bool:
    """Detecta muletillas o ruido puramente repetitivo (p. ej. "pau, pau, \
    pau", "eh, eh, eh") — no aportan contexto real, así que no deberían \
    quedar como el borde de un clip. Solo se usa para recortar los BORDES \
    (nunca el medio) después de que el resto del ajuste ya corrió, así que \
    no hace falta que sea perfecto: en el peor caso, deja pasar una muletilla \
    ocasional.
    """
    words = [w.strip(".,;:¿?¡!()\"'").lower() for w in text.split()]
    words = [w for w in words if w]
    if not words:
        return True
    return len(set(words)) == 1 and len(words) >= _FILLER_MIN_REPEAT


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

    def effective_dur() -> float:
        """Como total_dur(), pero sin contar huecos internos de silencio real
        (>= el mismo umbral que después usa el auto-recorte de pausas) como
        si ya fueran contenido. Sin esto, estirar el clip hasta el mínimo
        podía "cumplir" el número de segundos tragándose un silencio grande
        de un tirón — el auto-recorte lo sacaba después y el clip quedaba
        muy por debajo del mínimo real (pasó con un clip de 10s "de
        duración" que terminó en 4.8s tras el recorte, con un hueco de 5.3s
        escondido DENTRO de un solo segmento de faster-whisper, entre dos
        palabras — por eso esto mira gaps a nivel palabra, como hace
        `detect_micro_cuts`, no a nivel segmento: un segmento largo puede
        tener un silencio real adentro sin que se note mirando sus bordes.
        Midiendo así, estirar sigue de largo hasta juntar contenido real,
        no reloj de pared."""
        total = total_dur()
        for lo, hi in parts:
            words = [w for seg in segments[lo:hi + 1] for w in seg.get("words", [])]
            for i in range(1, len(words)):
                gap = words[i]["start"] - words[i - 1]["end"]
                if gap >= _MICRO_CUT_MIN_GAP:
                    total -= gap
        return total

    # Si el conjunto quedó corto, estiramos los bordes externos (antes de la
    # primera parte, después de la última). Proporción 1:1 — antes era 2
    # para atrás por cada 1 para adelante, pero eso perjudicaba justo a los
    # candidatos forzados por un pico de audio/visual (un solo índice de
    # anclaje, sin desarrollo previo real que rescatar): el remate casi
    # siempre viene DESPUÉS del pico, no antes, y priorizar tanto el
    # contexto previo hacía que la ventana se quedara corta antes de
    # llegar a la reacción real. Caso real medido: con el viejo 2:1 la
    # ventana llegaba hasta el segundo 1805.6 ("Oh, my God") y se quedaba
    # ahí — con 1:1 llega a 1814.6, alcanzando "la concha de la lora...
    # ¡sí, señor!", el remate real que antes se perdía.
    # Tope de seguridad: nunca estirar más allá de un múltiplo generoso de
    # max_dur en RELOJ DE PARED, sin importar qué diga effective_dur().
    # Encontrado en vivo (2026-08-21): un segmento roto de faster-whisper de
    # 51.6s (4 palabras reales, el resto un hueco enorme adentro) hacía que
    # effective_dur() midiera ~1s de "contenido real" ahí — el loop de
    # estirado, buscando desesperado más contenido, caminaba tan lejos por
    # la lista de segmentos que terminaba en una ventana final TOTALMENTE
    # desconectada del punto de anclaje original (a más de 100s de
    # distancia, en otro tema del video). Mejor devolver algo corto pero
    # coherente que algo "completo" pero sin relación con lo que se quiso
    # anclar.
    # No es que la ventana se ponga ANCHA (eso ya lo cubre hard_max más
    # abajo) — el problema real es que se ALEJA del punto de anclaje
    # original sin acumular duración, porque cada paso mueve UN índice de
    # segmento y muchos segmentos seguidos pueden tener gaps enormes
    # también. Por eso el tope se mide contra el ancla ORIGINAL (antes de
    # cualquier estirado), no contra el ancho de la ventana actual.
    anchor_lo_idx, anchor_hi_idx = parts[0][0], parts[-1][1]
    anchor_start = segments[anchor_lo_idx]["start"]
    anchor_end = segments[anchor_hi_idx]["end"]
    wall_clock_cap = max_dur * 3
    # La protección de ancla de abajo (no recortar más allá de anchor_lo_idx/
    # anchor_hi_idx) solo tiene sentido cuando el ancla es UN segmento (o
    # muy pocos) sin nada adentro para recortar — el caso real que motivó el
    # fix: un solo segmento roto de Whisper con duración de reloj de pared
    # inflada por un hueco interno enorme. Ahí no hay nada más chico que
    # devolver, así que proteger el segmento entero es lo correcto.
    #
    # Pero si el ancla ya abarca MUCHOS segmentos, la duración de reloj de
    # pared por sí sola NO alcanza como criterio — encontrado en vivo
    # (2026-08-22): el LLM propuso un rango de 156 a 399 (244 segmentos,
    # 434.7s — casi seguro una alucinación del LLM, no un momento real).
    # Usar el mismo chequeo de "¿ya es más grande que hard_max?" ahí
    # protegía el rango ENTERO, dejando el recorte de abajo sin poder hacer
    # nada — resultado: un "clip" de más de 7 minutos. La diferencia real
    # entre los dos casos no es la duración, es si HAY algo adentro para
    # recortar: con 244 segmentos de sobra para elegir una sub-ventana
    # razonable, no hace falta proteger el rango completo. Por eso el
    # criterio es CANTIDAD DE SEGMENTOS del ancla original, no su duración.
    # El umbral de "≤20 segmentos" (elegido a ojo la madrugada del Ciclo 4,
    # nunca calibrado contra un caso real de habla lenta) resultó demasiado
    # generoso: en podcast/entrevista, donde los segmentos de faster-whisper
    # duran varios segundos cada uno (habla pausada, no cruces rápidos como
    # en streaming), 19 segmentos pueden ser 71 SEGUNDOS reales de diálogo
    # continuo — muy por encima de hard_max, y sin embargo protegido de
    # recortarse. Caso real (2026-08-23, Podium Podcast, entrevista a Sofía
    # Vergara): un candidato de 19 segmentos/71.3s sobre "Griselda Blanco"
    # quedó con el score más alto de toda la corrida y `keep=true`, el doble
    # del máximo permitido. La única situación que de verdad necesita
    # protección es el ancla de UN SOLO segmento (el caso original: nada más
    # chico para devolver) — cualquier rango de 2+ segmentos con contenido
    # real ya tiene de dónde recortar, así que no hace falta protegerlo.
    anchor_segment_count = anchor_hi_idx - anchor_lo_idx + 1
    anchor_protected = anchor_segment_count <= 1

    step = 0
    while effective_dur() < min_dur and (parts[0][0] > 0 or parts[-1][1] < n - 1):
        back_capped = anchor_start - segments[parts[0][0]]["start"] >= wall_clock_cap
        fwd_capped = segments[parts[-1][1]]["end"] - anchor_end >= wall_clock_cap
        if back_capped and fwd_capped:
            break
        want_back = step % 2 == 0
        if want_back and not back_capped and parts[0][0] > 0:
            parts[0][0] -= 1
        elif not fwd_capped and parts[-1][1] < n - 1:
            parts[-1][1] += 1
        elif not back_capped and parts[0][0] > 0:
            parts[0][0] -= 1
        else:
            break
        step += 1

    # Si se pasa del techo duro, recortamos desde los bordes externos hacia
    # adentro (última parte primero), sin vaciar ninguna parte por completo.
    # IMPORTANTE: nunca recortar más allá del segmento ancla original
    # (anchor_lo_idx/anchor_hi_idx). Antes este bucle no tenía ese límite —
    # con un segmento roto de faster-whisper cuya duración de reloj de pared
    # YA por sí sola supera hard_max (caso real: 51.66s de un solo segmento
    # con casi todo hueco adentro, contra un hard_max de 43.75s), el recorte
    # seguía de largo achicando `hi` mucho más allá del propio punto de
    # anclaje, dejando una ventana final TOTALMENTE desconectada del
    # candidato original (a más de 100s de distancia). Con el tope, en el
    # peor caso el resultado es el segmento ancla tal cual (más largo que
    # max_dur, pero al menos anclado en el lugar correcto) en vez de basura
    # en otra parte del video.
    hard_max = max_dur * 1.25
    trimmed_hard_max = False
    while total_dur() > hard_max:
        last_lo, last_hi = parts[-1]
        if last_hi > last_lo and (not anchor_protected or last_hi > anchor_hi_idx):
            parts[-1][1] -= 1
            trimmed_hard_max = True
            continue
        first_lo, first_hi = parts[0]
        if first_hi > first_lo and (not anchor_protected or first_lo < anchor_lo_idx):
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

    # No dejar que un clip arranque o cierre justo en una muletilla repetitiva
    # ("pau, pau, pau") que quedó pegada ahí por el estiramiento a duración
    # mínima de más arriba — no aporta nada y se siente arbitrario. Solo se
    # recortan los bordes, nunca el medio, y solo si sobra contenido real en
    # esa misma parte (si toda la parte es puro relleno no hay nada que
    # rescatar acá, y está bien que quede corta: el estiramiento en segundos
    # de más abajo se encarga de compensar la duración si hace falta).
    last_lo, last_hi = parts[-1]
    while last_hi > last_lo and _is_filler_segment(segments[last_hi]["text"]):
        last_hi -= 1
    parts[-1][1] = last_hi

    first_lo, first_hi = parts[0]
    while first_lo < first_hi and _is_filler_segment(segments[first_lo]["text"]):
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


# Subido de 0.8 a 1.5 el 2026-08-20: sobre 1546 clips reales ya publicados y
# exitosos del corpus, la pausa interna típica dura 0.88s de mediana y hasta
# 1.68s en el 75% — con 0.8 se proponía cortar el ritmo normal de casi
# cualquier clip bueno, no silencio muerto real. Ver mismo cambio en server.py.
_MICRO_CUT_MIN_GAP = 1.5  # gaps de al menos esto entre palabras consecutivas se proponen como recorte


def detect_micro_cuts(
    segments: list[dict[str, Any]], parts: list[tuple[float, float]], min_gap: float = _MICRO_CUT_MIN_GAP,
) -> list[tuple[float, float]]:
    """Encuentra pausas/silencios INTERNOS dentro de un clip ya elegido (no \
    en los bordes — de eso se encarga `_fit_clip`) que se puedan recortar \
    para que el ritmo quede más ágil. Usa los timestamps por palabra que ya \
    tenemos de faster-whisper, no hace falta ningún análisis nuevo de audio.

    Devuelve rangos (start, end) en tiempo ABSOLUTO del video original, \
    listos para pasarle a `apply_micro_cuts`. Es solo DETECCIÓN — quien \
    llama decide si se los muestra al usuario como sugerencia editable \
    (nunca se aplican solos sin que alguien los vea).
    """
    words: list[dict[str, Any]] = []
    for part_start, part_end in parts:
        for seg in segments:
            if seg["end"] <= part_start or seg["start"] >= part_end:
                continue
            for w in seg.get("words", []):
                if w["end"] > part_start and w["start"] < part_end:
                    words.append(w)
    words.sort(key=lambda w: w["start"])

    cuts: list[tuple[float, float]] = []
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap >= min_gap:
            cuts.append((round(words[i - 1]["end"], 3), round(words[i]["start"], 3)))
    return cuts


def apply_micro_cuts(
    parts: list[tuple[float, float]], cuts: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Resta los rangos de `cuts` de `parts`, partiendo cada ventana que \
    cruza un corte en dos. El resultado sigue siendo una lista plana de \
    ventanas [start, end] — el mismo formato que ya entiende `cutter.py`, \
    que no tiene límite de cuántas partes acepta. No hace falta tocar nada \
    del motor de corte para esto."""
    result = list(parts)
    for cut_start, cut_end in cuts:
        new_result: list[tuple[float, float]] = []
        for p_start, p_end in result:
            if cut_end <= p_start or cut_start >= p_end:
                new_result.append((p_start, p_end))
                continue
            if cut_start > p_start:
                new_result.append((p_start, cut_start))
            if cut_end < p_end:
                new_result.append((cut_end, p_end))
        result = new_result
    result.sort(key=lambda p: p[0])
    return result


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
    client: OpenAI,
    segments: list[dict[str, Any]],
    clips: list[dict[str, Any]],
    video_stem: str,
    category: str = "general",
    custom_instruction: str | None = None,
    target_clips: int | None = None,
) -> list[dict[str, Any]]:
    """Segunda pasada: compara los candidatos ENTRE SÍ (la primera pasada
    evalúa cada uno aislado) y filtra los que, puestos al lado de los demás,
    son claramente más flojos — aunque técnicamente encajen en una categoría.

    Recibe el mismo `category`/`custom_instruction` que la primera pasada: si
    no se le suma el mismo contexto acá también, esta segunda pasada juzga
    con el criterio genérico (le exige "desarrollo claro", "remate", etc.)
    aunque la primera haya sido más permisiva o haya priorizado otra cosa —
    eso deshace en la validación lo que el addendum logró en la selección.
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
    if target_clips:
        if target_clips <= 5:
            system += (
                f"\n\nEl usuario pidió {target_clips} clips — priorizá MUCHO la calidad sobre "
                "la cantidad, sé especialmente exigente: quedate solo con lo mejor de lo mejor, "
                "aunque eso signifique devolver bastante menos de lo pedido."
            )
        elif target_clips >= 15:
            system += (
                f"\n\nEl usuario pidió {target_clips} clips — podés ser más permisivo que de "
                "costumbre con candidatos parejos o de relleno (no todos van a ser espectaculares), "
                "PERO nunca dejes pasar algo genuinamente vacío o sin ningún gancho real solo para "
                "completar el número."
            )
    if custom_instruction:
        system += custom_instruction_block(custom_instruction)

    print(f"[analyze] validando {len(clips)} candidatos entre sí (con {settings.openai_ranking_model})...")
    response = call_with_backoff(
        lambda: client.chat.completions.create(
            model=settings.openai_ranking_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_schema", "json_schema": RANKING_JSON_SCHEMA},
            temperature=0.3,
        ),
        label="analyze-ranking",
    )
    record_usage(video_stem, "ranking", settings.openai_ranking_model, response.usage)
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
            # UNA palabra sintética cubriendo todo el tramo, no lista vacía.
            # Encontrado en vivo (2026-08-21): con words=[], el detector de
            # pausas (que mide huecos entre palabras consecutivas) no ve
            # nada acá — el tramo entero queda invisible para él, y el hueco
            # entre la última palabra real ANTES y la primera palabra real
            # DESPUÉS de este momento visual se mide como una pausa gigante
            # recortable, borrando el momento visual completo (pasó con un
            # caso real: un highlight de 21s quedó en 0.2s). Con una palabra
            # sintética ocupando el tramo, el hueco real se mide correcto a
            # cada lado, sin tragarse el momento visual en el medio.
            "words": [{"start": m["start"], "end": m["end"], "word": "[visual]"}],
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


_HEATMAP_MIN_VALUE = 0.6
# Encontrado por investigación de competencia (2026-08-21): Opus Clip usa el
# heatmap de "más repetido" de YouTube como señal PRINCIPAL de selección, no
# solo de referencia. Validado en vivo esta misma noche: dos momentos reales
# que el pipeline se perdió (uno marcado a mano por el usuario, otro
# encontrado después) resultaron ser, medido con este heatmap, de los tramos
# más repetidos de sus videos — la señal es real. 0.6 es sobre el valor
# NORMALIZADO de cada video (el punto más visto de CUALQUIER video vale 1.0),
# así que 0.6 significa "entre los picos más fuertes de ESTE video puntual",
# no un número absoluto comparable entre videos.


def _candidates_from_heatmap(
    segments: list[dict[str, Any]],
    heatmap: list[dict[str, Any]] | None,
    existing_windows: list[list[tuple[float, float]]],
    min_value: float = _HEATMAP_MIN_VALUE,
) -> list[dict[str, Any]]:
    """Mismo mecanismo que los picos de audio/visual, pero con la señal de \
    audiencia real de YouTube en vez de una heurística nuestra: si un tramo \
    del heatmap es de los más repetidos de ESTE video y no está cubierto \
    por ningún candidato existente, lo forzamos a entrar a la validación. \
    `heatmap` puede ser None (video subido a mano, o sin suficientes vistas \
    para tener heatmap) — en ese caso simplemente no aporta candidatos, el \
    resto del pipeline sigue igual.
    """
    if not heatmap:
        return []
    # A propósito NO se salta picos ya "cubiertos" por un candidato débil
    # existente (a diferencia de audio/visual) — probado en vivo: dos picos
    # reales de heatmap quedaron afuera porque un candidato de audio flojo
    # ya ocupaba un window cercano, y el heatmap es la señal más confiable
    # que tenemos (datos reales de audiencia, no heurística nuestra). Se
    # agrega siempre; `_dedupe_by_score` decide después cuál gana si se
    # superponen — por eso el score es más alto que el de audio/visual
    # (7, no 6): para que gane el desempate cuando compiten por la misma
    # ventana.
    extra: list[dict[str, Any]] = []
    for p in heatmap:
        if p.get("value", 0) < min_value:
            continue
        mid = (p["start_time"] + p["end_time"]) / 2
        idx = _nearest_segment_index(segments, mid)
        extra.append({
            "parts": [{"start_index": idx, "end_index": idx}],
            "title": f"(más repetido, {p['value']:.0%}) {segments[idx]['text'].strip()[:40]}",
            "bridge": "N/A",
            "bridge_type": "n_a",
            "hook_type": "gracioso",
            "score": 7,
            "reason": f"Candidato agregado porque es uno de los tramos más repetidos reales de este video (heatmap de YouTube, {p['value']:.0%}).",
        })
    return extra


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


_VISUAL_FORCE_MIN_SCORE = 6
# Mismo piso que "notable" en visual_analyze.py — en la práctica casi todo lo
# que pasa ese piso puntúa justo 6, así que exigir más acá dejaría el forzado
# casi vacío.


def _candidates_from_visual_moments(
    segments: list[dict[str, Any]],
    visual_moments: list[dict[str, Any]],
    existing_windows: list[list[tuple[float, float]]],
    min_score: int = _VISUAL_FORCE_MIN_SCORE,
) -> list[dict[str, Any]]:
    """Mismo problema que con los picos de audio, pero para lo visual: un \
    momento notable insertado como pista suelta en una transcripción de \
    miles de líneas es fácil que el LLM lo pase por alto — sobre todo si NO \
    hay nada dicho en ese momento (el caso que más importa: alguien conteniendo \
    la risa en silencio, un gesto, algo que pasa en cámara sin que nadie hable). \
    Forzamos un candidato directo por cada momento visual fuerte no cubierto \
    ya por un candidato existente, en vez de confiar en que el LLM repare en \
    la pista.
    """
    existing = [w for windows in existing_windows for w in windows]
    extra: list[dict[str, Any]] = []
    for m in visual_moments:
        if m["score"] < min_score:
            continue
        mid = (m["start"] + m["end"]) / 2
        if any(ws - 5 <= mid <= we + 5 for ws, we in existing):
            continue
        idx = _nearest_segment_index(segments, mid)
        extra.append({
            "parts": [{"start_index": idx, "end_index": idx}],
            "title": f"(visual) {m['description']}",
            "bridge": "N/A",
            "bridge_type": "n_a",
            "hook_type": m.get("hook_type", "gracioso"),
            "score": 6,
            "reason": f"Candidato agregado por un momento visual notable: {m['description']}",
        })
        existing.append((m["start"], m["end"]))
    return extra


def analyze(
    transcript: dict[str, Any],
    video_stem: str,
    video_path: Path | None = None,
    use_visual: bool = False,
    review: bool = False,
    category: str = "general",
    custom_instruction: str | None = None,
    force: bool = False,
    target_clips: int | None = None,
    speed: str = "completo",
) -> list[dict[str, Any]]:
    """Le pide al LLM los mejores momentos y devuelve una lista de clips con partes ya calculadas.

    Si `review=True`, devuelve TODOS los candidatos evaluados (no solo los
    que pasaron el filtro), para que un humano elija en vez de confiar en el
    filtro automático — pensado para la interfaz de revisión.

    `category` ajusta el criterio de selección al género del video (p. ej.
    "streaming" para contenido de streaming/gaming/IRL, que no tiene la
    misma estructura narrativa que una entrevista y no debería juzgarse con
    la misma vara). "general" usa el criterio por defecto (entrevista/podcast).

    `custom_instruction` es un pedido libre del usuario en criollo (p. ej.
    "dame los momentos donde se ríe fuerte", "sacá los que hablan de plata")
    que se suma AL criterio de categoría, no lo reemplaza — reordena qué se
    prioriza sin tirar abajo el resto del criterio de calidad.

    `target_clips` es el número de clips finales que pidió el usuario (p. ej.
    5/10/20 desde la interfaz) — no es solo un tope, cambia la EXIGENCIA: con
    pocos, la validación se pone más estricta (solo lo mejor de lo mejor); con
    más, acepta candidatos más parejos/de relleno, siempre que tengan algo
    real, nunca algo vacío. None usa el default de `settings`. A propósito NO
    forma parte de la clave de caché (mismo motivo que category/instruction sí
    lo son: identifican una búsqueda distinta; la cantidad es un parámetro de
    la corrida) — pedir otra cantidad reusa el caché salvo que se fuerce.
    """
    cache_suffix = "candidates" if review else "clips"
    category_tag = f".{category}" if category != "general" else ""
    custom_tag = ""
    if custom_instruction:
        # hashlib en vez de hash() built-in: hash() varía entre corridas del
        # proceso (aleatorizado por seguridad), rompería el caché cada vez
        # que se reinicia el servidor.
        digest = hashlib.sha1(custom_instruction.strip().lower().encode("utf-8")).hexdigest()[:8]
        custom_tag = f".custom-{digest}"
    out_path = TRANSCRIPTS_DIR / f"{video_stem}{category_tag}{custom_tag}.{cache_suffix}.json"
    if out_path.exists() and not force:
        print(f"[analyze] usando selección cacheada: {out_path}")
        return json.loads(out_path.read_text(encoding="utf-8"))

    if not settings.openai_api_key:
        raise RuntimeError("Falta OPENAI_API_KEY (definila en tu archivo .env)")

    if speed == "rapido":
        use_visual = False  # "rápido" es audio+texto solo, sin importar el toggle

    client = OpenAI(api_key=settings.openai_api_key)
    segments = transcript["segments"]

    energy_spikes: list[tuple[float, float, float]] = []
    if video_path is not None:
        print("[analyze] detectando picos de energía/risas en el audio...")
        energy_spikes = detect_energy_spikes(video_path, video_stem, force=force)
        print(f"[analyze] {len(energy_spikes)} picos de energía detectados")

    visual_moments: list[dict[str, Any]] = []
    if use_visual and video_path is not None:
        # "medio" sacrifica densidad de sampleo y el detalle fino de
        # descripción para ir más rápido — se nota en candidatos visuales
        # menos específicos, pero corre en una fracción del tiempo.
        visual_interval = 8.0 if speed == "medio" else None
        visual_moments = analyze_visual(
            video_path, video_stem, force=force,
            interval=visual_interval, skip_refine=(speed == "medio"),
        )
        segments = _merge_visual_segments(segments, visual_moments)

    # Pedimos un pool más amplio que el objetivo final: la segunda pasada
    # (validación comparativa) necesita candidatos de sobra para poder
    # descartar los más flojos en vez de quedarse sin opciones.
    effective_max_clips = target_clips or settings.max_clips
    effective_min_clips = min(settings.min_clips, effective_max_clips)
    pool_max = max(effective_max_clips + 3, effective_max_clips * 2)
    system = SYSTEM_PROMPT.format(min_clips=effective_min_clips, max_clips=pool_max)
    system += CATEGORY_ADDENDUMS.get(category, "")
    if custom_instruction:
        system += custom_instruction_block(custom_instruction)
    user = USER_PROMPT_TEMPLATE.format(numbered_transcript=_numbered_transcript(segments, energy_spikes))

    print(f"[analyze] llamando a {settings.openai_model}...")
    response = call_with_backoff(
        lambda: client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_schema", "json_schema": CLIP_JSON_SCHEMA},
            temperature=0.4,
        ),
        label="analyze-selection",
    )
    record_usage(video_stem, "selection", settings.openai_model, response.usage)

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

    if visual_moments:
        existing_windows = [c["parts"] for c in candidates]
        forced_visual = _candidates_from_visual_moments(segments, visual_moments, existing_windows)
        if forced_visual:
            print(f"  [analyze] {len(forced_visual)} candidatos forzados por momentos visuales no cubiertos por el LLM")
        for c in forced_visual:
            windows = _fit_clip(
                segments, c["parts"], settings.min_clip_seconds, settings.max_clip_seconds, transcript["duration"],
            )
            if not windows:
                continue
            candidates.append({**c, "parts": windows})

    heatmap = None
    if video_path is not None:
        heatmap = get_heatmap(video_stem)
        if heatmap:
            existing_windows = [c["parts"] for c in candidates]
            forced_heatmap = _candidates_from_heatmap(segments, heatmap, existing_windows)
            if forced_heatmap:
                print(f"  [analyze] {len(forced_heatmap)} candidatos forzados por el heatmap de YouTube (más repetido)")
            for c in forced_heatmap:
                windows = _fit_clip(
                    segments, c["parts"], settings.min_clip_seconds, settings.max_clip_seconds, transcript["duration"],
                )
                if not windows:
                    continue
                candidates.append({**c, "parts": windows})

    clips = _dedupe_by_score(candidates)

    # `_dedupe_by_score` puede quedarse con un candidato generado por el LLM
    # (no por el forzado de heatmap) para una ventana que igual se superpone
    # con un pico real de audiencia — pasa cuando el candidato del LLM ya
    # traía un score de generación >= 7, empatando o ganando al del heatmap.
    # En ese caso se pierde la marca "(más repetido, X%)" en el título Y la
    # indulgencia especial que le da `RANKING_SYSTEM_PROMPT` a esa marca —
    # aunque la ventana siga estando físicamente sobre el mismo pico real.
    # Encontrado en vivo (2026-08-22, `u1O6Av1vO-I`): el pico #1 de todo el
    # video (202-211s, valor 1.00) terminó cubierto por un candidato del LLM
    # sin ninguna mención de heatmap, y ese candidato fue RECHAZADO por el
    # ranking (score 6) — la señal de audiencia real estaba disponible pero
    # nunca llegó al prompt de validación. Se re-etiqueta cualquier
    # sobreviviente del dedupe que se superponga con un pico fuerte, sin
    # importar de qué mecanismo haya salido originalmente.
    if heatmap:
        for clip in clips:
            if "(más repetido" in clip.get("title", ""):
                continue
            clip_start = clip["parts"][0][0]
            clip_end = clip["parts"][-1][1]
            best = None
            for p in heatmap:
                if p.get("value", 0) < _HEATMAP_MIN_VALUE:
                    continue
                if p["start_time"] < clip_end and p["end_time"] > clip_start:
                    if best is None or p["value"] > best["value"]:
                        best = p
            if best is not None:
                clip["title"] = f"(más repetido, {best['value']:.0%}) {clip['title']}"
                clip["reason"] = (
                    f"{clip.get('reason', '')} También coincide con uno de los tramos más "
                    f"repetidos reales de este video (heatmap de YouTube, {best['value']:.0%})."
                ).strip()

    if not clips:
        raise RuntimeError("El LLM no devolvió clips válidos. Revisá el transcript o el prompt.")

    ranked = _validate_ranking(
        client, segments, clips, video_stem,
        category=category, custom_instruction=custom_instruction, target_clips=target_clips,
    )

    # La instrucción de "sé más exigente" en el prompt de validación es una
    # señal blanda — probado en vivo, pedir 5 devolvió 17 candidatos con
    # keep=true (el LLM las juzgó buenas una por una, nada lo obliga a
    # cortar en un número). Acá sí se hace cumplir de verdad: si hay más
    # "keep=true" que lo pedido, los de menor score se pasan a keep=false
    # (nunca se los saca de la lista — en modo revisión igual se pueden ver
    # y elegir a mano, solo dejan de venir pre-marcados como recomendados).
    if target_clips:
        currently_kept = [c for c in ranked if c.get("keep", True)]
        if len(currently_kept) > target_clips:
            currently_kept.sort(key=lambda c: c.get("final_score", c.get("score", 0)), reverse=True)
            demote_ids = {id(c) for c in currently_kept[target_clips:]}
            for c in ranked:
                if id(c) in demote_ids:
                    c["keep"] = False

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
