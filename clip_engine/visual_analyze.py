"""Análisis visual del video: sampleo de frames + GPT-4o-mini con visión.

Complementa el análisis de audio/texto, no lo reemplaza. Existe porque en
contenido tipo streaming/IRL (grupo reaccionando, poca estructura verbal) el
texto solo no alcanza para encontrar los mejores momentos — mucho de lo bueno
pasa en la imagen (reacción fuerte, algo gracioso en pantalla, una mirada)
sin que quede un rastro fuerte en lo que se dice.

Sampleamos a intervalo fijo bastante denso (cada pocos segundos) en vez de
tratar de "adivinar" dónde mirar (probamos detección de movimiento y tiene
un punto ciego real: una mirada o gesto sutil casi no mueve píxeles). Como
cada foto cuesta centavos, no vale la pena la complejidad de ser selectivo —
mejor cubrir el video entero de forma pareja. El costo real de ir denso es
tiempo de procesamiento, no plata: lo compensamos evaluando varios lotes de
fotos en paralelo en vez de uno por uno.

Los frames notables se insertan como pseudo-líneas en la transcripción que
arma analyze.py, así el mismo criterio de selección (un solo LLM, un solo
pase) ve texto Y contexto visual juntos.
"""
from __future__ import annotations

import base64
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from .config import TRANSCRIPTS_DIR, settings
from .cost_tracker import record_usage
from .openai_retry import call_with_backoff

FRAME_INTERVAL_SECONDS = 3.0
BATCH_SIZE = 15
BATCH_PAUSE_SECONDS = 1.5
MIN_NOTABLE_SCORE = 6

# Tope duro de frames a scorear por video, sin importar el intervalo pedido.
# Existe porque un video mal categorizado (p. ej. un stream de horas metido
# en el bucket de "clips", que samplea denso) puede generar miles de frames
# y miles de llamadas a la API — pasó una vez y costó ~$7 en un solo video.
# En vez de truncar el video (perder el final), agrandamos el intervalo
# efectivo para cubrirlo entero pero con menos densidad.
MAX_FRAMES_PER_VIDEO = 900

VISUAL_SCHEMA = {
    "name": "visual_moments",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "frames": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "score": {"type": "integer"},
                        "description": {"type": "string"},
                    },
                    "required": ["index", "score", "description"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["frames"],
        "additionalProperties": False,
    },
}

VISUAL_SYSTEM_PROMPT = """\
Sos un editor de video buscando momentos visualmente interesantes para armar \
clips virales. Te paso una serie de frames tomados a intervalos regulares de \
un video (no son continuos, son fotos sueltas separadas por pocos segundos \
cada una). Para cada frame, en el orden en que te lo mando (empezando en \
índice 0), evaluá:
- `score` (1 a 10): qué tan visualmente notable es ESE INSTANTE puntual — \
  reacción facial fuerte (sorpresa, risa, shock, enojo), una mirada o gesto \
  cargado de intención entre personas, algo gracioso o inesperado en cámara, \
  una acción física llamativa, un objeto o situación rara en pantalla. La \
  gran mayoría de los frames van a ser gente sentada charlando normalmente \
  — eso es un 1-3, no le tengas miedo a puntuar bajo. Reservá 7+ solo para \
  algo genuinamente llamativo a simple vista.
- `description`: una frase muy corta (5-10 palabras) de qué se ve, en \
  español, útil para alguien que después decide si eso vale como clip.

No sabés qué se está diciendo en ese momento (no tenés el audio) — juzgá \
solo lo que se ve en la imagen.
"""


def extract_frame_at(video_path: Path, t: float, scale: int = 320) -> bytes | None:
    """Un solo frame en el segundo `t` — para darle contexto visual puntual a
    un pase del LLM (p. ej. el ranking comparativo, o el refinamiento de
    descripción de un momento ya marcado como notable) sin tener que escanear
    el video entero."""
    cmd = [
        "ffmpeg", "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(video_path),
        "-vf", f"scale={scale}:-1", "-frames:v", "1", "-q:v", "5", "-f", "image2", "-",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not result.stdout:
        return None
    return result.stdout


def _extract_frames(video_path: Path, interval: float) -> list[tuple[float, bytes]]:
    with tempfile.TemporaryDirectory() as tmp:
        pattern = str(Path(tmp) / "f_%06d.jpg")
        cmd = [
            "ffmpeg", "-v", "error", "-i", str(video_path),
            "-vf", f"fps=1/{interval},scale=320:-1",
            "-q:v", "5", pattern,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg no pudo samplear frames: {result.stderr[-2000:].decode(errors='ignore')}")

        files = sorted(Path(tmp).glob("f_*.jpg"))
        return [(i * interval, f.read_bytes()) for i, f in enumerate(files)]


_REFINE_BATCH_SIZE = 6
_REFINE_SCALE = 768  # bien más grande que el sampleo (320) — acá sí vale la pena, es solo para la fracción ya marcada notable

_HOOK_TYPES = ["hook_fuerte", "dato_sorprendente", "emocional", "gracioso", "consejo_practico", "debate_polemica"]

_REFINE_SCHEMA = {
    "name": "refined_descriptions",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "frames": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "description": {"type": "string"},
                        "hook_type": {"type": "string", "enum": _HOOK_TYPES},
                    },
                    "required": ["index", "description", "hook_type"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["frames"],
        "additionalProperties": False,
    },
}

_REFINE_SYSTEM_PROMPT = """\
Sos un editor de video. Te mando varias fotos, cada una YA IDENTIFICADA como \
un momento visualmente notable de un video (no hace falta que juzgues si \
vale la pena, eso ya se decidió). Para cada una, en el mismo orden en que \
te la mando (empezando en índice 0):

- `description`: en español, de 15 a 25 palabras, EXACTAMENTE qué está \
  pasando: quién hace qué, con qué objeto, qué expresión tienen las caras, \
  qué hay en pantalla. Sé concreto y literal — nombrá lo que ves (ropa, \
  objetos, gestos puntuales, expresiones), nunca uses solo una palabra \
  genérica como "reacciona", "gesticula" o "se ríe" sin decir de qué o con \
  qué. Si hay algo fuera de lo común en cuadro (disfraz, comida en la cara, \
  un objeto raro, alguien entrando de golpe, un gesto muy exagerado) es la \
  parte más importante y tenés que nombrarlo explícitamente.

  Mal: "reacciones fuertes entre los miembros del grupo"
  Bien: "a uno de los chicos le tiraron mayonesa en la cara y hace muecas \
  mientras el resto se tapa la boca de la risa"

- `hook_type`: qué tipo de gancho es ESTE instante puntual, para etiquetarlo \
  bien (no todo lo visualmente notable es gracioso):
  - `gracioso`: la cara/gesto/situación da risa
  - `emocional`: sorpresa genuina, shock, susto, angustia, ternura, orgullo
  - `hook_fuerte`: algo tenso, intenso o confrontativo (pelea, discusión \
    física, un desafío, un momento de alta tensión)
  - `dato_sorprendente`: se ve algo objetivamente insólito o llamativo en \
    pantalla que no encaja en las otras categorías
  - `debate_polemica`: gente discutiendo o en desacuerdo visible
  - `consejo_practico`: alguien mostrando cómo hacer algo, una demostración
  Si dudás entre dos, elegí la que más se acerca a "por qué alguien pararía \
  el dedo al ver esto".

No sabés qué se está diciendo en ese momento (no tenés el audio) — \
juzgá solo lo que se ve.
"""


def _refine_batch(client: OpenAI, batch: list[bytes], video_stem: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": f"Te mando {len(batch)} fotos, en orden (índice 0 a {len(batch) - 1})."}
    ]
    for img_bytes in batch:
        b64 = base64.b64encode(img_bytes).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
        })

    response = call_with_backoff(
        lambda: client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": _REFINE_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_schema", "json_schema": _REFINE_SCHEMA},
            temperature=0.3,
        ),
        label="visual-refine",
    )
    record_usage(video_stem, "visual-refine", settings.openai_model, response.usage)
    return json.loads(response.choices[0].message.content)["frames"]


def _refine_notable_descriptions(
    client: OpenAI, video_path: Path, notable: list[dict[str, Any]], video_stem: str,
) -> None:
    """Para la fracción chica de frames que ya pasó el piso de 'notable',
    vuelve a extraer ESE instante puntual en resolución alta y le pide al LLM
    una descripción larga y específica (con detail=high, no low como en el
    sampleo inicial). El sampleo inicial tiene que ser barato porque cubre
    el video entero; acá el volumen es chico (unos pocos frames por video)
    así que vale la pena gastar un poco más para que la descripción sea lo
    bastante específica como para competir sola en el ranking del LLM —
    "reacciones entre los miembros" no le gana a nada, "le tiraron mayonesa
    en la cara" sí.

    Refinamos TODOS los momentos notables, sin tope — antes había un límite
    acá para no arriesgar el item completo si un video de horas generaba
    miles de momentos, pero eso degradaba la calidad de datos silenciosamente
    (el resto se quedaba con descripción genérica) sin que nadie lo hubiera
    pedido. Ahora `MAX_FRAMES_PER_VIDEO` en `analyze_visual` ya limita cuántos
    frames se samplean por video, así que la cantidad de notables que puede
    salir de acá está acotada en el origen, no hace falta un segundo tope acá.
    """
    indexed = list(enumerate(notable))

    pairs: list[tuple[int, bytes]] = []
    for i, r in indexed:
        frame = extract_frame_at(video_path, r["start"], scale=_REFINE_SCALE)
        if frame is not None:
            pairs.append((i, frame))
    if not pairs:
        return

    for batch_start in range(0, len(pairs), _REFINE_BATCH_SIZE):
        if batch_start > 0:
            time.sleep(BATCH_PAUSE_SECONDS)
        batch = pairs[batch_start:batch_start + _REFINE_BATCH_SIZE]
        refined = _refine_batch(client, [img for _, img in batch], video_stem)
        for item in refined:
            local_idx = item["index"]
            if 0 <= local_idx < len(batch):
                orig_idx = batch[local_idx][0]
                notable[orig_idx]["description"] = item["description"]
                notable[orig_idx]["hook_type"] = item["hook_type"]


def _score_batch(client: OpenAI, batch: list[tuple[float, bytes]], video_stem: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": f"Te mando {len(batch)} frames, en orden (índice 0 a {len(batch) - 1})."}
    ]
    for _, img_bytes in batch:
        b64 = base64.b64encode(img_bytes).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
        })

    response = call_with_backoff(
        lambda: client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": VISUAL_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_schema", "json_schema": VISUAL_SCHEMA},
            temperature=0.3,
        ),
        label="visual-score",
    )
    record_usage(video_stem, "visual-score", settings.openai_model, response.usage)
    return json.loads(response.choices[0].message.content)["frames"]


def analyze_visual(
    video_path: Path, video_stem: str, force: bool = False, interval: float | None = None,
    skip_refine: bool = False,
) -> list[dict[str, Any]]:
    """Devuelve una lista de {start, end, score, description} para tramos del
    video con algo visualmente notable. Cachea el resultado igual que la
    transcripción, para no re-pagar la llamada a la API en corridas siguientes.

    Va secuencial, no en paralelo: el límite de tokens/minuto es un techo
    fijo de la cuenta, no algo que se esquive mandando varios pedidos a la
    vez — probamos paralelizar y lo único que logró fue que varios pedidos
    chocaran contra el límite al mismo tiempo y se agotaran los reintentos.

    `interval` permite pisar el intervalo de sampleo por defecto (p. ej. para
    un análisis de investigación sobre videos largos, donde no hace falta la
    misma densidad que en producción y correr más liviano ahorra tiempo/costo
    en un corpus grande) — también lo usa el modo "medio" de la interfaz.

    `skip_refine` salta el segundo pase de detalle alto (el que convierte
    "reacciones fuertes" en "le tiraron mayonesa en la cara") — el modo
    "medio" de la interfaz lo usa para ir más rápido, a costa de descripciones
    más genéricas. El cache queda separado por intervalo/refine (sufijo en el
    nombre) para que no se pisen resultados de distinta calidad entre modos.
    """
    interval = interval or FRAME_INTERVAL_SECONDS
    mode_tag = ""
    if interval != FRAME_INTERVAL_SECONDS or skip_refine:
        mode_tag = f".fast{interval:.0f}" if skip_refine else f".i{interval:.0f}"
    out_path = TRANSCRIPTS_DIR / f"{video_stem}{mode_tag}.visual.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))

    if not settings.openai_api_key:
        raise RuntimeError("Falta OPENAI_API_KEY (definila en tu archivo .env)")

    client = OpenAI(api_key=settings.openai_api_key)

    print(f"[visual] sampleando 1 frame cada {interval:.0f}s...")
    frames = _extract_frames(video_path, interval)
    if len(frames) > MAX_FRAMES_PER_VIDEO:
        # Video más largo de lo que el intervalo pedido puede cubrir barato
        # (p. ej. quedó mal categorizado y usó el intervalo denso de "clip").
        # Reextraemos con un intervalo más grande en vez de truncar el video.
        scale = len(frames) / MAX_FRAMES_PER_VIDEO
        old_interval = interval
        interval = interval * scale
        print(f"[visual] {len(frames)} frames supera el tope de {MAX_FRAMES_PER_VIDEO} "
              f"— re-sampleando cada {interval:.1f}s (era {old_interval:.0f}s) para cubrir el video entero sin disparar el costo")
        frames = _extract_frames(video_path, interval)

    # Cache de progreso parcial: si una corrida anterior se interrumpió a
    # mitad de camino (reinicio del daemon, crash, etc.), retomamos desde el
    # último lote scoreado en vez de volver a pagar por los frames ya hechos.
    partial_path = TRANSCRIPTS_DIR / f"{video_stem}.visual_partial.json"
    results: list[dict[str, Any]] = []
    resume_from = 0
    if partial_path.exists():
        try:
            partial = json.loads(partial_path.read_text(encoding="utf-8"))
            if partial.get("interval") == interval and partial.get("n_frames") == len(frames):
                results = partial["results"]
                resume_from = partial["next_batch_start"]
                print(f"[visual] retomando desde el frame {resume_from}/{len(frames)} (corrida anterior interrumpida)")
        except (json.JSONDecodeError, KeyError):
            pass

    n_batches = (len(frames) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"[visual] {len(frames)} frames extraídos, evaluando con {settings.openai_model} ({n_batches} lotes)...")

    for batch_num, batch_start in enumerate(range(0, len(frames), BATCH_SIZE), start=1):
        if batch_start < resume_from:
            continue
        if batch_start > 0:
            time.sleep(BATCH_PAUSE_SECONDS)
        batch = frames[batch_start:batch_start + BATCH_SIZE]
        scored = _score_batch(client, batch, video_stem)
        for item in scored:
            idx = item["index"]
            if not (0 <= idx < len(batch)):
                continue
            t = batch[idx][0]
            results.append({
                "start": round(t, 2),
                "end": round(t + interval, 2),
                "score": item["score"],
                "description": item["description"],
            })
        partial_path.write_text(json.dumps({
            "interval": interval,
            "n_frames": len(frames),
            "next_batch_start": batch_start + BATCH_SIZE,
            "results": results,
        }, ensure_ascii=False), encoding="utf-8")
        if batch_num % 5 == 0 or batch_num == n_batches:
            print(f"[visual] lote {batch_num}/{n_batches} evaluado")

    partial_path.unlink(missing_ok=True)
    notable = [r for r in results if r["score"] >= MIN_NOTABLE_SCORE]
    notable.sort(key=lambda r: r["start"])
    print(f"[visual] {len(notable)}/{len(results)} frames notables (score >= {MIN_NOTABLE_SCORE})")

    if notable:
        print(f"[visual] refinando descripción de {len(notable)} momentos notables (detalle alto)...")
        _refine_notable_descriptions(client, video_path, notable, video_stem)

    out_path.write_text(json.dumps(notable, ensure_ascii=False, indent=2), encoding="utf-8")
    return notable
