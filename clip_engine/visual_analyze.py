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

from openai import OpenAI, RateLimitError

from .config import TRANSCRIPTS_DIR, settings

FRAME_INTERVAL_SECONDS = 3.0
BATCH_SIZE = 15
BATCH_PAUSE_SECONDS = 1.5
MIN_NOTABLE_SCORE = 6

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


def _score_batch(client: OpenAI, batch: list[tuple[float, bytes]]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": f"Te mando {len(batch)} frames, en orden (índice 0 a {len(batch) - 1})."}
    ]
    for _, img_bytes in batch:
        b64 = base64.b64encode(img_bytes).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
        })

    max_attempts = 12
    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": VISUAL_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                response_format={"type": "json_schema", "json_schema": VISUAL_SCHEMA},
                temperature=0.3,
            )
            return json.loads(response.choices[0].message.content)["frames"]
        except RateLimitError:
            # Backoff con techo: en una corrida larga (cientos de lotes) no
            # tiene sentido tirar toda la corrida abajo por un par de picos
            # de rate limit — esperamos más y reintentamos más veces en vez
            # de rendirnos.
            wait = min(60, 5 * (attempt + 1))
            print(f"[visual] rate limit, esperando {wait}s... (intento {attempt + 1}/{max_attempts})")
            time.sleep(wait)
    raise RuntimeError("Rate limit persistente evaluando frames visuales, probá de nuevo en un rato.")


def analyze_visual(video_path: Path, video_stem: str, force: bool = False) -> list[dict[str, Any]]:
    """Devuelve una lista de {start, end, score, description} para tramos del
    video con algo visualmente notable. Cachea el resultado igual que la
    transcripción, para no re-pagar la llamada a la API en corridas siguientes.

    Va secuencial, no en paralelo: el límite de tokens/minuto es un techo
    fijo de la cuenta, no algo que se esquive mandando varios pedidos a la
    vez — probamos paralelizar y lo único que logró fue que varios pedidos
    chocaran contra el límite al mismo tiempo y se agotaran los reintentos.
    """
    out_path = TRANSCRIPTS_DIR / f"{video_stem}.visual.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))

    if not settings.openai_api_key:
        raise RuntimeError("Falta OPENAI_API_KEY (definila en tu archivo .env)")

    client = OpenAI(api_key=settings.openai_api_key)

    print(f"[visual] sampleando 1 frame cada {FRAME_INTERVAL_SECONDS:.0f}s...")
    frames = _extract_frames(video_path, FRAME_INTERVAL_SECONDS)
    n_batches = (len(frames) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"[visual] {len(frames)} frames extraídos, evaluando con {settings.openai_model} ({n_batches} lotes)...")

    results: list[dict[str, Any]] = []
    for batch_num, batch_start in enumerate(range(0, len(frames), BATCH_SIZE), start=1):
        if batch_start > 0:
            time.sleep(BATCH_PAUSE_SECONDS)
        batch = frames[batch_start:batch_start + BATCH_SIZE]
        scored = _score_batch(client, batch)
        for item in scored:
            idx = item["index"]
            if not (0 <= idx < len(batch)):
                continue
            t = batch[idx][0]
            results.append({
                "start": round(t, 2),
                "end": round(t + FRAME_INTERVAL_SECONDS, 2),
                "score": item["score"],
                "description": item["description"],
            })
        if batch_num % 5 == 0 or batch_num == n_batches:
            print(f"[visual] lote {batch_num}/{n_batches} evaluado")

    notable = [r for r in results if r["score"] >= MIN_NOTABLE_SCORE]
    notable.sort(key=lambda r: r["start"])
    print(f"[visual] {len(notable)}/{len(results)} frames notables (score >= {MIN_NOTABLE_SCORE})")

    out_path.write_text(json.dumps(notable, ensure_ascii=False, indent=2), encoding="utf-8")
    return notable
