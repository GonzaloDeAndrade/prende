"""Batch de investigación: descarga + transcribe + analiza visualmente un
corpus de clips virales reales + sus videos largos de origen, para sacar
patrones reales en vez de intuición. Corre standalone, separado del
pipeline principal — no toca data/input ni la webui.

Corre como daemon: procesa todo lo "pending" del manifest, y si no le queda
nada, espera y vuelve a leer el archivo por si se agregaron items nuevos
(así una tanda nueva de URLs se suma sola sin tener que reiniciar esto).
Sale solo después de un rato largo sin nada nuevo que hacer.

IMPORTANTE sobre concurrencia: cada item puede tardar varios minutos en
procesarse (descarga + transcripción + análisis visual). En ese tiempo el
archivo manifest.json puede cambiar por fuera (otra tanda de URLs agregada a
mano). Por eso NUNCA se pisa el archivo entero con una copia en memoria
vieja — cada guardado relee el archivo fresco del disco y mezcla ahí el
estado actualizado de un solo item. Ya nos comimos una pérdida de ~1500
items una vez esta noche por no hacer esto bien.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import truststore

truststore.inject_into_ssl()

from yt_dlp_helper import NODE22_PATH, YT_DLP_FIX_ARGS

# Los títulos reales de YouTube (sobre todo de canales de shorts) traen
# emojis a menudo — la consola/redirección de Windows por defecto usa
# cp1252, que no los puede codificar y tira UnicodeEncodeError, tumbando
# todo el proceso en un simple print(). Forzamos UTF-8 con reemplazo en vez
# de crash.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
RESEARCH_DIR = ROOT / "data" / "research"
VIDEOS_DIR = RESEARCH_DIR / "videos"
TRANSCRIPTS_DIR = RESEARCH_DIR / "transcripts"
MANIFEST_PATH = RESEARCH_DIR / "manifest.json"

VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

# Sampleo mas liviano que el de produccion para los videos largos (muchos
# son 30-60+ min) — acá el objetivo es encontrar patrones en el corpus, no
# forzar candidatos de clip con precisión quirúrgica, así que no hace falta
# la misma densidad y ahorra tiempo/costo real en una tanda de decenas de
# videos largos.
CLIP_VISUAL_INTERVAL = 3.0
LARGO_VISUAL_INTERVAL = 8.0
LARGO_VISUAL_MAX_DURATION = 3600 * 1.5  # no analizamos visualmente lo que pase de 1.5h, muy caro para el propósito

IDLE_EXIT_SECONDS = 45 * 60  # si no aparece nada nuevo en 45 min, corta solo
IDLE_POLL_SECONDS = 60

# YouTube nos tiró 403 varias veces esta noche incluso con pausa de 12s —
# vamos más conservador ahora que no hay apuro (puede tardar días, está bien).
#
# Bajado de 25 a 10 el 2026-08-23: esta pausa corre DESPUÉS DE CADA ITEM sin
# importar su tamaño (un short de 30s paga la misma pausa que un podcast de
# 4 horas) — con la cola de esa noche en ~30 items no se notaba, pero al
# crecer a ~480 items (268 clips + 211 largos, pedido por el usuario) la
# pausa sola ya sumaba ~3.3 horas de espera pura, aparte del trabajo real.
# 10s sigue siendo una pausa real entre pedidos (no es sacarla del todo,
# el bloqueo de 403 fue real), pero escala mejor con una cola de este tamaño.
DOWNLOAD_PAUSE_SECONDS = 10.0
_MAX_DOWNLOAD_ATTEMPTS = 2


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def save_manifest(m: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


def _find_item(manifest: dict, bucket: str, item_id: str) -> dict | None:
    for it in manifest.get(bucket, []):
        if it["id"] == item_id:
            return it
    return None


def persist_item(item: dict, bucket: str) -> None:
    """Guarda el estado actual de UN item, mezclado con lo que haya en disco
    en este momento — nunca pisa el archivo entero con una copia en memoria
    vieja. Si el item no existe más en el archivo fresco (no debería pasar,
    pero por las dudas) lo re-agrega en vez de perder el progreso."""
    fresh = load_manifest()
    fresh_item = _find_item(fresh, bucket, item["id"])
    if fresh_item is not None:
        fresh_item.update(item)
    else:
        fresh[bucket].append(dict(item))
    save_manifest(fresh)


def get_heatmap(url: str) -> list[dict] | None:
    import yt_dlp

    # Encontrado en vivo (2026-08-23): esto usaba {"node": {}} sin el path
    # explícito del Node 22 portátil (a diferencia de download_video(), que
    # sí pasa YT_DLP_FIX_ARGS) — sin él, yt-dlp no encuentra un runtime JS
    # compatible (el Node del sistema es v20) y el challenge de YouTube
    # falla en silencio, guardando heatmap=None. No es que el video no
    # tuviera heatmap — nunca se llegó a pedir bien. Mismo fix que ya usa
    # clip_engine/heatmap.py para el mismo problema del lado de la app.
    ydl_opts = {
        "js_runtimes": {"node": {"path": str(NODE22_PATH)}},
        "extractor_args": {"youtube": {"player_client": ["android", "tv"]}},
        "quiet": True,
        "skip_download": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get("heatmap")
    except Exception:
        return None


def _is_block_error(err: str) -> bool:
    """Si el error es un bloqueo/rate-limit de YouTube (vale la pena
    enfriar y reintentar más tarde) en vez de un fallo real del video
    puntual (video privado/borrado/etc, no vale la pena insistir).

    "403"/"Forbidden" fue el único síntoma la primera vez que nos bloquearon
    esta noche. Después, con el fix del runtime de JS, apareció un SEGUNDO
    síntoma del mismo tipo de bloqueo ("Sign in to confirm you're not a
    bot") que no contiene ninguna de esas dos palabras — el enfriamiento
    nunca se disparaba y el daemon chocaba en bucle contra el mismo bloqueo
    sin pausar. Cualquier síntoma nuevo de bloqueo que aparezca después va
    acá, no se asume que ya vimos todos."""
    markers = ("403", "forbidden", "sign in to confirm", "not a bot")
    err_lower = err.lower()
    return any(m in err_lower for m in markers)


def download_video(url: str, item_id: str) -> Path | None:
    dest_pattern = str(VIDEOS_DIR / f"{item_id}.%(ext)s")
    cmd = [
        sys.executable, "-c",
        "import truststore; truststore.inject_into_ssl(); from yt_dlp import main; main()",
        *YT_DLP_FIX_ARGS,
        "-f", "bv*[height<=1080]+ba/b[height<=1080]",
        "-o", dest_pattern,
        "--merge-output-format", "mp4",
        "--print", "after_move:filepath",
        url,
    ]
    last_err = ""
    for attempt in range(_MAX_DOWNLOAD_ATTEMPTS):
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if result.returncode == 0:
            lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
            if lines:
                return Path(lines[-1])
            last_err = "yt-dlp no devolvio la ruta del archivo"
        else:
            last_err = result.stderr[-1500:] or "yt-dlp fallo sin stderr"
        if _is_block_error(last_err):
            wait = 45 * (attempt + 1)
            print(f"  [dl] bloqueo de YouTube, esperando {wait}s antes de reintentar (intento {attempt + 1}/{_MAX_DOWNLOAD_ATTEMPTS})...")
            time.sleep(wait)
        else:
            break  # error no relacionado a rate-limit, no vale la pena reintentar igual
    raise RuntimeError(last_err)


def transcribe_video(video_path: Path, item_id: str) -> dict:
    from clip_engine.transcribe import transcribe as _transcribe
    out_path = TRANSCRIPTS_DIR / f"{item_id}.json"
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    result = _transcribe(video_path, force=False)
    # transcribe() ya cachea en data/transcripts con el stem del archivo,
    # pero acá lo copiamos también a research/transcripts para tener todo
    # el corpus junto y no mezclarlo con los videos de trabajo normales.
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def analyze_energy_video(video_path: Path, item_id: str) -> list[tuple[float, float, float]]:
    from clip_engine.audio_energy import detect_energy_spikes as _detect_energy_spikes
    out_path = TRANSCRIPTS_DIR / f"{item_id}.energy.json"
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    result = _detect_energy_spikes(video_path, item_id, force=False)
    out_path.write_text(json.dumps(result), encoding="utf-8")
    return result


def visually_analyze_video(video_path: Path, item_id: str, is_clip: bool, duration: float | None) -> list[dict]:
    from clip_engine.visual_analyze import analyze_visual as _analyze_visual
    out_path = TRANSCRIPTS_DIR / f"{item_id}.visual.json"
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    interval = CLIP_VISUAL_INTERVAL if is_clip else LARGO_VISUAL_INTERVAL
    result = _analyze_visual(video_path, item_id, force=False, interval=interval)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def process_item(item: dict, is_clip: bool, bucket: str) -> str:
    """Devuelve 'ok', 'error_403' (fallo de descarga por bloqueo/rate-limit
    de YouTube — el llamador usa esto para saber cuándo hay que enfriar en
    vez de seguir insistiendo) u 'error_other' (cualquier otro fallo, no
    relacionado a bloqueo, no amerita esperar más de lo normal)."""
    item_id = item["id"]
    print(f"\n=== {'CLIP' if is_clip else 'LARGO'} {item_id}: {item['title']} ===")

    # Si el daemon se reinició (o lo maté) mientras este item estaba a
    # mitad de una etapa, queda con un status "en curso" que ninguna de las
    # condiciones de abajo reconoce como punto de partida — sin esto, el
    # item queda huérfano para siempre: process_item no hace nada, devuelve
    # "ok", y el loop principal vuelve a elegir el MISMO item de nuevo en
    # la próxima vuelta. En la práctica esto trabó el daemon en un bucle
    # infinito sobre 2 items sin avanzar nunca con el resto de la cola.
    # Lo arreglamos retrocediendo el status al último punto seguro
    # (la etapa anterior ya terminada) para que se reintente esa etapa.
    _RESUME_FROM = {
        "descargando": "pending",
        "transcribiendo": "descargado",
        "analizando audio": "transcripto",
        "analizando visual": "audio listo",
    }
    if item["status"] in _RESUME_FROM:
        print(f"  (retomando: estaba en '{item['status']}' de una corrida anterior)")
        item["status"] = _RESUME_FROM[item["status"]]
        persist_item(item, bucket)

    try:
        if item["status"] in ("pending",):
            item["status"] = "descargando"
            persist_item(item, bucket)
            try:
                video_path = download_video(item["url"], item_id)
            except RuntimeError as exc:
                if _is_block_error(str(exc)):
                    item["status"] = "error"
                    item["error"] = str(exc)[:500]
                    persist_item(item, bucket)
                    return "error_403"
                raise
            item["local_path"] = str(video_path)
            item["status"] = "descargado"
            persist_item(item, bucket)
            print(f"  descargado: {video_path}")

        if "youtube.com" in item["url"] and "heatmap" not in item:
            hm = get_heatmap(item["url"])
            item["heatmap"] = hm
            n_points = len(hm) if hm else 0
            print(f"  heatmap: {n_points} puntos" if hm else "  heatmap: no disponible")
            persist_item(item, bucket)

        if item["status"] == "descargado":
            video_path = Path(item["local_path"])
            if not video_path.exists():
                item["status"] = "error"
                item["error"] = "archivo descargado no encontrado"
                persist_item(item, bucket)
                return "error_other"
            item["status"] = "transcribiendo"
            persist_item(item, bucket)
            t = transcribe_video(video_path, item_id)
            item["duration"] = t.get("duration")
            item["status"] = "transcripto"
            persist_item(item, bucket)
            print(f"  transcripto, duracion {t.get('duration'):.1f}s")

        if item["status"] == "transcripto":
            video_path = Path(item["local_path"])
            item["status"] = "analizando audio"
            persist_item(item, bucket)
            spikes = analyze_energy_video(video_path, item_id)
            item["energy_spike_count"] = len(spikes)
            item["status"] = "audio listo"
            persist_item(item, bucket)
            print(f"  analisis de audio listo: {len(spikes)} picos de energia")

        if item["status"] == "audio listo":
            video_path = Path(item["local_path"])
            duration = item.get("duration") or 0
            if not is_clip and duration > LARGO_VISUAL_MAX_DURATION:
                print(f"  salteando analisis visual: {duration:.0f}s supera el limite de {LARGO_VISUAL_MAX_DURATION:.0f}s")
                item["status"] = "listo"
                item["visual_skipped"] = True
                persist_item(item, bucket)
            else:
                item["status"] = "analizando visual"
                persist_item(item, bucket)
                notable = visually_analyze_video(video_path, item_id, is_clip, duration)
                item["visual_notable_count"] = len(notable)
                item["status"] = "listo"
                persist_item(item, bucket)
                print(f"  analisis visual listo: {len(notable)} momentos notables")

        return "ok"

    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        item["status"] = "error"
        item["error"] = str(exc)[:500]
        persist_item(item, bucket)
        # "Fallo persistente en visual-..." = call_with_backoff ya agotó sus
        # 12 reintentos contra un rate-limit de OpenAI (cuenta entera, no por
        # fuente como YouTube/Kick). Vimos un video de 5+ horas con 1000+
        # momentos notables dejar la cuenta tan ajustada que el SIGUIENTE
        # item, uno cualquiera, chocaba contra el mismo límite todavía
        # caliente — perdiendo ~6 min de reintentos sin lograr nada, dos
        # veces seguidas. Se enfría a nivel daemon igual que con YouTube.
        if "Fallo persistente en visual-" in str(exc):
            return "error_openai_ratelimit"
        return "error_other"


LOCK_PATH = RESEARCH_DIR / "batch.lock"


def _acquire_lock() -> None:
    # Guarda contra lanzar dos corridas en paralelo por error — ya pasó una
    # vez esta noche y duplicó cada pedido a YouTube, lo que aceleró el
    # rate-limit en vez de evitarlo.
    if LOCK_PATH.exists():
        pid = LOCK_PATH.read_text(encoding="utf-8").strip()
        raise RuntimeError(
            f"Ya hay (o hubo) una corrida con lock activo (pid guardado: {pid}). "
            f"Si estás seguro de que no hay otra corriendo, borrá {LOCK_PATH} y reintentá."
        )
    import os
    LOCK_PATH.write_text(str(os.getpid()), encoding="utf-8")


def _release_lock() -> None:
    LOCK_PATH.unlink(missing_ok=True)


def _url_source(url: str) -> str:
    if "kick.com" in url:
        return "kick.com"
    if "youtube.com" in url:
        return "youtube.com"
    return "other"


_LARGO_EVERY = 8  # cada tantos items procesados, intercalamos un intento de video largo
# Los largos tienen el heatmap de "más reproducido" — la señal más valiosa
# que tenemos y que NO existe para los clips sueltos. Sin esto, con ~1650
# clips en cola, los 23 largos nunca arrancarían hasta terminar todos los
# clips primero (podían ser 50+ horas de espera para ese dato).


def _next_clip(manifest: dict, blocked_until: dict[str, float], now: float):
    for item in manifest["clips"]:
        if item["status"] not in ("listo", "error") and blocked_until.get(_url_source(item["url"]), 0) <= now:
            return item, True, "clips"
    return None


def _next_largo(manifest: dict, blocked_until: dict[str, float], now: float):
    largos_sorted = sorted(manifest["largos"], key=lambda x: not x.get("priority", False))
    for item in largos_sorted:
        if item["status"] not in ("listo", "error") and blocked_until.get(_url_source(item["url"]), 0) <= now:
            return item, False, "largos"
    return None


def _next_pending(manifest: dict, blocked_until: dict[str, float], tick: int) -> tuple[dict, bool, str] | None:
    """SalteA items cuya fuente esté en enfriamiento ahora mismo — así, si
    YouTube nos frena pero hay pendientes de otra fuente (Kick, etc.), esos
    siguen avanzando en vez de que todo el daemon quede esperando por algo
    que no le pasa a esa otra fuente. Además intercala videos largos cada
    `_LARGO_EVERY` items en vez de dejarlos todos para el final."""
    now = time.time()
    prefer_largo = (tick % _LARGO_EVERY == 0)
    first, second = (_next_largo, _next_clip) if prefer_largo else (_next_clip, _next_largo)
    return first(manifest, blocked_until, now) or second(manifest, blocked_until, now)


def _has_any_pending(manifest: dict) -> bool:
    return any(i["status"] not in ("listo", "error") for i in manifest["clips"] + manifest["largos"])


_COOLDOWN_TRIGGER = 3  # esta cantidad de 403 seguidos (de la MISMA fuente) dispara el enfriamiento largo
_COOLDOWN_BASE_SECONDS = 25 * 60
_COOLDOWN_MAX_SECONDS = 3 * 3600  # tope de 3h — si sigue bloqueado más que eso, ya no vale la pena escalar más
_BLOCKED_NAP_SECONDS = 5 * 60  # mientras todo lo disponible está bloqueado, revisamos cada tanto por si se sumó algo nuevo

# El rate-limit de OpenAI es de TODA la cuenta, no por fuente como YouTube/
# Kick — no tiene sentido "saltar a otra fuente" cuando pasa esto, porque
# cualquier item choca contra el mismo límite en su propio paso de análisis
# visual. Directamente se pausa el daemon entero un rato.
_OPENAI_COOLDOWN_TRIGGER = 2
_OPENAI_COOLDOWN_BASE_SECONDS = 10 * 60
_OPENAI_COOLDOWN_MAX_SECONDS = 2 * 3600
_OPENAI_HEALTHCHECK_EVERY_SECONDS = 120  # cada cuanto revisar si ya se puede cortar la espera antes


def _openai_healthy() -> bool:
    """Pedido mínimo (1 token) para saber si ya se puede volver a pegarle a
    la API — sirve tanto para "se liberó el rate-limit" como para "ya
    cargaron crédito". Nos pasó una vez: el usuario cargó crédito a la mitad
    de un enfriamiento de 80 min y el daemon durmió la espera completa
    igual, sin necesidad — con esto corta apenas se puede seguir."""
    try:
        from clip_engine.config import settings
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": "ok"}],
            max_tokens=1,
        )
        return True
    except Exception:
        return False


def _cooldown_sleep_with_healthcheck(total_seconds: float) -> None:
    """Como time.sleep(total_seconds), pero revisando cada tanto si ya se
    puede cortar la espera antes de tiempo."""
    elapsed = 0.0
    while elapsed < total_seconds:
        chunk = min(_OPENAI_HEALTHCHECK_EVERY_SECONDS, total_seconds - elapsed)
        time.sleep(chunk)
        elapsed += chunk
        if elapsed < total_seconds and _openai_healthy():
            print(f"[daemon] la API de OpenAI ya responde de nuevo (a los {elapsed / 60:.0f} min) — corto el enfriamiento antes de tiempo")
            return

# Windows tiene un límite de recursos (desktop heap / handles) POR PROCESO
# para crear procesos hijos — un daemon que corre muchas horas y lanza
# subprocess.run (yt-dlp, ffmpeg) miles de veces de a poco lo agota, aunque
# el sistema tenga RAM libre de sobra. Lo vimos en carne propia: una noche
# entera, 554 items fallaron con "WinError 8: no hay suficientes recursos
# de memoria" apenas el proceso llevaba muchas horas vivo — mientras que un
# proceso NUEVO podía lanzar subprocess sin problema en la misma máquina en
# el mismo momento. La corrida en sí no tenía nada malo, era el proceso
# viejo el que estaba agotado. Por eso el daemon se retira solo cada tantos
# items procesados, ANTES de llegar a ese límite — el wrapper que lo lanza
# (ver el .bat/.sh de arranque) lo vuelve a levantar automáticamente,
# fresco, sin que haga falta que alguien lo note y lo reinicie a mano.
_MAX_ITEMS_PER_RUN = 100  # conservador a propósito: perder una corrida por reiniciar
                          # de más sale barato, perder una noche entera por reiniciar
                          # de menos (lo que pasó) sale carísimo


def main():
    _acquire_lock()
    idle_since = None
    consecutive_403: dict[str, int] = {}
    cooldown_rounds: dict[str, int] = {}
    blocked_until: dict[str, float] = {}
    consecutive_openai = 0
    openai_cooldown_rounds = 0
    tick = 0
    items_this_run = 0
    try:
        while True:
            manifest = load_manifest()
            nxt = _next_pending(manifest, blocked_until, tick)
            tick += 1

            if nxt is None:
                if _has_any_pending(manifest):
                    # hay trabajo pero todo lo disponible ahora mismo es de
                    # una fuente en enfriamiento — dormimos un rato corto
                    # (no el enfriamiento entero de una) para poder
                    # reaccionar rápido si se agrega algo de otra fuente.
                    idle_since = None
                    soonest = min(blocked_until.values()) if blocked_until else time.time()
                    mins_left = max(0, (soonest - time.time()) / 60)
                    print(f"\n[daemon] todo lo pendiente disponible está en enfriamiento (~{mins_left:.0f} min restantes en la fuente más próxima), reviso de nuevo en {_BLOCKED_NAP_SECONDS // 60} min...")
                    time.sleep(_BLOCKED_NAP_SECONDS)
                    continue
                if idle_since is None:
                    idle_since = time.time()
                    print(f"\n[daemon] cola vacia, esperando nuevos items (corta solo tras {IDLE_EXIT_SECONDS // 60} min sin nada nuevo)...")
                elif time.time() - idle_since > IDLE_EXIT_SECONDS:
                    print("\n[daemon] mucho tiempo sin trabajo nuevo, cierro.")
                    break
                time.sleep(IDLE_POLL_SECONDS)
                continue

            idle_since = None
            item, is_clip, bucket = nxt
            src = _url_source(item["url"])
            try:
                outcome = process_item(item, is_clip, bucket)
            except Exception as exc:  # noqa: BLE001
                # Red de seguridad de último recurso: process_item ya tiene su
                # propio try/except, pero algo puede fallar ANTES de entrar
                # ahí (como el print del título con un emoji que tumbó todo
                # el daemon una vez). Un solo item roto nunca tiene que poder
                # matar la corrida entera de nuevo.
                traceback.print_exc()
                try:
                    item["status"] = "error"
                    item["error"] = f"fallo no capturado: {exc}"[:500]
                    persist_item(item, bucket)
                except Exception:
                    pass
                outcome = "error_other"

            if outcome == "error_403":
                consecutive_403[src] = consecutive_403.get(src, 0) + 1
                if consecutive_403[src] >= _COOLDOWN_TRIGGER:
                    rounds = cooldown_rounds.get(src, 0)
                    wait = min(_COOLDOWN_MAX_SECONDS, _COOLDOWN_BASE_SECONDS * (2 ** rounds))
                    cooldown_rounds[src] = rounds + 1
                    blocked_until[src] = time.time() + wait
                    print(
                        f"\n[daemon] {consecutive_403[src]} bloqueos 403 seguidos en '{src}' — "
                        f"enfriando esa fuente {wait // 60} min (ronda #{rounds + 1}). "
                        f"Sigo con otras fuentes mientras tanto si hay."
                    )
                    consecutive_403[src] = 0
            else:
                consecutive_403[src] = 0
                cooldown_rounds[src] = 0  # se logró avanzar de nuevo en esta fuente, reseteamos la escalada

            if outcome == "error_openai_ratelimit":
                consecutive_openai += 1
                if consecutive_openai >= _OPENAI_COOLDOWN_TRIGGER:
                    wait = min(_OPENAI_COOLDOWN_MAX_SECONDS, _OPENAI_COOLDOWN_BASE_SECONDS * (2 ** openai_cooldown_rounds))
                    openai_cooldown_rounds += 1
                    print(
                        f"\n[daemon] {consecutive_openai} fallos de rate-limit de OpenAI seguidos — "
                        f"enfriando TODO el daemon {wait // 60} min (ronda #{openai_cooldown_rounds}), "
                        f"la cuenta entera está saturada, no hay otra fuente a la que saltar."
                    )
                    consecutive_openai = 0
                    _cooldown_sleep_with_healthcheck(wait)
            else:
                consecutive_openai = 0
                openai_cooldown_rounds = 0

            items_this_run += 1
            if items_this_run >= _MAX_ITEMS_PER_RUN:
                print(
                    f"\n[daemon] {items_this_run} items procesados en esta corrida — me retiro solo "
                    f"antes de agotar recursos del proceso (ver comentario de _MAX_ITEMS_PER_RUN). "
                    f"El wrapper me vuelve a levantar fresco."
                )
                break

            time.sleep(DOWNLOAD_PAUSE_SECONDS)

        final = load_manifest()
        print("\n=== RESUMEN FINAL ===")
        for bucket in ("clips", "largos"):
            listo = sum(1 for i in final[bucket] if i["status"] == "listo")
            error = sum(1 for i in final[bucket] if i["status"] == "error")
            print(f"{bucket}: {listo} listos, {error} con error, {len(final[bucket])} total")
    finally:
        _release_lock()


if __name__ == "__main__":
    main()
