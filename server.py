"""Servidor local para revisar candidatos de clips y elegir cuáles cortar.

Reemplaza el flujo 100% automático por uno asistido: se analiza el video,
se muestran TODOS los candidatos evaluados (no solo los que pasaron el
filtro) con su score/razón, y un humano elige cuáles convertir en el clip
vertical final con subtítulos.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import threading
import traceback
import uuid
import zipfile
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_file, send_from_directory

from clip_engine.analyze import analyze, apply_micro_cuts, detect_micro_cuts
from clip_engine.config import CLIPS_DIR, DATA_DIR, INPUT_DIR, TMP_DIR, TRANSCRIPTS_DIR
from clip_engine.cost_tracker import cost_summary
from clip_engine.cutter import render_clip, render_filmstrip, render_preview
from clip_engine.subtitles import SUBTITLE_COLOR_PRESETS, build_clip_subtitles


def _subtitle_color(clip: dict[str, Any]) -> str:
    return SUBTITLE_COLOR_PRESETS.get(clip.get("subtitle_color", ""), SUBTITLE_COLOR_PRESETS["amarillo"])
from clip_engine.transcribe import transcribe
from clip_engine.video_validate import get_duration_seconds, validate_video
from yt_dlp_helper import YT_DLP_FIX_ARGS

app = Flask(__name__)

_jobs: dict[str, dict[str, Any]] = {}
_downloads: dict[str, dict[str, Any]] = {}

_VIDEO_NAMES_PATH = DATA_DIR / "video_names.json"


def _load_video_names() -> dict[str, str]:
    if not _VIDEO_NAMES_PATH.exists():
        return {}
    try:
        return json.loads(_VIDEO_NAMES_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_video_name(stem: str, name: str) -> None:
    names = _load_video_names()
    if name.strip():
        names[stem] = name.strip()
    else:
        names.pop(stem, None)
    _VIDEO_NAMES_PATH.write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")


def _slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:60] or "clip"


def _instruction_digest(custom_instruction: str | None) -> str:
    if not custom_instruction:
        return ""
    return hashlib.sha1(custom_instruction.strip().lower().encode("utf-8")).hexdigest()[:8]


def _candidates_path(video_stem: str, category: str, custom_instruction: str | None = None) -> Path:
    # Mismo esquema de nombres que usa analyze.py para cachear en disco —
    # tiene que coincidir exacto, si no el server busca en un lado y
    # analyze() escribe en otro. `target_clips` (cuántos clips pidió el
    # usuario) NO forma parte de esta clave a propósito — es un parámetro de
    # la corrida, no de identidad del resultado; cambiar la cantidad pedida
    # reusa el mismo caché salvo que se togue "Re-analizar" (fuerza fresco).
    # Sumarlo a la clave hubiera significado tocar ~25 rutas que la usan.
    category_tag = f".{category}" if category != "general" else ""
    digest = _instruction_digest(custom_instruction)
    custom_tag = f".custom-{digest}" if digest else ""
    return TRANSCRIPTS_DIR / f"{video_stem}{category_tag}{custom_tag}.candidates.json"


def _variant(category: str, custom_instruction: str | None = None) -> str:
    """Nombre de variante para carpetas de previews/finales y para la clave
    de job en progreso — no necesita coincidir con analyze.py (esas carpetas
    las maneja solo el server), solo tiene que ser estable y sin pisarse
    entre categoría/búsqueda distintas del mismo video."""
    digest = _instruction_digest(custom_instruction)
    return f"{category}-custom-{digest}" if digest else category


def _preview_dir(video_stem: str, variant: str) -> Path:
    return TMP_DIR / video_stem / "previews" / (variant or "general")


def _preview_path(video_stem: str, variant: str, index: int, title: str) -> Path:
    slug = _slugify(title)
    return _preview_dir(video_stem, variant) / f"{index:02d}_{slug}.mp4"


def _filmstrip_path(video_stem: str, variant: str, index: int, title: str) -> Path:
    slug = _slugify(title)
    return _preview_dir(video_stem, variant) / f"{index:02d}_{slug}_strip.jpg"


def _style_cache_paths(video_stem: str, variant: str, index: int, title: str, style: str) -> tuple[Path, Path]:
    """Cada nivel de ritmo (sin_cortes/balanceado/acelerado) siempre da el
    MISMO resultado para un clip dado (set_edit_style recalcula siempre
    desde el corte original, nunca desde el estado actual) — así que una vez
    renderizado un nivel, se puede reusar sin volver a pasar por ffmpeg la
    próxima vez que el usuario vuelva a ese mismo nivel."""
    slug = _slugify(title)
    base = _preview_dir(video_stem, variant) / f"{index:02d}_{slug}_style-{style}"
    return base.with_suffix(".mp4"), Path(str(base) + "_strip.jpg")


def _final_dir(video_stem: str, variant: str) -> Path:
    return CLIPS_DIR / video_stem / (variant or "general")


def _final_path(video_stem: str, variant: str, index: int, title: str) -> Path:
    slug = _slugify(title)
    return _final_dir(video_stem, variant) / f"{index:02d}_{slug}.mp4"


def _job_key(video_stem: str, variant: str) -> str:
    return f"{video_stem}.{variant}"


@app.route("/")
def index():
    return send_from_directory(Path(__file__).parent / "webui", "index.html")


@app.route("/assets/<path:filename>")
def assets(filename: str):
    return send_from_directory(Path(__file__).parent / "webui" / "assets", filename)


@app.route("/api/upload", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return jsonify({"error": "no se mandó ningún archivo"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "archivo sin nombre"}), 400
    stem = _slugify(Path(f.filename).stem)
    if not stem:
        return jsonify({"error": "nombre de archivo inválido"}), 400
    dest = INPUT_DIR / f"{stem}.mp4"
    f.save(dest)
    ok, msg = validate_video(dest)
    if not ok:
        dest.unlink(missing_ok=True)
        return jsonify({"error": f"video inválido: {msg}"}), 400
    return jsonify({"stem": stem, "name": dest.name})


UPLOADS_TMP_DIR = TMP_DIR / "_uploads"
UPLOADS_TMP_DIR.mkdir(parents=True, exist_ok=True)


def _upload_id_slug(raw_id: str) -> str:
    # El id lo arma el cliente (nombre+tamaño del archivo) para que, si la
    # subida se corta y el usuario reintenta con el mismo archivo, el server
    # reconozca que es la misma subida y le diga desde dónde retomar — sin
    # necesitar sesión ni login. Se sanitiza igual que cualquier input que
    # termina siendo parte de una ruta de archivo.
    slug = re.sub(r"[^\w.-]", "_", raw_id)[:200]
    if not slug:
        raise ValueError("upload_id inválido")
    return slug


def _partial_upload_path(upload_id: str) -> Path:
    return UPLOADS_TMP_DIR / f"{_upload_id_slug(upload_id)}.part"


@app.route("/api/upload_status/<upload_id>")
def upload_status(upload_id: str):
    """Cuántos bytes ya recibió el server de esta subida — el cliente lo
    consulta antes de mandar chunks para saber si está arrancando de cero o
    retomando una subida que se cortó a mitad de camino."""
    try:
        path = _partial_upload_path(upload_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    received = path.stat().st_size if path.exists() else 0
    return jsonify({"received_bytes": received})


@app.route("/api/upload_chunk/<upload_id>", methods=["POST"])
def upload_chunk(upload_id: str):
    """Recibe un pedazo del archivo y lo agrega al final del archivo parcial,
    siempre que el offset que manda el cliente coincida con lo que el server
    ya tiene — así un chunk duplicado (reintento) o fuera de orden no corrompe
    el archivo resultante."""
    try:
        path = _partial_upload_path(upload_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        offset = int(request.args.get("offset", "-1"))
    except ValueError:
        return jsonify({"error": "offset inválido"}), 400

    current_size = path.stat().st_size if path.exists() else 0
    if offset != current_size:
        return jsonify({"error": "offset desincronizado", "received_bytes": current_size}), 409

    chunk = request.get_data()
    with open(path, "ab") as fh:
        fh.write(chunk)
    return jsonify({"received_bytes": current_size + len(chunk)})


@app.route("/api/upload_complete", methods=["POST"])
def upload_complete():
    """Cierra una subida por chunks: valida que el tamaño final coincida con
    lo esperado y mueve el archivo parcial a data/input/ con su nombre real."""
    body = request.get_json(silent=True) or {}
    upload_id = body.get("upload_id", "")
    filename = body.get("filename", "")
    total_size = body.get("total_size")
    try:
        path = _partial_upload_path(upload_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if not path.exists():
        return jsonify({"error": "no hay ninguna subida en curso con ese id"}), 404
    actual_size = path.stat().st_size
    if total_size is not None and actual_size != total_size:
        return jsonify({"error": f"tamaño incompleto: {actual_size}/{total_size} bytes", "received_bytes": actual_size}), 409

    stem = _slugify(Path(filename).stem) if filename else ""
    if not stem:
        return jsonify({"error": "nombre de archivo inválido"}), 400
    dest = INPUT_DIR / f"{stem}.mp4"
    path.replace(dest)
    ok, msg = validate_video(dest)
    if not ok:
        dest.unlink(missing_ok=True)
        return jsonify({"error": f"video inválido: {msg}"}), 400
    return jsonify({"stem": stem, "name": dest.name})


def _run_youtube_download(job_id: str, url: str) -> None:
    try:
        _downloads[job_id] = {"status": "descargando (puede tardar varios minutos)", "error": None, "stem": None}
        cmd = [
            # yt-dlp corre en su propio proceso, así que el truststore.inject_into_ssl()
            # de config.py (que arregla el CERTIFICATE_VERIFY_FAILED de Avast) nunca lo
            # alcanza — se lo inyectamos acá también, antes de que yt-dlp arranque.
            sys.executable, "-c",
            "import truststore; truststore.inject_into_ssl(); from yt_dlp import main; main()",
            *YT_DLP_FIX_ARGS,
            "-f", "bv*[height<=1080]+ba/b[height<=1080]",
            "-o", str(INPUT_DIR / "%(id)s.%(ext)s"),
            "--merge-output-format", "mp4",
            "--restrict-filenames",
            "--print", "after_move:filepath",
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-2000:] or "yt-dlp falló")
        lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
        if not lines:
            raise RuntimeError("yt-dlp no devolvió la ruta del archivo descargado")
        downloaded_path = Path(lines[-1])
        stem = downloaded_path.stem
        ok, msg = validate_video(downloaded_path)
        if not ok:
            raise RuntimeError(f"la descarga terminó pero el video no sirve: {msg}")
        _downloads[job_id] = {"status": "listo", "error": None, "stem": stem}
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        _downloads[job_id] = {"status": "error", "error": str(exc), "stem": None}


@app.route("/api/download_youtube", methods=["POST"])
def download_youtube():
    body = request.get_json(silent=True) or {}
    url = (body.get("url") or "").strip()
    if not url:
        return jsonify({"error": "falta la url"}), 400
    job_id = uuid.uuid4().hex[:12]
    threading.Thread(target=_run_youtube_download, args=(job_id, url), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/download_status/<job_id>")
def download_status(job_id: str):
    return jsonify(_downloads.get(job_id, {"status": "sin iniciar", "error": None, "stem": None}))


@app.route("/api/videos")
def list_videos():
    names = _load_video_names()
    videos = []
    for f in sorted(INPUT_DIR.glob("*.mp4")):
        videos.append({
            "stem": f.stem,
            "name": f.name,
            "display_name": names.get(f.stem, f.stem),
            "duration": get_duration_seconds(f),
            "has_candidates": _candidates_path(f.stem, "general").exists(),
        })
    return jsonify(videos)


@app.route("/api/set_video_name/<video_stem>", methods=["POST"])
def set_video_name(video_stem: str):
    body = request.get_json(silent=True) or {}
    name = str(body.get("name", "")).strip()[:120]
    _save_video_name(video_stem, name)
    return jsonify({"ok": True, "display_name": name or video_stem})


_AUTO_CUT_MIN_GAP = 1.5  # antes 0.8 — quedó sin actualizar el 2026-08-20 cuando subí
# _EDIT_STYLE_MIN_GAP["balanceado"] al mismo valor; esta constante separada
# (la que se usa para el auto-recorte automático al generar el clip por
# primera vez, no la de los estilos manuales) seguía en 0.8. Encontrado en
# vivo el 2026-08-21: un candidato real quedó con 5 cortes/21.6s sacados de
# un tirón por este umbral viejo, comiéndose el remate real del clip.

_MAX_AUTO_CUTS = 3  # ver comentario en _generate_previews


def _generate_previews(
    video_stem: str, category: str, custom_instruction: str | None, transcript: dict, clips: list[dict[str, Any]],
) -> None:
    variant = _variant(category, custom_instruction)
    job_key = _job_key(video_stem, variant)
    preview_dir = _preview_dir(video_stem, variant)
    preview_dir.mkdir(parents=True, exist_ok=True)
    video_path = INPUT_DIR / f"{video_stem}.mp4"
    candidates_path = _candidates_path(video_stem, category, custom_instruction)
    changed = False

    for i, clip in enumerate(clips):
        _jobs[job_key] = {"status": f"generando vistas previas ({i + 1}/{len(clips)})", "error": None}
        out_path = _preview_path(video_stem, variant, i, clip["title"])
        strip_path = _filmstrip_path(video_stem, variant, i, clip["title"])
        if out_path.exists() and strip_path.exists():
            continue

        # Auto-recorte de pausas internas (nivel "balanceado") ANTES del
        # primer preview — el clip ya llega editado y ágil sin que el
        # usuario tenga que tocar nada. Sigue siendo reversible: el corte
        # original de la IA queda guardado en `original_parts` y "Restaurar"
        # lo devuelve tal cual. Nunca se pierde nada, solo se adelanta el
        # trabajo de recorte que el usuario probablemente hubiera aplicado
        # igual a mano.
        if "original_parts" not in clip:
            raw_parts = [tuple(p) for p in clip["parts"]]
            cuts = detect_micro_cuts(transcript["segments"], raw_parts, min_gap=_AUTO_CUT_MIN_GAP)
            if len(cuts) > _MAX_AUTO_CUTS:
                # Encontrado en vivo (2026-08-21): un candidato con 5 cortes
                # separados quedó en 6 pedacitos pegados (9.7s de 31s
                # originales) — un Frankenstein irreconocible que se comía
                # justo el remate real (la reacción fuerte llegaba DESPUÉS
                # del último corte). Muchos cortes en un mismo clip es señal
                # de que el tramo es demasiado discontinuo para el
                # auto-recorte automático — mejor dejarlo con las pausas
                # originales (el usuario puede recortar a mano si quiere)
                # que fragmentarlo solo.
                cuts = sorted(cuts, key=lambda c: c[1] - c[0], reverse=True)[:_MAX_AUTO_CUTS]
                cuts.sort(key=lambda c: c[0])
            if cuts:
                cut_parts = apply_micro_cuts(raw_parts, cuts)
                if cut_parts:
                    saved = sum(e - s for s, e in raw_parts) - sum(e - s for s, e in cut_parts)
                    clip["original_parts"] = clip["parts"]
                    clip["parts"] = [list(p) for p in cut_parts]
                    clip["duration"] = round(sum(e - s for s, e in cut_parts), 1)
                    clip["auto_cut_info"] = {"count": len(cuts), "saved_seconds": round(saved, 1)}
                    clip["edit_style"] = "balanceado"
                    changed = True

        ass_path = preview_dir / f"{out_path.stem}.ass"
        parts = [tuple(p) for p in clip["parts"]]
        duration = sum(e - s for s, e in parts)
        build_clip_subtitles(transcript["segments"], parts, ass_path, highlight_color=_subtitle_color(clip))
        render_preview(video_path, parts, ass_path, out_path)
        render_filmstrip(out_path, strip_path, duration)

    if changed:
        candidates_path.write_text(json.dumps(clips, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_analysis(
    video_stem: str, use_visual: bool, category: str, custom_instruction: str | None, force: bool,
    target_clips: int | None = None, speed: str = "completo",
) -> None:
    variant = _variant(category, custom_instruction)
    job_key = _job_key(video_stem, variant)
    stage = "validando archivo"
    try:
        _jobs[job_key] = {"status": stage, "error": None}
        video_path = INPUT_DIR / f"{video_stem}.mp4"
        ok, msg = validate_video(video_path)
        if not ok:
            raise RuntimeError(msg)

        stage = "transcribiendo"
        _jobs[job_key] = {"status": stage, "error": None}
        transcript = transcribe(video_path, force=force)
        stage = "analizando (esto puede tardar)"
        _jobs[job_key] = {"status": stage, "error": None}
        clips = analyze(
            transcript, video_stem, video_path=video_path, use_visual=use_visual, review=True,
            category=category, custom_instruction=custom_instruction, force=force,
            target_clips=target_clips, speed=speed,
        )
        stage = "generando vistas previas"
        _generate_previews(video_stem, category, custom_instruction, transcript, clips)
        _jobs[job_key] = {"status": "listo", "error": None}
    except Exception as exc:  # noqa: BLE001
        print(f"[pipeline] falló en etapa '{stage}' para {video_stem}:")
        traceback.print_exc()
        _jobs[job_key] = {"status": "error", "error": f"[{stage}] {exc}"}


def _run_previews_only(video_stem: str, category: str, custom_instruction: str | None) -> None:
    variant = _variant(category, custom_instruction)
    job_key = _job_key(video_stem, variant)
    stage = "generando vistas previas"
    try:
        transcript = json.loads((TRANSCRIPTS_DIR / f"{video_stem}.json").read_text(encoding="utf-8"))
        clips = json.loads(_candidates_path(video_stem, category, custom_instruction).read_text(encoding="utf-8"))
        _generate_previews(video_stem, category, custom_instruction, transcript, clips)
        _jobs[job_key] = {"status": "listo", "error": None}
    except Exception as exc:  # noqa: BLE001
        print(f"[pipeline] falló en etapa '{stage}' para {video_stem}:")
        traceback.print_exc()
        _jobs[job_key] = {"status": "error", "error": f"[{stage}] {exc}"}


@app.route("/api/generate_previews/<video_stem>", methods=["POST"])
def generate_previews(video_stem: str):
    body = request.get_json(silent=True) or {}
    category = body.get("category", "general")
    custom_instruction = (body.get("instruction") or "").strip() or None
    if not _candidates_path(video_stem, category, custom_instruction).exists():
        return jsonify({"error": "todavía no se analizó este video"}), 404
    variant = _variant(category, custom_instruction)
    job_key = _job_key(video_stem, variant)
    current = _jobs.get(job_key, {}).get("status")
    if current not in (None, "listo", "error"):
        return jsonify({"status": current}), 409
    threading.Thread(target=_run_previews_only, args=(video_stem, category, custom_instruction), daemon=True).start()
    return jsonify({"status": "iniciado"})


@app.route("/api/analyze/<video_stem>", methods=["POST"])
def start_analysis(video_stem: str):
    body = request.get_json(silent=True) or {}
    use_visual = bool(body.get("use_visual", False))
    force = bool(body.get("force", False))
    category = body.get("category", "general")
    custom_instruction = (body.get("instruction") or "").strip() or None
    target_clips = body.get("target_clips")
    target_clips = int(target_clips) if target_clips else None
    speed = body.get("speed") or "completo"
    if speed not in ("rapido", "medio", "completo"):
        speed = "completo"

    variant = _variant(category, custom_instruction)
    job_key = _job_key(video_stem, variant)
    current = _jobs.get(job_key, {}).get("status")
    if current not in (None, "listo", "error"):
        return jsonify({"status": current}), 409

    threading.Thread(
        target=_run_analysis,
        args=(video_stem, use_visual, category, custom_instruction, force, target_clips, speed),
        daemon=True,
    ).start()
    return jsonify({"status": "iniciado"})


@app.route("/api/status/<video_stem>")
def status(video_stem: str):
    category = request.args.get("category", "general")
    custom_instruction = (request.args.get("instruction") or "").strip() or None
    variant = _variant(category, custom_instruction)
    job_key = _job_key(video_stem, variant)
    return jsonify(_jobs.get(job_key, {"status": "sin iniciar", "error": None}))


@app.route("/api/cost/<video_stem>")
def get_cost(video_stem: str):
    return jsonify(cost_summary(video_stem))


@app.route("/api/candidates/<video_stem>")
def get_candidates(video_stem: str):
    category = request.args.get("category", "general")
    custom_instruction = (request.args.get("instruction") or "").strip() or None
    variant = _variant(category, custom_instruction)
    path = _candidates_path(video_stem, category, custom_instruction)
    if not path.exists():
        return jsonify({"error": "todavía no se analizó este video"}), 404
    clips = json.loads(path.read_text(encoding="utf-8"))
    for i, c in enumerate(clips):
        c["index"] = i
        c["duration"] = round(sum(e - s for s, e in c["parts"]), 1)
        c["has_original"] = bool(c.get("original_parts"))
        out_path = _final_path(video_stem, variant, i, c["title"])
        c["rendered_url"] = (
            f"/clips/{video_stem}/{variant}/{out_path.name}?t={int(out_path.stat().st_mtime)}"
            if out_path.exists() else None
        )
        preview_path = _preview_path(video_stem, variant, i, c["title"])
        c["preview_url"] = (
            f"/previews/{video_stem}/{variant}/{preview_path.name}?t={int(preview_path.stat().st_mtime)}"
            if preview_path.exists() else None
        )
        strip_path = _filmstrip_path(video_stem, variant, i, c["title"])
        c["filmstrip_url"] = (
            f"/previews/{video_stem}/{variant}/{strip_path.name}?t={int(strip_path.stat().st_mtime)}"
            if strip_path.exists() else None
        )
    return jsonify(clips)


@app.route("/previews/<video_stem>/<variant>/<path:filename>")
def serve_preview(video_stem: str, variant: str, filename: str):
    return send_from_directory(_preview_dir(video_stem, variant), filename)


def _apply_new_parts(video_stem: str, variant: str, index: int, clip: dict[str, Any], new_parts: list) -> dict[str, Any]:
    """Recalcula preview + tira de miniaturas para un candidato con partes
    nuevas, invalida la versión final vieja (quedaría desactualizada) y
    devuelve la respuesta lista para la API. Compartido por "ajustar" y
    "restaurar" — son la misma operación, solo cambian las partes de origen.
    """
    transcript = json.loads((TRANSCRIPTS_DIR / f"{video_stem}.json").read_text(encoding="utf-8"))
    video_path = INPUT_DIR / f"{video_stem}.mp4"
    preview_dir = _preview_dir(video_stem, variant)
    preview_dir.mkdir(parents=True, exist_ok=True)
    out_path = _preview_path(video_stem, variant, index, clip["title"])
    ass_path = preview_dir / f"{out_path.stem}.ass"
    parts_t = [tuple(p) for p in new_parts]

    strip_path = _filmstrip_path(video_stem, variant, index, clip["title"])
    build_clip_subtitles(transcript["segments"], parts_t, ass_path, highlight_color=_subtitle_color(clip))
    if out_path.exists():
        out_path.unlink()
    render_preview(video_path, parts_t, ass_path, out_path)
    if strip_path.exists():
        strip_path.unlink()
    render_filmstrip(out_path, strip_path, clip["duration"])

    final_path = _final_path(video_stem, variant, index, clip["title"])
    if final_path.exists():
        final_path.unlink()

    t = int(out_path.stat().st_mtime)
    return {
        "preview_url": f"/previews/{video_stem}/{variant}/{out_path.name}?t={t}",
        "filmstrip_url": f"/previews/{video_stem}/{variant}/{strip_path.name}?t={t}",
        "duration": clip["duration"],
    }


@app.route("/api/adjust/<video_stem>/<int:index>", methods=["POST"])
def adjust_candidate(video_stem: str, index: int):
    """Corrige a mano el arranque/final de un candidato (unos segundos de
    más o de menos) sin tener que re-analizar nada — el ajuste fino que a
    veces hace falta cuando el corte automático se pasa por poco."""
    body = request.get_json(silent=True) or {}
    delta_start = float(body.get("delta_start", 0))
    delta_end = float(body.get("delta_end", 0))
    category = body.get("category", "general")
    custom_instruction = (body.get("instruction") or "").strip() or None
    variant = _variant(category, custom_instruction)

    candidates_path = _candidates_path(video_stem, category, custom_instruction)
    if not candidates_path.exists() or not (TRANSCRIPTS_DIR / f"{video_stem}.json").exists():
        return jsonify({"error": "no hay análisis para este video"}), 404

    clips = json.loads(candidates_path.read_text(encoding="utf-8"))
    if not (0 <= index < len(clips)):
        return jsonify({"error": "índice inválido"}), 400
    clip = clips[index]

    # Guardamos el corte original de la IA la primera vez que se toca a
    # mano, para poder volver atrás después con "Restaurar".
    if "original_parts" not in clip:
        clip["original_parts"] = clip["parts"]

    parts = [list(p) for p in clip["parts"]]
    parts[0][0] = max(0.0, parts[0][0] + delta_start)
    parts[-1][1] = max(parts[-1][0] + 1.0, parts[-1][1] + delta_end)
    clip["parts"] = parts
    clip["duration"] = round(sum(e - s for s, e in parts), 1)
    candidates_path.write_text(json.dumps(clips, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        result = _apply_new_parts(video_stem, variant, index, clip, parts)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

    result["has_original"] = True
    return jsonify(result)


_EDIT_STYLE_MIN_GAP = {
    "sin_cortes": None,   # None = no detectar nada, el cliente ni siquiera deberia pedir esto
    # "balanceado" subido de 0.8 a 1.5 el 2026-08-20: medido sobre 1546 clips
    # reales ya publicados y exitosos del corpus, la pausa interna típica
    # dura 0.88s de mediana y hasta 1.68s en el 75% — con el umbral viejo el
    # auto-recorte marcaba como "pausa muerta recortable" el ritmo normal de
    # casi cualquier clip bueno, no silencio real. 1.5s deja pasar el ritmo
    # natural y solo marca pausas genuinamente largas.
    "balanceado": 1.5,
    "acelerado": 0.4,     # mas agresivo, agarra pausas mas cortas tambien — opt-in, no default
}


@app.route("/api/micro_cuts/<video_stem>/<int:index>")
def get_micro_cuts(video_stem: str, index: int):
    """Detecta TODAS las pausas/silencios internos recortables a este umbral,
    calculadas siempre sobre la versión SIN CORTAR (original_parts) — no
    sobre el estado actual. Así una pausa que ya está cortada sigue
    apareciendo en la lista (marcada como `cut: true`), y destildarla
    después sí la restaura, en vez de desaparecer para siempre en cuanto se
    corta una vez (que es lo que pasaba antes: una vez cortada, el gap ya no
    existe en el timeline actual y no había forma de volver a detectarlo)."""
    category = request.args.get("category", "general")
    custom_instruction = (request.args.get("instruction") or "").strip() or None
    edit_style = request.args.get("edit_style", "balanceado")
    min_gap = _EDIT_STYLE_MIN_GAP.get(edit_style, 0.8)
    if min_gap is None:
        return jsonify({"cuts": []})

    transcript_path = TRANSCRIPTS_DIR / f"{video_stem}.json"
    candidates_path = _candidates_path(video_stem, category, custom_instruction)
    if not transcript_path.exists() or not candidates_path.exists():
        return jsonify({"error": "no hay análisis para este video"}), 404

    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    clips = json.loads(candidates_path.read_text(encoding="utf-8"))
    if not (0 <= index < len(clips)):
        return jsonify({"error": "índice inválido"}), 400
    clip = clips[index]

    base_parts = [tuple(p) for p in clip.get("original_parts", clip["parts"])]
    current_parts = [tuple(p) for p in clip["parts"]]
    all_gaps = detect_micro_cuts(transcript["segments"], base_parts, min_gap=min_gap)

    def _is_currently_cut(gap: tuple[float, float]) -> bool:
        gs, ge = gap
        # si ningún tramo actual cubre este rango completo, es porque se cortó
        return not any(ps <= gs and ge <= pe for ps, pe in current_parts)

    result = [{"start": s, "end": e, "cut": _is_currently_cut((s, e))} for s, e in all_gaps]
    return jsonify({"cuts": result})


@app.route("/api/apply_micro_cuts/<video_stem>/<int:index>", methods=["POST"])
def apply_micro_cuts_route(video_stem: str, index: int):
    """Recibe la lista completa de pausas que el usuario quiere cortada
    (tildadas) y recalcula SIEMPRE desde la versión sin cortar — nunca
    incremental sobre el estado actual. Así tildar/destildar es reversible:
    destildar una pausa ya cortada la vuelve a poner, no hace falta ningún
    caso especial para "deshacer" un corte puntual."""
    body = request.get_json(silent=True) or {}
    selected_cuts = body.get("cuts") or []
    category = body.get("category", "general")
    custom_instruction = (body.get("instruction") or "").strip() or None
    variant = _variant(category, custom_instruction)

    candidates_path = _candidates_path(video_stem, category, custom_instruction)
    if not candidates_path.exists() or not (TRANSCRIPTS_DIR / f"{video_stem}.json").exists():
        return jsonify({"error": "no hay análisis para este video"}), 404

    clips = json.loads(candidates_path.read_text(encoding="utf-8"))
    if not (0 <= index < len(clips)):
        return jsonify({"error": "índice inválido"}), 400
    clip = clips[index]

    base_parts = [tuple(p) for p in clip.get("original_parts", clip["parts"])]
    cuts = [(float(c[0]), float(c[1])) for c in selected_cuts]
    new_parts = apply_micro_cuts(base_parts, cuts)
    if not new_parts:
        return jsonify({"error": "esos cortes dejarían el clip vacío"}), 400

    same_as_base = new_parts == base_parts
    if same_as_base:
        clip["parts"] = [list(p) for p in base_parts]
        clip.pop("original_parts", None)
    else:
        clip["original_parts"] = [list(p) for p in base_parts]
        clip["parts"] = [list(p) for p in new_parts]
    clip["duration"] = round(sum(e - s for s, e in new_parts), 1)
    candidates_path.write_text(json.dumps(clips, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        result = _apply_new_parts(video_stem, variant, index, clip, new_parts)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

    result["has_original"] = not same_as_base
    result["parts"] = clip["parts"]
    return jsonify(result)


@app.route("/api/set_edit_style/<video_stem>/<int:index>", methods=["POST"])
def set_edit_style(video_stem: str, index: int):
    """Re-edita el candidato al nivel elegido (sin_cortes/balanceado/acelerado)
    de una — no hace falta pasar por la lista de sugerencias para que el
    cambio de nivel se note. Siempre parte del corte ORIGINAL de la IA
    (nunca acumula cortes sobre cortes), así cambiar de nivel da siempre el
    mismo resultado sin importar qué nivel estaba aplicado antes."""
    body = request.get_json(silent=True) or {}
    edit_style = body.get("edit_style", "balanceado")
    category = body.get("category", "general")
    custom_instruction = (body.get("instruction") or "").strip() or None
    variant = _variant(category, custom_instruction)
    min_gap = _EDIT_STYLE_MIN_GAP.get(edit_style, 0.8)

    transcript_path = TRANSCRIPTS_DIR / f"{video_stem}.json"
    candidates_path = _candidates_path(video_stem, category, custom_instruction)
    if not transcript_path.exists() or not candidates_path.exists():
        return jsonify({"error": "no hay análisis para este video"}), 404

    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    clips = json.loads(candidates_path.read_text(encoding="utf-8"))
    if not (0 <= index < len(clips)):
        return jsonify({"error": "índice inválido"}), 400
    clip = clips[index]

    base_parts = [tuple(p) for p in clip.get("original_parts", clip["parts"])]

    # Cada nivel de ritmo da siempre el mismo resultado para este clip (ver
    # docstring) — si ya lo calculamos antes, reusamos el preview ya
    # renderizado en vez de volver a pasar por ffmpeg. Piso rápido: un
    # cambio de nivel que ya se vio antes es casi instantáneo.
    style_cache = clip.setdefault("_style_cache", {})
    cached = style_cache.get(edit_style)
    cache_preview_path, cache_strip_path = _style_cache_paths(video_stem, variant, index, clip["title"], edit_style)
    if cached and cache_preview_path.exists() and cache_strip_path.exists():
        clip["parts"] = cached["parts"]
        if cached.get("original_parts"):
            clip["original_parts"] = cached["original_parts"]
        else:
            clip.pop("original_parts", None)
        if cached.get("auto_cut_info"):
            clip["auto_cut_info"] = cached["auto_cut_info"]
        else:
            clip.pop("auto_cut_info", None)
        clip["duration"] = cached["duration"]
        clip["edit_style"] = edit_style
        candidates_path.write_text(json.dumps(clips, ensure_ascii=False, indent=2), encoding="utf-8")

        preview_dir = _preview_dir(video_stem, variant)
        preview_dir.mkdir(parents=True, exist_ok=True)
        out_path = _preview_path(video_stem, variant, index, clip["title"])
        strip_path = _filmstrip_path(video_stem, variant, index, clip["title"])
        shutil.copyfile(cache_preview_path, out_path)
        shutil.copyfile(cache_strip_path, strip_path)
        final_path = _final_path(video_stem, variant, index, clip["title"])
        if final_path.exists():
            final_path.unlink()
        t = int(out_path.stat().st_mtime)
        return jsonify({
            "preview_url": f"/previews/{video_stem}/{variant}/{out_path.name}?t={t}",
            "filmstrip_url": f"/previews/{video_stem}/{variant}/{strip_path.name}?t={t}",
            "duration": clip["duration"],
            "has_original": bool(cached.get("original_parts")),
            "auto_cut_info": clip.get("auto_cut_info"),
            "edit_style": edit_style,
        })

    if min_gap is None:
        new_parts = base_parts
        clip.pop("auto_cut_info", None)
    else:
        cuts = detect_micro_cuts(transcript["segments"], base_parts, min_gap=min_gap)
        new_parts = apply_micro_cuts(base_parts, cuts) if cuts else base_parts
        if not new_parts:
            new_parts = base_parts
        if cuts:
            saved = sum(e - s for s, e in base_parts) - sum(e - s for s, e in new_parts)
            clip["auto_cut_info"] = {"count": len(cuts), "saved_seconds": round(saved, 1)}
        else:
            clip.pop("auto_cut_info", None)

    same_as_base = new_parts == base_parts
    if same_as_base:
        clip["parts"] = [list(p) for p in base_parts]
        clip.pop("original_parts", None)
    else:
        clip["original_parts"] = [list(p) for p in base_parts]
        clip["parts"] = [list(p) for p in new_parts]
    clip["duration"] = round(sum(e - s for s, e in new_parts), 1)
    clip["edit_style"] = edit_style

    try:
        result = _apply_new_parts(video_stem, variant, index, clip, new_parts)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

    # Guardamos una copia aparte para la próxima vez que se elija este nivel.
    out_path = _preview_path(video_stem, variant, index, clip["title"])
    strip_path = _filmstrip_path(video_stem, variant, index, clip["title"])
    cache_preview_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out_path, cache_preview_path)
    shutil.copyfile(strip_path, cache_strip_path)
    style_cache[edit_style] = {
        "parts": clip["parts"],
        "original_parts": clip.get("original_parts"),
        "auto_cut_info": clip.get("auto_cut_info"),
        "duration": clip["duration"],
    }
    candidates_path.write_text(json.dumps(clips, ensure_ascii=False, indent=2), encoding="utf-8")

    result["has_original"] = not same_as_base
    result["auto_cut_info"] = clip.get("auto_cut_info")
    result["edit_style"] = edit_style
    return jsonify(result)


@app.route("/api/set_subtitle_color/<video_stem>/<int:index>", methods=["POST"])
def set_subtitle_color(video_stem: str, index: int):
    """Cambia solo el color de resaltado del subtítulo — no toca los cortes
    ni las partes del clip para nada, solo vuelve a quemar el .ass con el
    color elegido. Mucho más liviano que set_edit_style: no hay que
    recalcular pausas, solo re-renderizar con el mismo recorte de siempre."""
    body = request.get_json(silent=True) or {}
    color = body.get("color", "amarillo")
    if color not in SUBTITLE_COLOR_PRESETS:
        return jsonify({"error": f"color inválido, opciones: {list(SUBTITLE_COLOR_PRESETS)}"}), 400
    category = body.get("category", "general")
    custom_instruction = (body.get("instruction") or "").strip() or None
    variant = _variant(category, custom_instruction)

    candidates_path = _candidates_path(video_stem, category, custom_instruction)
    if not candidates_path.exists() or not (TRANSCRIPTS_DIR / f"{video_stem}.json").exists():
        return jsonify({"error": "no hay análisis para este video"}), 404

    clips = json.loads(candidates_path.read_text(encoding="utf-8"))
    if not (0 <= index < len(clips)):
        return jsonify({"error": "índice inválido"}), 400
    clip = clips[index]
    clip["subtitle_color"] = color
    candidates_path.write_text(json.dumps(clips, ensure_ascii=False, indent=2), encoding="utf-8")

    parts = [tuple(p) for p in clip["parts"]]
    try:
        result = _apply_new_parts(video_stem, variant, index, clip, parts)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

    result["subtitle_color"] = color
    return jsonify(result)


@app.route("/api/restore/<video_stem>/<int:index>", methods=["POST"])
def restore_candidate(video_stem: str, index: int):
    """Vuelve el candidato al corte original que eligió la IA, deshaciendo
    cualquier ajuste manual hecho después."""
    body = request.get_json(silent=True) or {}
    category = body.get("category", "general")
    custom_instruction = (body.get("instruction") or "").strip() or None
    variant = _variant(category, custom_instruction)

    candidates_path = _candidates_path(video_stem, category, custom_instruction)
    if not candidates_path.exists() or not (TRANSCRIPTS_DIR / f"{video_stem}.json").exists():
        return jsonify({"error": "no hay análisis para este video"}), 404

    clips = json.loads(candidates_path.read_text(encoding="utf-8"))
    if not (0 <= index < len(clips)):
        return jsonify({"error": "índice inválido"}), 400
    clip = clips[index]

    original = clip.get("original_parts")
    if not original:
        return jsonify({"error": "este clip no tiene ningún ajuste manual para deshacer"}), 400

    clip["parts"] = original
    clip["duration"] = round(sum(e - s for s, e in original), 1)
    del clip["original_parts"]
    clip.pop("auto_cut_info", None)  # ya no aplica — el clip volvió a la version sin cortar
    clip["edit_style"] = "sin_cortes"
    candidates_path.write_text(json.dumps(clips, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        result = _apply_new_parts(video_stem, variant, index, clip, original)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

    result["has_original"] = False
    result["edit_style"] = "sin_cortes"
    return jsonify(result)


@app.route("/api/render/<video_stem>/<int:index>", methods=["POST"])
def render_candidate(video_stem: str, index: int):
    body = request.get_json(silent=True) or {}
    category = body.get("category", "general")
    custom_instruction = (body.get("instruction") or "").strip() or None
    variant = _variant(category, custom_instruction)

    transcript_path = TRANSCRIPTS_DIR / f"{video_stem}.json"
    candidates_path = _candidates_path(video_stem, category, custom_instruction)
    if not transcript_path.exists() or not candidates_path.exists():
        return jsonify({"error": "no hay análisis para este video"}), 404

    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    clips = json.loads(candidates_path.read_text(encoding="utf-8"))
    if not (0 <= index < len(clips)):
        return jsonify({"error": "índice inválido"}), 400
    clip = clips[index]

    video_path = INPUT_DIR / f"{video_stem}.mp4"
    out_dir = _final_dir(video_stem, variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = _preview_dir(video_stem, variant)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    out_path = _final_path(video_stem, variant, index, clip["title"])
    ass_path = tmp_dir / f"{out_path.stem}_final.ass"
    parts = [tuple(p) for p in clip["parts"]]

    try:
        build_clip_subtitles(transcript["segments"], parts, ass_path, highlight_color=_subtitle_color(clip))
        render_clip(video_path, parts, ass_path, out_path)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

    return jsonify({"url": f"/clips/{video_stem}/{variant}/{out_path.name}"})


@app.route("/clips/<video_stem>/<variant>/<path:filename>")
def serve_clip(video_stem: str, variant: str, filename: str):
    return send_from_directory(_final_dir(video_stem, variant), filename)


@app.route("/api/download_all/<video_stem>")
def download_all(video_stem: str):
    """Todos los clips ya finalizados de este video+búsqueda, en un solo zip —
    para no tener que descargarlos de a uno cuando ya elegiste varios."""
    category = request.args.get("category", "general")
    custom_instruction = (request.args.get("instruction") or "").strip() or None
    variant = _variant(category, custom_instruction)

    final_dir = _final_dir(video_stem, variant)
    files = sorted(final_dir.glob("*.mp4")) if final_dir.exists() else []
    if not files:
        return jsonify({"error": "todavía no finalizaste ningún clip"}), 404

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f.name)
    buf.seek(0)
    return send_file(
        buf, mimetype="application/zip", as_attachment=True,
        download_name=f"{video_stem}_{variant}_clips.zip",
    )


if __name__ == "__main__":
    try:
        from cleanup_temp import cleanup_stale_tmp

        summary = cleanup_stale_tmp()
        if summary["removed"]:
            print(f"[cleanup] liberados {summary['freed_mb']}MB de temporales viejos ({len(summary['removed'])} items)")
    except Exception as exc:  # noqa: BLE001
        print(f"[cleanup] no se pudo limpiar temporales al arrancar: {exc}")

    app.run(debug=False, host="127.0.0.1", port=8765)
