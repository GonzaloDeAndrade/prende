"""Servidor local para revisar candidatos de clips y elegir cuáles cortar.

Reemplaza el flujo 100% automático por uno asistido: se analiza el video,
se muestran TODOS los candidatos evaluados (no solo los que pasaron el
filtro) con su score/razón, y un humano elige cuáles convertir en el clip
vertical final con subtítulos.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory

from clip_engine.analyze import analyze
from clip_engine.config import CLIPS_DIR, INPUT_DIR, TMP_DIR, TRANSCRIPTS_DIR
from clip_engine.cutter import render_clip, render_filmstrip, render_preview
from clip_engine.subtitles import build_clip_subtitles
from clip_engine.transcribe import transcribe

app = Flask(__name__)

_jobs: dict[str, dict[str, Any]] = {}
_downloads: dict[str, dict[str, Any]] = {}


def _slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:60] or "clip"


def _cat_tag(category: str) -> str:
    return "" if category == "general" else f".{category}"


def _candidates_path(video_stem: str, category: str) -> Path:
    return TRANSCRIPTS_DIR / f"{video_stem}{_cat_tag(category)}.candidates.json"


def _preview_dir(video_stem: str, category: str) -> Path:
    return TMP_DIR / video_stem / "previews" / (category or "general")


def _preview_path(video_stem: str, category: str, index: int, title: str) -> Path:
    slug = _slugify(title)
    return _preview_dir(video_stem, category) / f"{index:02d}_{slug}.mp4"


def _filmstrip_path(video_stem: str, category: str, index: int, title: str) -> Path:
    slug = _slugify(title)
    return _preview_dir(video_stem, category) / f"{index:02d}_{slug}_strip.jpg"


def _final_dir(video_stem: str, category: str) -> Path:
    return CLIPS_DIR / video_stem / (category or "general")


def _final_path(video_stem: str, category: str, index: int, title: str) -> Path:
    slug = _slugify(title)
    return _final_dir(video_stem, category) / f"{index:02d}_{slug}.mp4"


def _job_key(video_stem: str, category: str) -> str:
    return f"{video_stem}{_cat_tag(category)}"


@app.route("/")
def index():
    return send_from_directory(Path(__file__).parent / "webui", "index.html")


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
    return jsonify({"stem": stem, "name": dest.name})


def _run_youtube_download(job_id: str, url: str) -> None:
    try:
        _downloads[job_id] = {"status": "descargando (puede tardar varios minutos)", "error": None, "stem": None}
        cmd = [
            sys.executable, "-m", "yt_dlp",
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
        stem = Path(lines[-1]).stem
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
    videos = []
    for f in sorted(INPUT_DIR.glob("*.mp4")):
        videos.append({
            "stem": f.stem,
            "name": f.name,
            "has_candidates": _candidates_path(f.stem, "general").exists(),
        })
    return jsonify(videos)


def _generate_previews(video_stem: str, category: str, transcript: dict, clips: list[dict[str, Any]]) -> None:
    job_key = _job_key(video_stem, category)
    preview_dir = _preview_dir(video_stem, category)
    preview_dir.mkdir(parents=True, exist_ok=True)
    video_path = INPUT_DIR / f"{video_stem}.mp4"
    for i, clip in enumerate(clips):
        _jobs[job_key] = {"status": f"generando vistas previas ({i + 1}/{len(clips)})", "error": None}
        out_path = _preview_path(video_stem, category, i, clip["title"])
        strip_path = _filmstrip_path(video_stem, category, i, clip["title"])
        if out_path.exists() and strip_path.exists():
            continue
        ass_path = preview_dir / f"{out_path.stem}.ass"
        parts = [tuple(p) for p in clip["parts"]]
        duration = sum(e - s for s, e in parts)
        build_clip_subtitles(transcript["segments"], parts, ass_path)
        render_preview(video_path, parts, ass_path, out_path)
        render_filmstrip(out_path, strip_path, duration)


def _run_analysis(video_stem: str, use_visual: bool, category: str, force: bool) -> None:
    job_key = _job_key(video_stem, category)
    try:
        _jobs[job_key] = {"status": "transcribiendo", "error": None}
        video_path = INPUT_DIR / f"{video_stem}.mp4"
        transcript = transcribe(video_path, force=force)
        _jobs[job_key] = {"status": "analizando (esto puede tardar)", "error": None}
        clips = analyze(
            transcript, video_stem, video_path=video_path,
            use_visual=use_visual, review=True, category=category, force=force,
        )
        _generate_previews(video_stem, category, transcript, clips)
        _jobs[job_key] = {"status": "listo", "error": None}
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        _jobs[job_key] = {"status": "error", "error": str(exc)}


def _run_previews_only(video_stem: str, category: str) -> None:
    job_key = _job_key(video_stem, category)
    try:
        transcript = json.loads((TRANSCRIPTS_DIR / f"{video_stem}.json").read_text(encoding="utf-8"))
        clips = json.loads(_candidates_path(video_stem, category).read_text(encoding="utf-8"))
        _generate_previews(video_stem, category, transcript, clips)
        _jobs[job_key] = {"status": "listo", "error": None}
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        _jobs[job_key] = {"status": "error", "error": str(exc)}


@app.route("/api/generate_previews/<video_stem>", methods=["POST"])
def generate_previews(video_stem: str):
    body = request.get_json(silent=True) or {}
    category = body.get("category", "general")
    if not _candidates_path(video_stem, category).exists():
        return jsonify({"error": "todavía no se analizó este video"}), 404
    job_key = _job_key(video_stem, category)
    current = _jobs.get(job_key, {}).get("status")
    if current not in (None, "listo", "error"):
        return jsonify({"status": current}), 409
    threading.Thread(target=_run_previews_only, args=(video_stem, category), daemon=True).start()
    return jsonify({"status": "iniciado"})


@app.route("/api/analyze/<video_stem>", methods=["POST"])
def start_analysis(video_stem: str):
    body = request.get_json(silent=True) or {}
    use_visual = bool(body.get("use_visual", False))
    force = bool(body.get("force", False))
    category = body.get("category", "general")

    job_key = _job_key(video_stem, category)
    current = _jobs.get(job_key, {}).get("status")
    if current not in (None, "listo", "error"):
        return jsonify({"status": current}), 409

    threading.Thread(target=_run_analysis, args=(video_stem, use_visual, category, force), daemon=True).start()
    return jsonify({"status": "iniciado"})


@app.route("/api/status/<video_stem>")
def status(video_stem: str):
    category = request.args.get("category", "general")
    job_key = _job_key(video_stem, category)
    return jsonify(_jobs.get(job_key, {"status": "sin iniciar", "error": None}))


@app.route("/api/candidates/<video_stem>")
def get_candidates(video_stem: str):
    category = request.args.get("category", "general")
    path = _candidates_path(video_stem, category)
    if not path.exists():
        return jsonify({"error": "todavía no se analizó este video"}), 404
    clips = json.loads(path.read_text(encoding="utf-8"))
    for i, c in enumerate(clips):
        c["index"] = i
        c["duration"] = round(sum(e - s for s, e in c["parts"]), 1)
        c["has_original"] = bool(c.get("original_parts"))
        out_path = _final_path(video_stem, category, i, c["title"])
        c["rendered_url"] = (
            f"/clips/{video_stem}/{category}/{out_path.name}?t={int(out_path.stat().st_mtime)}"
            if out_path.exists() else None
        )
        preview_path = _preview_path(video_stem, category, i, c["title"])
        c["preview_url"] = (
            f"/previews/{video_stem}/{category}/{preview_path.name}?t={int(preview_path.stat().st_mtime)}"
            if preview_path.exists() else None
        )
        strip_path = _filmstrip_path(video_stem, category, i, c["title"])
        c["filmstrip_url"] = (
            f"/previews/{video_stem}/{category}/{strip_path.name}?t={int(strip_path.stat().st_mtime)}"
            if strip_path.exists() else None
        )
    return jsonify(clips)


@app.route("/previews/<video_stem>/<category>/<path:filename>")
def serve_preview(video_stem: str, category: str, filename: str):
    return send_from_directory(_preview_dir(video_stem, category), filename)


def _apply_new_parts(video_stem: str, category: str, index: int, clip: dict[str, Any], new_parts: list) -> dict[str, Any]:
    """Recalcula preview + tira de miniaturas para un candidato con partes
    nuevas, invalida la versión final vieja (quedaría desactualizada) y
    devuelve la respuesta lista para la API. Compartido por "ajustar" y
    "restaurar" — son la misma operación, solo cambian las partes de origen.
    """
    transcript = json.loads((TRANSCRIPTS_DIR / f"{video_stem}.json").read_text(encoding="utf-8"))
    video_path = INPUT_DIR / f"{video_stem}.mp4"
    preview_dir = _preview_dir(video_stem, category)
    preview_dir.mkdir(parents=True, exist_ok=True)
    out_path = _preview_path(video_stem, category, index, clip["title"])
    ass_path = preview_dir / f"{out_path.stem}.ass"
    parts_t = [tuple(p) for p in new_parts]

    strip_path = _filmstrip_path(video_stem, category, index, clip["title"])
    build_clip_subtitles(transcript["segments"], parts_t, ass_path)
    if out_path.exists():
        out_path.unlink()
    render_preview(video_path, parts_t, ass_path, out_path)
    if strip_path.exists():
        strip_path.unlink()
    render_filmstrip(out_path, strip_path, clip["duration"])

    final_path = _final_path(video_stem, category, index, clip["title"])
    if final_path.exists():
        final_path.unlink()

    t = int(out_path.stat().st_mtime)
    return {
        "preview_url": f"/previews/{video_stem}/{category}/{out_path.name}?t={t}",
        "filmstrip_url": f"/previews/{video_stem}/{category}/{strip_path.name}?t={t}",
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

    candidates_path = _candidates_path(video_stem, category)
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
        result = _apply_new_parts(video_stem, category, index, clip, parts)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

    result["has_original"] = True
    return jsonify(result)


@app.route("/api/restore/<video_stem>/<int:index>", methods=["POST"])
def restore_candidate(video_stem: str, index: int):
    """Vuelve el candidato al corte original que eligió la IA, deshaciendo
    cualquier ajuste manual hecho después."""
    body = request.get_json(silent=True) or {}
    category = body.get("category", "general")

    candidates_path = _candidates_path(video_stem, category)
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
    candidates_path.write_text(json.dumps(clips, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        result = _apply_new_parts(video_stem, category, index, clip, original)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

    result["has_original"] = False
    return jsonify(result)


@app.route("/api/render/<video_stem>/<int:index>", methods=["POST"])
def render_candidate(video_stem: str, index: int):
    body = request.get_json(silent=True) or {}
    category = body.get("category", "general")

    transcript_path = TRANSCRIPTS_DIR / f"{video_stem}.json"
    candidates_path = _candidates_path(video_stem, category)
    if not transcript_path.exists() or not candidates_path.exists():
        return jsonify({"error": "no hay análisis para este video"}), 404

    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    clips = json.loads(candidates_path.read_text(encoding="utf-8"))
    if not (0 <= index < len(clips)):
        return jsonify({"error": "índice inválido"}), 400
    clip = clips[index]

    video_path = INPUT_DIR / f"{video_stem}.mp4"
    out_dir = _final_dir(video_stem, category)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = _preview_dir(video_stem, category)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    out_path = _final_path(video_stem, category, index, clip["title"])
    ass_path = tmp_dir / f"{out_path.stem}_final.ass"
    parts = [tuple(p) for p in clip["parts"]]

    try:
        build_clip_subtitles(transcript["segments"], parts, ass_path)
        render_clip(video_path, parts, ass_path, out_path)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

    return jsonify({"url": f"/clips/{video_stem}/{category}/{out_path.name}"})


@app.route("/clips/<video_stem>/<category>/<path:filename>")
def serve_clip(video_stem: str, category: str, filename: str):
    return send_from_directory(_final_dir(video_stem, category), filename)


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8765)
