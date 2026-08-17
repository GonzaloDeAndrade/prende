"""Orquesta el pipeline completo: transcribir -> analizar -> cortar clips."""
from __future__ import annotations

import re
from pathlib import Path

from .analyze import analyze
from .config import CLIPS_DIR, TMP_DIR
from .cutter import render_clip
from .subtitles import build_clip_subtitles
from .transcribe import transcribe


def _slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:60] or "clip"


def run_pipeline(
    video_path: str | Path, force: bool = False, use_visual: bool = False, category: str = "general",
) -> list[Path]:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    print(f"\n=== 1/3 Transcribiendo: {video_path.name} ===")
    transcript = transcribe(video_path, force=force)

    print(f"\n=== 2/3 Analizando transcripción con LLM ===")
    clips = analyze(
        transcript, video_path.stem, video_path=video_path,
        use_visual=use_visual, category=category, force=force,
    )

    print(f"\n=== 3/3 Cortando {len(clips)} clips ===")
    out_dir = CLIPS_DIR / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_subs_dir = TMP_DIR / video_path.stem
    tmp_subs_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for i, clip in enumerate(clips, start=1):
        slug = _slugify(clip["title"])
        out_path = out_dir / f"{i:02d}_{slug}.mp4"
        ass_path = tmp_subs_dir / f"{i:02d}_{slug}.ass"
        parts = [tuple(p) for p in clip["parts"]]

        rango = " + ".join(f"{s:.1f}s-{e:.1f}s" for s, e in parts)
        print(f"  -> clip {i}/{len(clips)}: {clip['title']} [{rango}]")

        build_clip_subtitles(transcript["segments"], parts, ass_path)
        render_clip(video_path, parts, ass_path, out_path)
        outputs.append(out_path)

    print(f"\nListo. {len(outputs)} clips en {out_dir}")
    return outputs
