"""Transcripción local con faster-whisper, con timestamps por palabra."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from .config import TRANSCRIPTS_DIR, settings


def _add_cuda_dll_dirs() -> None:
    """En Windows, ctranslate2 necesita encontrar cublas/cudnn instalados vía pip."""
    if sys.platform != "win32":
        return
    try:
        import nvidia  # type: ignore

        nvidia_dir = Path(nvidia.__path__[0])
        for sub in ("cudnn/bin", "cublas/bin", "cuda_nvrtc/bin"):
            bin_dir = nvidia_dir / sub
            if bin_dir.is_dir():
                os.add_dll_directory(str(bin_dir))
                # ctranslate2 carga estas DLLs con LoadLibraryA (búsqueda clásica),
                # que ignora add_dll_directory; hace falta también sumarlas al PATH.
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def _load_model():
    from faster_whisper import WhisperModel

    device = settings.whisper_device
    if device == "auto":
        _add_cuda_dll_dirs()
        try:
            model = WhisperModel(settings.whisper_model_size, device="cuda", compute_type="float16")
            print(f"[transcribe] usando GPU (cuda) con modelo {settings.whisper_model_size}")
            return model
        except Exception as exc:
            print(f"[transcribe] GPU no disponible ({exc}); usando CPU (más lento)")
            device = "cpu"

    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel(settings.whisper_model_size, device=device, compute_type=compute_type)


def transcribe(video_path: Path, force: bool = False) -> dict[str, Any]:
    """Transcribe un video/audio y devuelve segmentos + palabras con timestamps.

    Guarda el resultado como JSON en data/transcripts/<nombre>.json para no
    tener que re-transcribir en corridas siguientes.
    """
    video_path = Path(video_path)
    out_path = TRANSCRIPTS_DIR / f"{video_path.stem}.json"

    if out_path.exists() and not force:
        print(f"[transcribe] usando transcripción cacheada: {out_path}")
        return json.loads(out_path.read_text(encoding="utf-8"))

    model = _load_model()

    segments_iter, info = model.transcribe(
        str(video_path),
        language=settings.whisper_language,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    segments: list[dict[str, Any]] = []
    for seg in segments_iter:
        words = [
            {"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word}
            for w in (seg.words or [])
        ]
        segments.append(
            {
                "id": seg.id,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
                "words": words,
            }
        )
        print(f"  [{seg.start:7.1f}s -> {seg.end:7.1f}s] {seg.text.strip()}")

    result = {
        "source": str(video_path),
        "language": info.language,
        "duration": info.duration,
        "segments": segments,
    }

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[transcribe] guardado en {out_path}")
    return result
