"""Limpieza de archivos temporales viejos en data/tmp/ (previews, subtítulos
.ass, filmstrips) para no llenar el disco con corridas de prueba pasadas.

Todo lo que borra es reproducible: si se necesita de nuevo, `_generate_previews`
en server.py lo vuelve a generar la próxima vez que se pidan previews de ese
video (chequea `out_path.exists()` antes de regenerar, así que no hay drama).

A propósito NUNCA toca:
- data/tmp/facecam_experiment/ (experimento aparte, en curso — no meterse)
- data/tmp/_uploads/ (se limpia con su propio umbral, más corto, más abajo)
- data/clips/, data/input/, data/research/ (no son temporales)

Uso: python cleanup_temp.py [--max-age-days N] [--dry-run]
También se corre automáticamente al arrancar server.py (umbral conservador).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
TMP_DIR = ROOT / "data" / "tmp"
UPLOADS_DIR = TMP_DIR / "_uploads"

_EXCLUDED_NAMES = {"facecam_experiment", "_uploads"}

DEFAULT_MAX_AGE_DAYS = 14
DEFAULT_UPLOADS_MAX_AGE_HOURS = 48


def _dir_last_activity(path: Path) -> float:
    """mtime más reciente entre el directorio y todo lo que tiene adentro —
    así una carpeta vieja que se volvió a tocar hace poco no se borra."""
    newest = path.stat().st_mtime
    for p in path.rglob("*"):
        try:
            newest = max(newest, p.stat().st_mtime)
        except OSError:
            continue
    return newest


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                continue
    return total


def cleanup_stale_tmp(
    max_age_days: float = DEFAULT_MAX_AGE_DAYS,
    uploads_max_age_hours: float = DEFAULT_UPLOADS_MAX_AGE_HOURS,
    dry_run: bool = False,
) -> dict:
    """Borra subcarpetas de data/tmp/ (excepto las excluidas) sin actividad
    reciente, y archivos .part de subidas cortadas y abandonadas en
    data/tmp/_uploads/. Devuelve un resumen para loguear."""
    now = time.time()
    removed: list[str] = []
    freed_bytes = 0

    if TMP_DIR.exists():
        for entry in TMP_DIR.iterdir():
            if not entry.is_dir() or entry.name in _EXCLUDED_NAMES:
                continue
            age_days = (now - _dir_last_activity(entry)) / 86400
            if age_days < max_age_days:
                continue
            size = _dir_size(entry)
            removed.append(f"{entry.relative_to(TMP_DIR)} ({size / 1e6:.1f}MB, {age_days:.0f}d sin uso)")
            freed_bytes += size
            if not dry_run:
                shutil.rmtree(entry, ignore_errors=True)

    if UPLOADS_DIR.exists():
        for part in UPLOADS_DIR.glob("*.part"):
            age_hours = (now - part.stat().st_mtime) / 3600
            if age_hours < uploads_max_age_hours:
                continue
            size = part.stat().st_size
            removed.append(f"_uploads/{part.name} ({size / 1e6:.1f}MB, {age_hours:.0f}h abandonado)")
            freed_bytes += size
            if not dry_run:
                part.unlink(missing_ok=True)

    return {"removed": removed, "freed_mb": round(freed_bytes / 1e6, 1)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-age-days", type=float, default=DEFAULT_MAX_AGE_DAYS)
    parser.add_argument("--uploads-max-age-hours", type=float, default=DEFAULT_UPLOADS_MAX_AGE_HOURS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summary = cleanup_stale_tmp(args.max_age_days, args.uploads_max_age_hours, args.dry_run)
    if summary["removed"]:
        prefix = "[dry-run] se borraría" if args.dry_run else "borrado"
        print(f"{prefix}:")
        for line in summary["removed"]:
            print(f"  - {line}")
        print(f"Total: {summary['freed_mb']}MB")
    else:
        print("Nada para limpiar.")
