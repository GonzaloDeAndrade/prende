"""Re-intenta el heatmap para items del corpus que quedaron con heatmap=None
por un bug real de research_batch.py::get_heatmap() (encontrado 2026-08-23):
esa función usaba {"node": {}} sin el path explícito del Node 22 portátil,
a diferencia de download_video() y de clip_engine/heatmap.py — sin el path
correcto, yt-dlp no resuelve el challenge de YouTube (el Node del sistema es
v20) y el heatmap se guardaba como None SIEMPRE, sin importar si el video
realmente tenía uno o no. No sabemos separar, sin re-consultar, cuáles de
los "sin heatmap" actuales son de verdad (pocas vistas) y cuáles son este
bug — así que se re-intentan todos los que tengan heatmap=None y URL de
YouTube.

Separado de research_batch.py a propósito: esto es un backfill puntual de
un bug ya arreglado, no parte del pipeline normal de un item nuevo.

Uso: python research_heatmap_backfill.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import truststore
truststore.inject_into_ssl()

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from yt_dlp_helper import NODE22_PATH

MANIFEST_PATH = ROOT / "data" / "research" / "manifest.json"
PAUSE_SECONDS = 6.0  # mas liviano que la descarga de video, pero conservador


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def save_manifest(m: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


def persist_heatmap(bucket: str, item_id: str, heatmap) -> None:
    fresh = load_manifest()
    for it in fresh.get(bucket, []):
        if it["id"] == item_id:
            it["heatmap"] = heatmap
            break
    save_manifest(fresh)


def fetch_heatmap(url: str) -> list[dict] | None:
    import yt_dlp

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
    except Exception as exc:
        print(f"    error: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    manifest = load_manifest()
    targets = []
    for bucket in ("clips", "largos"):
        for item in manifest.get(bucket, []):
            if item.get("status") in ("pending", "error"):
                continue
            if "youtube.com" not in (item.get("url") or ""):
                continue
            if item.get("heatmap") is not None:
                continue
            targets.append((bucket, item["id"], item["url"]))

    if args.limit:
        targets = targets[: args.limit]

    print(f"Re-intentando heatmap para {len(targets)} items (heatmap=None + YouTube)...")
    recovered = 0
    still_none = 0
    for i, (bucket, item_id, url) in enumerate(targets):
        hm = fetch_heatmap(url)
        persist_heatmap(bucket, item_id, hm)
        if hm:
            recovered += 1
            print(f"  [{i + 1}/{len(targets)}] {item_id}: RECUPERADO, {len(hm)} puntos")
        else:
            still_none += 1
            if (i + 1) % 50 == 0:
                print(f"  [{i + 1}/{len(targets)}] ... (recuperados hasta ahora: {recovered})")
        time.sleep(PAUSE_SECONDS)

    print(f"\nTerminado. Recuperados: {recovered}, siguen sin heatmap real: {still_none}")


if __name__ == "__main__":
    main()
