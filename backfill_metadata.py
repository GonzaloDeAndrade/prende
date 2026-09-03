"""Backfill de view_count/like_count/channel para los items ya procesados
del corpus de investigación — dato clave para la Parte 3 (que patrones
predicen éxito real, no solo "es viral" a ojo). Solo pide METADATA (nunca
descarga el video), así que no compite con el bloqueo de descargas de
YouTube que tiene el daemon principal.

Separado de research_batch.py a propósito — mismo criterio que todo lo
demás esta noche: no tocar lo que ya corre.

Uso: python backfill_metadata.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import truststore

truststore.inject_into_ssl()

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
MANIFEST_PATH = ROOT / "data" / "research" / "manifest.json"

PAUSE_SECONDS = 2.0


def _load() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _save(m: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


def _find(manifest: dict, bucket: str, item_id: str) -> dict | None:
    for it in manifest.get(bucket, []):
        if it["id"] == item_id:
            return it
    return None


def _fetch_metadata(url: str) -> dict | None:
    import yt_dlp

    ydl_opts = {"js_runtimes": {"node": {}}, "quiet": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                "view_count": info.get("view_count"),
                "like_count": info.get("like_count"),
                "channel": info.get("channel"),
            }
    except Exception as exc:  # noqa: BLE001
        print(f"  [error] {exc}")
        return None


def main() -> None:
    manifest = _load()
    pending = []
    for bucket in ("clips", "largos"):
        for it in manifest[bucket]:
            if it["status"] != "listo":
                continue
            if "view_count" in it:
                continue
            if "youtube.com" not in it.get("url", ""):
                continue
            pending.append((bucket, it["id"], it["url"]))

    print(f"{len(pending)} items para backfillear (de los ya listos)")
    for i, (bucket, item_id, url) in enumerate(pending):
        print(f"[{i + 1}/{len(pending)}] {item_id}...")
        meta = _fetch_metadata(url)
        if meta is not None:
            fresh = _load()
            fresh_item = _find(fresh, bucket, item_id)
            if fresh_item is not None:
                fresh_item.update(meta)
                _save(fresh)
                print(f"  views={meta['view_count']} canal={meta['channel']}")
        time.sleep(PAUSE_SECONDS)

    print("listo")


if __name__ == "__main__":
    main()
