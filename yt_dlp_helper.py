"""Args compartidos para invocar yt-dlp contra YouTube, con el fix real del
bloqueo de descargas que tuvimos toda la noche.

Causa real (no era switch de IP como pensábamos): el Node del sistema es
v20, pero el solver de challenges JS de yt-dlp pide v22+ — con Node viejo,
yt-dlp lo marca "unsupported" y no puede resolver el challenge. Además el
cliente por defecto (web_safari) exige un PO Token que sin un provider
configurado nunca se genera, así que la descarga arranca pero corta con 403
a los pocos segundos. Se resuelve con dos cosas juntas:

1. Un Node 22 portátil (bajado aparte en .tools/, sin tocar el Node del
   sistema) para que el challenge JS se resuelva.
2. Forzar los clientes android+tv (`--extractor-args`), que sirven formatos
   progresivos sin el gate de PO Token/SABR que tiene web_safari.

Probado de punta a punta: descarga completa de un video de streaming de
~500MB / 105min sin un solo 403, con este fix. Sin él, cortaba siempre
antes del 1%.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent
NODE22_PATH = ROOT / ".tools" / "node-v22.23.2-win-x64" / "node.exe"

YT_DLP_FIX_ARGS = [
    "--js-runtimes", f"node:{NODE22_PATH}",
    "--extractor-args", "youtube:player_client=android,tv",
]
