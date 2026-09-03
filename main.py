"""CLI: python main.py <video> [--force]"""
from __future__ import annotations

import argparse

from clip_engine.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera clips virales verticales a partir de un video largo.")
    parser.add_argument("video", help="Ruta al video de entrada (mp4, mkv, etc.)")
    parser.add_argument("--force", action="store_true", help="Ignora caché de transcripción/análisis")
    parser.add_argument("--visual", action="store_true", help="Suma análisis visual (frames) a la selección")
    parser.add_argument(
        "--category", default="general",
        choices=["general", "streaming", "educativo", "comedia", "motivacion", "opinion", "reaccion"],
        help="Ajusta el criterio de selección al género del video",
    )
    parser.add_argument(
        "--instruction", default=None,
        help='Pedido libre en criollo, p. ej. "dame los momentos donde se ríe fuerte"',
    )
    args = parser.parse_args()

    run_pipeline(
        args.video, force=args.force, use_visual=args.visual,
        category=args.category, custom_instruction=args.instruction,
    )


if __name__ == "__main__":
    main()
