"""Reintento con backoff para llamadas a la API de OpenAI.

El límite de tokens/minuto es un techo fijo de la cuenta, no algo que se
esquive: un solo pico (un prompt grande, o varias llamadas seguidas en un
video largo) puede superarlo aunque el uso total sea razonable. Sin
reintento, ese pico tira abajo toda la corrida en vez de esperar un rato y
seguir — este helper es compartido por todos los puntos del pipeline que
llaman a la API (selección de clips, ranking, análisis visual) para que el
mismo comportamiento aplique en todos lados.
"""
from __future__ import annotations

import time
from typing import Callable, TypeVar

from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

T = TypeVar("T")

# Errores transitorios: vale la pena reintentar (rate limit, corte de red,
# timeout, o un 5xx del lado de OpenAI). Cualquier otro error (prompt
# inválido, auth, etc.) no se arregla reintentando, así que se deja propagar.
_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)


def call_with_backoff(fn: Callable[[], T], *, label: str, max_attempts: int = 12) -> T:
    for attempt in range(max_attempts):
        try:
            return fn()
        except _RETRYABLE as exc:
            wait = min(60, 5 * (attempt + 1))
            print(f"[{label}] {type(exc).__name__}, esperando {wait}s... (intento {attempt + 1}/{max_attempts})")
            time.sleep(wait)
    raise RuntimeError(f"Fallo persistente en {label} tras {max_attempts} intentos, probá de nuevo en un rato.")
