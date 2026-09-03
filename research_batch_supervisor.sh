#!/usr/bin/env bash
# Supervisor del daemon de investigación: lo relanza automáticamente cada
# vez que termina, sea porque se retiró solo (cada _MAX_ITEMS_PER_RUN items,
# a propósito — ver el comentario en research_batch.py sobre el límite de
# recursos de Windows para procesos longevos), porque se quedó sin trabajo
# nuevo por mucho tiempo, o por cualquier otro motivo. Sin esto, un cierre
# silencioso puede pasar horas sin que nadie lo note (ya nos pasó).
cd "$(dirname "$0")"

while true; do
  echo ""
  echo "=== [supervisor] arrancando research_batch.py $(date) ==="
  ".venv/Scripts/python.exe" -u research_batch.py >> research_batch.log 2>&1
  code=$?
  echo "=== [supervisor] research_batch.py terminó (código $code), relanzando en 10s $(date) ===" >> research_batch.log
  sleep 10
done
