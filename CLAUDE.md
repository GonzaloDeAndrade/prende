# Prende — contexto del proyecto

SaaS de clipping automático para hispanohablantes de LatAm: subís un video largo
(podcast, entrevista, streaming) y la app encuentra los mejores momentos, los
corta en 9:16 con subtítulos quemados y zoom, listos para TikTok/Reels/Shorts.
Diferencial: calibrado para jerga y humor LatAm real, no traducido de criterios
en inglés.

Este archivo existe porque el proyecto lleva varios días de trabajo con
sesiones largas que se resumen automáticamente — cada resumen pierde matices.
Lo que importa de fondo va ACÁ, no solo en el historial de chat, para que
sobreviva a cualquier resumen o sesión nueva.

## Pipeline de producción (ya construido)

`clip_engine/analyze.py` hace la selección real, en dos pasadas de LLM:
1. **Generación de candidatos** (`SYSTEM_PROMPT` en `clip_engine/prompts.py`):
   el LLM propone momentos, con índices verificados contra el texto real
   (evita alucinaciones). Se complementa con dos mecanismos que FUERZAN
   candidatos que el LLM podría pasar por alto: picos de energía/risa en el
   audio, y momentos visuales notables (`_candidates_from_visual_moments`).
2. **Validación comparativa** (`RANKING_SYSTEM_PROMPT`): un segundo pase más
   duro que compara todos los candidatos entre sí y descarta los flojos —
   preferí menos clips pero más fuertes, no completar una cuota.

Hay 6 categorías (`CATEGORY_ADDENDUMS`) que ajustan el criterio según el
género del video (streaming, educativo, comedia, motivación, reacción,
opinión) — porque no hay una sola fórmula de "buen clip", depende del
contexto.

## El objetivo de fondo del análisis del corpus grande

Motivo original: un test real (`gaming_test`, streaming, 4 personas hablando
encima) dio clips malos. La reacción fue "esto no funciona, agregá análisis
visual" — de ahí salió la idea de analizar un corpus grande de videos reales
(no 10 o 20 — cientos/miles, porque con pocos ejemplos cualquier coincidencia
parece un patrón; con volumen se separa señal real de casualidad) para dejar
de depender solo de la intuición al escribir las reglas del prompt.

**Principio central**: no hay una fórmula única de "buen clip". Un remate
gracioso, uno con un dato sorprendente y uno emocional no se estructuran
igual, y lo que funciona en una entrevista no es lo mismo que en streaming.
Ya hay categorías para el género — pero dentro de cada categoría también
puede haber distintos tipos de remate/gancho que conviene distinguir, no
tratar como una sola regla.

**Por eso se analizan DOS tipos de material, con preguntas distintas**:

1. **Videos largos completos** (bucket `largos` en `data/research/manifest.json`)
   → encontrar, dentro de todo el contenido, cuáles son los mejores momentos
   candidatos a convertirse en clips. **Esto ya está construido y corriendo**
   (`research_batch.py`: transcripción + energía de audio + análisis visual
   por frame).

2. **Clips cortos ya publicados y exitosos** (bucket `clips` en el mismo
   manifest — contenido real ya cortado, de canales tipo "Clipeados") →
   aprender qué caracteriza a un clip YA TERMINADO que funciona bien como
   pieza independiente: duración, cómo arranca, cómo cierra, ritmo interno.
   Es una pregunta DISTINTA a "encontrar el momento dentro de un video
   largo" — acá se juzga el resultado final, no la selección.
   **Empezado el 2026-08-20** (antes no existía nada de esto): se midió
   duración real, patrón de arranque, patrón de cierre y ritmo/pausas sobre
   ~1380-1546 clips reales ya publicados (script ad-hoc, no productizado
   todavía — los números están abajo, en "Qué se hace con los patrones").
   El bucket `clips` en `research_batch.py` sigue analizándose con la MISMA
   lógica que `largos` (buscar momentos notables adentro, solo con muestreo
   visual más denso vía `CLIP_VISUAL_INTERVAL`) — eso en sí no cambió, lo
   que se agregó es un análisis estructural APARTE sobre los mismos archivos
   ya descargados. Falta: que quede como script reproducible en vez de
   comandos sueltos, y ampliar a más dimensiones (tipo de remate por
   hook_type, no solo agregado).

Estos dos análisis se relacionan (un buen clip sale de un buen momento
dentro de un video largo) pero responden preguntas complementarias, no son
lo mismo — no alcanza con solo el primero.

## Qué se hace con los patrones encontrados

No quedan como estadística aparte. Los patrones confirmados con evidencia
(no una sola vez, con volumen suficiente para descartar casualidad) se
traducen en cambios concretos al código o a los prompts — afinando el
criterio que ya usa el pipeline, no reemplazándolo. Ejemplos reales ya
hechos, todos el 2026-08-20:

- **Candidatos visuales rechazados de más**: 6 de 7 candidatos forzados por
  momentos visuales se rechazaban en la validación por "falta de diálogo" o
  "sin remate claro". Se corrigió `RANKING_SYSTEM_PROMPT` (clip_engine/prompts.py)
  para juzgarlos por especificidad de la descripción, no por presencia de
  texto ni por estructura de remate (un gesto no tiene "remate").
- **"Dato sorprendente" sin anclar**: un clip real (panchos_test, "tengo 900
  de colesterol") pasó el filtro con un dato que nunca explica por qué es
  alto/raro — ni comparación ni reacción de asombro. Se agregó a
  `SYSTEM_PROMPT` y `RANKING_SYSTEM_PROMPT` la exigencia de que el dato
  tiene que LLEGAR a sorprender, no solo estar nombrado.
- **`_fit_clip` "cumplía" el mínimo de duración tragándose silencio (encontrado
  probando con un video real de streaming de casino, 2026-08-21)**: al
  estirar un clip corto hasta `min_clip_seconds`, medía la duración como el
  lapso entre el primer y último segmento — si ese lapso incluía un hueco de
  silencio real (a veces ESCONDIDO adentro de un solo segmento de
  faster-whisper, entre dos palabras, no entre segmentos), lo contaba igual
  como "ya llegamos" y paraba de extender. El auto-recorte de pausas
  eliminaba ese hueco después, dejando el clip muy por debajo del mínimo real
  (caso medido: "10s" de ventana con un hueco de 5.3s adentro → 4.8s finales
  tras el recorte). Corregido: ahora mide duración de CONTENIDO real
  (excluyendo huecos a nivel palabra, igual que `detect_micro_cuts`), así
  sigue extendiendo hasta juntar material real. Verificado con el caso real:
  antes 4.8s, después del arreglo 10.0s exactos. Esto era probablemente la
  causa más grande de "clips muy cortos, casi no se entienden".
- **Umbral de auto-recorte de pausas mal calibrado (el más importante)**:
  medido sobre 1546 clips reales ya publicados y exitosos, la pausa interna
  típica dura 0.88s de mediana y hasta 1.68s en el 75% — el umbral viejo
  (`_MICRO_CUT_MIN_GAP` / `_AUTO_CUT_MIN_GAP` = 0.8s) marcaba como "pausa
  muerta recortable" el ritmo NORMAL de casi cualquier clip bueno, no
  silencio real. Subido a 1.5s en `clip_engine/analyze.py` y `server.py`
  (estilo "balanceado"). Esto explica por qué clips con buen contenido
  salían fragmentados/robóticos — no era la selección, era el auto-corte
  de después comiéndose el ritmo natural.
- **Duración máxima**: `max_clip_seconds` 30→35 (`.env`), porque el p75 real
  de clips exitosos es 32.9s — el tope viejo cortaba por abajo un cuarto de
  lo que el contenido real hace de forma natural.
- **Arranque sin aire muerto — validado, no cambiado**: la mediana del
  silencio antes de la primera palabra en clips reales es 0.00s. Confirma
  que la instrucción ya existente ("el arranque tiene que enganchar YA") ya
  estaba bien calibrada.

## Reglas operativas del corpus

- Sin límite de tiempo fijo — está bien que tarde días, la prioridad es
  volumen suficiente para que los patrones sean reales.
- El corpus general (`research_batch.py`, daemon con supervisor) corre solo,
  sin necesidad de que el usuario esté presente — bajar/analizar sin pedir
  permiso video por video.
- Excepción: el experimento paralelo de detección de facecam (archivos
  `facecam_*.py`) usa SOLO videos que el usuario pasa explícitamente, nunca
  del corpus general — son dos esfuerzos separados, no mezclar.

## Funciones nuevas en la app (2026-08-21), agregadas por pedido directo del usuario

- **3 velocidades de análisis** (reemplazó el toggle viejo "solo audio /
  audio+imagen"): `speed` = `rapido` (solo audio, ~3x más rápido, sin
  imagen), `medio` (imagen liviana, intervalo 8s, sin refinamiento fino),
  `completo` (como era antes, intervalo 3s + refinamiento). Parámetro en
  `analyze()` (`clip_engine/analyze.py`), selector `#speedSelect` en la web.
- **Cantidad de clips pedida** (`target_clips`, 5/10/20 en la interfaz):
  NO alcanza con pedírselo al LLM en el prompt — probado en vivo, pedir 5
  devolvió 17 candidatos con `keep=true` igual. Hace falta un tope real
  después del ranking: si sobran por encima de `target_clips`, los de menor
  score se pasan a `keep=false` (nunca se sacan de la lista — en modo
  revisión se siguen viendo y se pueden elegir a mano, solo dejan de venir
  pre-marcados como recomendados).
- **Nombre custom por video** y **duración real en `/api/videos`** (para
  estimar tiempo de análisis) — `data/video_names.json`.
- Ninguno de estos parámetros nuevos (`target_clips`, `speed`) forma parte
  de la clave de caché a propósito (mismo criterio que `use_visual`, que
  tampoco lo era) — cambiar el pedido reusa el caché salvo que se fuerce
  ("Re-analizar"). Meterlos en la clave hubiera significado tocar ~25 rutas
  de `server.py` que arman el path del caché.

## Estado al cierre de la sesión del 2026-08-20/21 (para retomar sin perder el hilo)

Bug real encontrado probando con un video real del usuario (`Pah7u3Ja-lY`,
streaming de casino/tragamonedas): `_fit_clip` "cumplía" el mínimo de
duración estirando la ventana por CANTIDAD DE SEGMENTOS, sin darse cuenta de
que parte de ese tiempo era silencio real que el auto-recorte iba a sacar
después — resultado: clips de "10s" que terminaban en 4-5s. Arreglado
(`effective_dur()` en `_fit_clip`, mide huecos a nivel palabra). Verificado
en el mismo video real: antes 7 de 8 clips por debajo de 10s, después del
arreglo (+ categoría correcta "streaming" en vez de "general") solo 2 de 8.

Quedó corriendo un análisis "completo" (audio+imagen a fondo) de
`Pah7u3Ja-lY` con categoría streaming — el usuario cargó $8.94 de crédito y
pidió explícitamente tomarse el tiempo, sin apurar. Terminó: 46 candidatos
evaluados, 6 recomendados (9-12s cada uno).

**Modelo de ranking separado, agregado el mismo día**: el paso de validación
final ahora usa `settings.openai_ranking_model` (default `gpt-4o`, ver
`clip_engine/config.py`) en vez de `openai_model` (mini) — medido real sobre
este mismo video: escanear frames con gpt-4o hubiera costado $11.51 (no
viable), pero el ranking (solo texto) cuesta $0.08 con gpt-4o vs $0.005 con
mini — diferencia despreciable por un juicio final más consistente
(probado: mini dejó 3 recomendados, gpt-4o dejó 6, con verdicts más
matizados, ej. reconociendo que un dato "podría beneficiarse de más
contexto" en vez de solo aprobar/rechazar en seco). El escaneo visual se
queda en `openai_model` (mini) a propósito.

**Duración corta en este video — investigado, es del género, no un bug
nuevo**: los 6 recomendados quedaron en 9-12s, lejos de la mediana real de
21.7s del corpus. Medido sobre los 46 candidatos COMPLETOS (no solo los
recomendados): mediana 9.7s, p75 10.9s — es decir, es el pool ENTERO el que
sale corto, no un sesgo del ranking hacia lo breve. Los 3 candidatos más
largos (27-36s) fueron los que el ranking rechazó (score 4/10). Lectura: el
contenido de casino/tragamonedas tiene ráfagas verbales cortas y punzantes
(una exclamación al ganar) separadas por tramos "muertos" de la tirada
girando — a diferencia de una charla de streaming grupal o una historia
armada, acá lo genuinamente bueno dura poco por naturaleza del género, no
porque el sistema esté fallando. Sigue siendo una hipótesis razonable, no
un hecho 100% confirmado — falta contrastar con más videos de casino para
saber si el mínimo de 21.7s del corpus general no aplica bien a este
género específico (parecido al hallazgo de `dato_sorprendente` necesitando
más pausa que `debate_polemica` — cada tipo de contenido tiene su propio
ritmo, no hay un número único).

**CORRECCIÓN a lo de arriba, misma noche**: la hipótesis "es el género, no
un bug" estaba incompleta. El usuario miró el video ENTERO a mano y señaló
un momento real (min 29:47-30:13, "Oh my God... la concha de la lora,
amigo... ¡Sí, señor! Volvimos, volvimos") que el sistema sí detectó (pico
de audio real, 4.06 de intensidad) pero arruinó en el camino, por DOS bugs
reales, no por naturaleza del género:

1. **`_AUTO_CUT_MIN_GAP` en `server.py` seguía en 0.8s** — quedó sin
   actualizar cuando subí el umbral a 1.5 en otro lado (`_EDIT_STYLE_MIN_GAP`
   y `_MICRO_CUT_MIN_GAP` de `analyze.py`) más temprano esa misma noche. Con
   el umbral viejo, este candidato puntual quedó con 5 cortes separados
   (21.6s sacados de 31.3s), un Frankenstein de 6 pedacitos que cortaba
   ANTES del remate real. Corregido a 1.5, y agregado `_MAX_AUTO_CUTS = 3`
   como tope defensivo (si hacen falta más de 3 cortes, es señal de que el
   tramo es demasiado discontinuo para el auto-recorte automático).
2. **`_fit_clip` estiraba 2 segmentos hacia atrás por cada 1 hacia
   adelante** — para un candidato forzado por un pico de audio (ancla de un
   solo índice, sin desarrollo previo que rescatar), el remate casi siempre
   viene DESPUÉS del pico, no antes. Con el viejo 2:1, la ventana paraba en
   el segundo 1805.6 ("Oh, my God") sin llegar a "la concha de la lora...
   ¡sí, señor!" a los 1812-1815. Cambiado a proporción 1:1 — mismo caso real
   ahora llega hasta 1814.6, alcanzando el remate. Verificado que no rompe
   el caso anterior (colesterol sigue dando 10.0s exactos).

Los tres arreglos (umbral, tope de cortes, proporción de estiramiento)
salieron de UN SOLO timestamp real que el usuario marcó mirando el video
completo — mucho más efectivo que cualquier estadística agregada del
corpus. Método a repetir: cuando el usuario señale un momento puntual que
el sistema se perdió, rastrearlo hasta la causa exacta (¿se detectó?
¿dónde se rompió?) en vez de ajustar prompts a ciegas.

## Método nuevo para seguir SIN que el usuario marque nada a mano (2026-08-21)

El usuario se fue a dormir pidiendo explícitamente seguir de forma
autodidacta: analizar videos, generar clips, comparar contra lo real. Encontré
una forma concreta y gratuita de hacer esto sin necesitar que él marque
momentos: **yt-dlp expone el heatmap de "más repetido" de YouTube**
(`info['heatmap']`, 100 puntos con `start_time`/`end_time`/`value` 0-1 — ya se
usaba en `research_batch.py::get_heatmap()` para el corpus, pero nunca para
videos sueltos analizados en la app). Es la señal más objetiva que existe:
no es intuición de nadie, es qué partes REALMENTE re-mira la audiencia real.

Verificado con `Pah7u3Ja-lY`: el momento que el usuario marcó a mano
(1789-1813s) resultó ser el **2do tramo más repetido de todo el video**
(0.891 sobre 1.0) — confirma con datos externos que su instinto era
correcto, no casualidad.

**Encontrado revisando el pico #1** (211.9-235.5s, valor=1.000, el tramo
MÁS repetido de todo el video): tiene picos de audio reales fuertes (uno de
6.44, el más alto medido en este video) y un momento visual notable a los
234-237s ("un hombre sonríe mientras se toca el brazo, mostrando una
herida...") — pero el candidato final quedó en una ventana de **0.2
segundos**, rechazada por "descripción vaga, no memorable".

**Causa raíz, DIAGNOSTICADA COMPLETA (2026-08-21, madrugada)** — bug real,
no arreglado todavía a propósito (ver por qué abajo):

`_merge_visual_segments` inserta los momentos visuales como pseudo-líneas
sintéticas en la lista de segmentos que usa `_fit_clip` (texto tipo
"(visual, sin diálogo) <descripción>", con `words: []` — CERO palabras,
porque no viene de audio real). El pico de energía de audio cercano ancló
un candidato en el índice de ESA pseudo-línea (index 75, 234.0-237.0s).
`_fit_clip` extendió bien la ventana a partir de ahí: `(215.57, 237.0)`,
21.4s — verificado, esto funciona bien.

El problema es el paso DESPUÉS: `detect_micro_cuts` mide pausas contando
huecos entre PALABRAS consecutivas. La pseudo-línea visual no tiene
palabras — entonces, entre la última palabra real antes de ella (fin de
"¡Los diseñadores!", ~222.3s) y la primera palabra real después (inicio de
"Nos quedan dos horas...", ~238.2s) hay un hueco de ~16 segundos que el
detector interpreta como UNA PAUSA GIGANTE recortable — y la corta entera,
llevándose puesto el momento visual completo (que está JUSTO en el medio
de ese "hueco"), dejando apenas 0.2s de sobra. **El auto-recorte no sabe
distinguir "silencio real" de "silencio con contenido visual notable
adentro" — trata highlights sin diálogo exactamente igual que aire
muerto, y se los come.**

Esto es más grave que los otros 3 bugs de esta noche porque ataca
específicamente el caso que la señal visual existe PARA resolver (recordar
el hallazgo de 17.3% de videos con contenido visual-only) — un momento
visual fuerte sin diálogo es exactamente lo más vulnerable a este bug.

**ARREGLADO** (más simple de lo que pensé al principio — no hizo falta
tocar `detect_micro_cuts`/`apply_micro_cuts` en varios lugares): en
`_merge_visual_segments` (`clip_engine/analyze.py`), la pseudo-línea
visual sintética pasó de `"words": []` a una sola palabra sintética que
cubre todo el tramo (`[{"start": m["start"], "end": m["end"], "word":
"[visual]"}]`). Con eso, el detector de pausas ya no ve un hueco gigante
donde está el momento visual — mide el silencio real a cada lado, sin
tragarse el contenido del medio. Cambio de UNA sola función, sin tocar
firmas en otros archivos.

Verificado con el caso real: antes el highlight de 21s quedaba en 0.2s
(contenido visual borrado por completo); con el arreglo, la ventana final
queda en 4 pedacitos/10.1s — sigue algo fragmentado, pero el momento
visual real (234.0-241.3s) YA NO desaparece, está presente en el último
pedazo. Regresión chequeada: el caso de audio puro (sin ningún momento
visual involucrado, el del "festejo/la concha de la lora") da EXACTAMENTE
el mismo resultado que antes de este cambio — no se tocó nada para ese
camino, como era de esperar (el fix solo afecta segmentos que vienen de
`_merge_visual_segments`).

Nota de por qué inicialmente pensé que era más riesgoso: la primera idea
que se me ocurrió (pasarle `visual_moments` a `detect_micro_cuts` en cada
lugar donde se llama) SÍ hubiera sido un cambio grande. La que terminé
aplicando es mucho más chica — vale la pena, antes de asumir que un fix
necesita tocar muchos archivos, buscar si hay una intervención más
temprana en el pipeline que resuelva lo mismo con menos superficie de
cambio.

**Plan para seguir de forma autodidacta sin el usuario presente**: para
cada video real ya descargado, bajar su heatmap (gratis, ya con el fix de
SSL/truststore + `YT_DLP_FIX_ARGS` que usa el resto del proyecto), cruzar
los picos más altos contra qué generó/rechazó el pipeline, y tratar cada
desajuste real como un caso de diagnóstico como los de arriba — mismo
método que con el timestamp que dio el usuario, pero de origen automático.
Ojo: no todos los videos tienen heatmap (`KUiIY5I9FyA` dio 0 puntos,
probablemente por pocas vistas — YouTube necesita un mínimo para generar
el gráfico). No asumir que la ausencia de heatmap significa error propio.

**Segundo caso real cruzado con heatmap** (`qPTNRAo0yRU`, IRL de Westcol,
2026-08-21): 7 recomendados, 2 caen a metros de picos reales (844s cerca
de un pico de 0.92, 1013s cerca de uno de 0.59 — aciertos genuinos). Pero
el tramo MÁS repetido de todo el video (2271-2346s, tres picos seguidos
incluyendo el #1 en 1.00) quedó sin un solo candidato. Causa: un segmento
de Whisper roto de **51.6 segundos** ("Va a perder camisa.") — la
transcripción colapsó un tramo largo (probablemente muy visual/físico,
poco diálogo claro) en un blob gigante casi sin texto. Corrió en modo
"rápido" (sin imagen) — hipótesis fuerte: con análisis visual encima esto
se hubiera detectado (coincide con el patrón ya confirmado de 17.3% de
contenido visual-only). **No confirmado todavía con "completo" en este
video específico** — sería el test natural que sigue si hay crédito:
correr `qPTNRAo0yRU` con visual y ver si ahora sí aparece un candidato en
2271-2346s.

Patrón que se repite en los dos casos cruzados con heatmap: los momentos
más repetidos que el sistema se pierde tienden a ser tramos con MUY poco
diálogo útil (silencio con contenido visual, o transcripción rota) — no
error de criterio del LLM, error de que la señal de texto sola no alcanza
ahí. Refuerza (por tercera vez esta noche, con evidencia independiente)
que lo visual no es opcional para este tipo de contenido.

**Hipótesis probada — resultado MIXTO, no una victoria limpia**: corrí
`qPTNRAo0yRU` en modo "completo" (con imagen) para ver si esta vez sí
aparecía un candidato en el tramo más repetido (2271-2346s). Resultado
honesto: apareció UN candidato visual cerca (2348.8s, "hombre con gorra
negra grita y señala... ambiente festivo") — pero está en el BORDE del
tramo real (2.8s después de que termina), no adentro, y se rechazó por
"genérico" (score 4), el mismo patrón de rechazo de siempre para
candidatos visuales poco específicos. No es que el arreglo de esta noche
haya fallado — es que el análisis visual encontró ALGO ahí, cerca, pero
ni la ubicación ni la descripción fueron lo bastante precisas como para
que sobreviva. Conclusión: la señal visual ayuda pero no resuelve esto
solo — el intervalo de muestreo (cada 3s) puede no caer justo en el
instante correcto de un tramo de 75s, y la descripción del refine sigue
sin ser lo bastante específica para este tipo de acción física/caótica
(algo se rompe/vuela, gente reaccionando). Sin resolver del todo — próximo
paso si se retoma: mirar el frame real en 2270-2350s a mano (extraer con
ffmpeg) para entender qué pasa ahí visualmente y si el prompt de refine
necesita vocabulario específico para este tipo de acción física.

Nota honesta sobre el ritmo de esta madrugada: van 4 bugs reales
encontrados y arreglados, más este hallazgo parcial sin resolver. Es
mucho para una sesión — quedó documentado con detalle suficiente como
para retomar sin perder nada, no hace falta forzar una resolución
completa de todo en una sola noche.

**Cierre del hilo del tramo 2270-2350s** (extraje frames a mano con
ffmpeg para ver qué hay ahí de verdad): es una revelación de multitud
ENORME — el streamer mirando desde un balcón/reja a una multitud masiva
de gente y motos abajo, con fotógrafos filmando. Visualmente espectacular,
pero con audio de gritos de multitud, sin diálogo limpio (coincide con el
segmento roto de Whisper de 51.6s). El muestreo visual (cada 3s en modo
"completo") sí pasó por esta zona, pero el frame que terminó puntuando
"notable" fue uno más genérico (alguien con gorra gritando y señalando)
en vez de la toma panorámica de la multitud — cuestión de qué instante
exacto cae en el muestreo de 3s, no un bug de lógica. No es algo para
arreglar con código a esta hora: es un trade-off de densidad de muestreo
ya documentado y evaluado (cuánto más denso, más caro — ver la sección de
costos reales más arriba). Considero este hilo suficientemente explorado
por ahora — quedó entendido el POR QUÉ, no hace falta seguir cavando la
misma pregunta esta noche.

## Señal nueva: heatmap de "más repetido" de YouTube como candidato forzado
(implementado y verificado end-to-end, 2026-08-21)

Investigación de competencia (pedida explícitamente por el usuario esa
noche) reveló que **Opus Clip usa el heatmap de "más repetido" de YouTube
como mecanismo PRINCIPAL de selección** — no solo referencia. Coincide
exacto con el método de diagnóstico que veníamos usando esa madrugada
(cruzar picos reales contra lo que el pipeline generaba). Se implementó
como una señal más de las que ya fuerzan candidatos (mismo patrón que
picos de audio y momentos visuales):

- **`clip_engine/heatmap.py`** (nuevo): `get_heatmap(video_stem)` — el
  video_stem de algo bajado de YouTube ES el video ID, así que no hace
  falta guardar la URL original en ningún lado, se reconstruye directo.
  Cachea en disco (`{stem}.heatmap.json`), degrada a `None` sin romper
  nada si el video no tiene heatmap (subido a mano, o pocas vistas —
  confirmado real: `KUiIY5I9FyA` dio 0 puntos).
- **`_candidates_from_heatmap`** en `analyze.py`: fuerza un candidato por
  cada tramo del heatmap con valor ≥0.6 (normalizado por video, no
  absoluto). A diferencia de los otros dos mecanismos de forzado, este
  NO se salta tramos ya "cubiertos" por un candidato débil existente —
  se agrega siempre con score 7 (más alto que el 6 de audio/visual) para
  que gane el desempate en `_dedupe_by_score`. Motivo: probado en vivo,
  con el chequeo de "ya cubierto" activado, dos picos reales de heatmap
  quedaron afuera porque un candidato de audio flojo ya ocupaba esa
  ventana — el heatmap es la señal más confiable que hay (datos reales
  de audiencia), no debería perder contra una heurística nuestra.
- **`RANKING_SYSTEM_PROMPT`**: nueva instrucción para candidatos "(más
  repetido, X%)" — dale mucho más beneficio de la duda que a un
  candidato común, no lo rechaces por "sonar genérico" (esa razón no
  aplica acá, ya hay respaldo real de audiencia).

**Verificado de punta a punta con plata real, sobre `Pah7u3Ja-lY`**: los
dos momentos que el pipeline se había perdido toda la noche (el que marcó
el usuario a mano, "la concha de la lora", y el que encontré después,
"los diseñadores") ahora SÍ generan candidatos de heatmap. Uno de los dos
("los diseñadores") quedó recomendado (`keep=true`, score 8) con el
verdict citando explícitamente *"el respaldo de ser uno de los más
repetidos lo hace destacar"* — la instrucción nueva está funcionando tal
como se escribió. El otro ("la concha de la lora") recibió el mismo trato
positivo (score 7, verdict igualmente bueno) pero no entró en el tope de
`target_clips` pedido esa corrida por competir parejo con otros 8
candidatos — no fue un rechazo de calidad, fue un límite de cupo.

Iteración del arreglo (documentado por transparencia, no solo el
resultado final): la primera versión SÍ tenía el chequeo de "ya cubierto"
activado y los candidatos de heatmap nunca sobrevivían — encontrado
comparando contra un dry-run sin gastar API antes de asumir que
funcionaba. Vale la pena repetir este hábito: verificar con un test en
frío (sin LLM) antes de gastar en una corrida completa.

## Loop de mejora autodirigido, mandato explícito (2026-08-22)

El usuario pidió explícitamente, antes de irse, un modo de trabajo distinto
al de "arreglar bugs sueltos": un LOOP disciplinado — generar candidatos,
compararlos contra clips que YA sabemos que funcionaron, identificar el
error específico, ajustar el prompt/criterio, volver a correr, verificar
si mejoró o empeoró (revertir si empeoró), y repetir, documentando cada
ciclo — hasta consumir el crédito disponible. Con un resumen final
obligatorio: qué se mantuvo y por qué, qué se descartó, qué patrones
nuevos por categoría/tipo de gancho, y el criterio final recomendado.

### Ciclo 1 — ventana de `_fit_clip` completamente desconectada del ancla

**Paso 1-2 (generar + comparar contra señal real)**: re-corrí `qPTNRAo0yRU`
en modo rápido y crucé el resultado contra su heatmap real de YouTube. El
pico #1 de todo el video (2296-2321s, valor 1.00 — el tramo más repetido)
no generó NINGÚN candidato, ni evaluado ni rechazado — desapareció antes
de llegar a la validación.

**Paso 3 (identificar el error específico)**: reproducido de forma
aislada. El segmento de faster-whisper que ancla ese punto es un blob roto
de 51.66s ("Va a perder camisa.") con solo 4 palabras reales pegadas a los
bordes y ~50.6s de hueco interno — la misma familia de problema que el
"revelación de multitud" de la sección anterior (contenido muy visual/poco
diálogo colapsado en un segmento gigante por Whisper). `effective_dur()`
mide esto como ~1s de contenido real, así que el loop de estirado de
`_fit_clip` sale a buscar desesperadamente más contenido real — y el
recorte a `hard_max` que corre DESPUÉS no tenía ningún límite sobre hasta
dónde podía achicar la ventana: como el segmento roto por sí solo (51.66s)
ya supera `hard_max` (43.75s), el recorte seguía de largo empujando el
borde `hi` mucho más allá del propio segmento ancla, dejando una ventana
final en OTRA PARTE del video — verificado en aislado:
`_fit_clip` devolvía `[(2160.12, 2194.72)]`, a más de 130s del ancla real
(2291-2343s), sin ninguna superposición.

**Paso 4 (ajustar) — dos intentos, documentados los dos**:
- *Intento 1 (insuficiente)*: agregar un tope de "reloj de pared" al loop
  de ESTIRADO (`wall_clock_cap = max_dur * 3`), para que no camine
  indefinidamente lejos del ancla. No alcanzó — probado, dio
  `[(2161.88, 2194.72)]`, prácticamente idéntico al resultado roto. Causa:
  el bug real no estaba en el estirado (que sí camina, pero de forma
  acotada), sino en el recorte posterior a `hard_max`, que no respetaba
  ningún límite y podía comerse el propio segmento ancla.
- *Intento 2 (el que funcionó)*: además del tope de estirado (ahora por
  dirección independiente, no las dos a la vez), el recorte a `hard_max`
  ahora tiene prohibido reducir `hi` por debajo del índice del segmento
  ancla original, o subir `lo` por encima de él — nunca puede comerse el
  ancla, aunque eso signifique devolver un clip más largo que `max_dur`
  (peor caso: el segmento roto entero, 51.66s, en vez de basura de otra
  parte del video).

**Paso 5 (re-correr y verificar)**: `_fit_clip` sobre el mismo caso ahora
da `[(2291.58, 2343.24)]` — el segmento ancla completo, correctamente
ubicado en el punto real. Regresión chequeada contra los dos casos ya
verificados esta sesión: colesterol (`panchos_test`) sigue dando ~10.8s
bien anclado, festejo (`Pah7u3Ja-lY`, "la concha de la lora") sigue
llegando hasta el remate real (~1815.5s, antes 1814.6s) — sin
empeoramiento en ninguno de los dos.

**Paso 6 (decisión)**: SE MANTIENE. Mejora clara y verificada, sin
regresión en los dos casos de referencia.

**Confirmación de escala, hecha después (sin costo de LLM)**: escaneado el
corpus completo (`data/research/transcripts`, 72.600 segmentos reales)
buscando la misma familia de segmento patológico (≥15s de duración, con
menos del 15% de ese tiempo realmente ocupado por palabras) — 139 casos
reales encontrados en el corpus. No es un bug de un solo video: es un
patrón que aparece con regularidad real (~0.19% de los segmentos, pero
139 puntos concretos donde el código viejo podía producir una ventana
desconectada del ancla). Confirma que valía la pena arreglarlo a nivel de
`_fit_clip` en vez de tratarlo como un caso aislado de un video.

**Nota honesta sobre el resultado real, no solo el mecanismo**: este
arreglo resuelve el bug MECÁNICO (ventana desconectada del ancla) pero no
resuelve por sí solo si el CLIP final va a ser bueno — como el segmento
roto tiene casi cero diálogo real, cuando el auto-recorte de pausas
(`_AUTO_CUT_MIN_GAP`, en `server.py`, tiempo de generación de preview) se
aplique más tarde sobre este candidato, muy probablemente le va a comer
casi todo el hueco de 50s interno, dejando un clip final muy corto — a
menos que el análisis visual (modo completo) aporte una pseudo-palabra
sintética ahí (mismo mecanismo que `_merge_visual_segments`) que le dé
algo que no sea puro silencio. Esto conecta directo con el hallazgo ya
cerrado esa madrugada sobre `qPTNRAo0yRU` en modo completo (tramo
2270-2350s, revelación de multitud): la señal visual encontró algo cerca
pero no lo bastante específico como para sobrevivir el ranking. No se
reabre esa investigación — ya está documentada arriba como explorada y
entendida. Lo que este ciclo aporta es distinto y real igual: antes, este
tipo de segmento roto arruinaba la UBICACIÓN del candidato (ni siquiera
llegaba a la validación); ahora al menos llega, anclado en el lugar
correcto, para que el resto del pipeline (visual + ranking) tenga la
oportunidad de juzgarlo bien o mal — ya no se pierde por un bug mecánico
antes de esa oportunidad.

### Ciclo 2 — la marca de heatmap se perdía contra un candidato del LLM

**Paso 1-2**: para seguir comparando contra señal real (no solo intuición),
crucé OTRO video real con heatmap+candidatos ya generados,
`u1O6Av1vO-I` (no tocado hasta ahora en la sesión de esta noche). El
candidatos file existente era de ANTES de que existiera el forzado por
heatmap (se conservó una copia como
`u1O6Av1vO-I.streaming.candidates.BEFORE_HEATMAP.json` para el
antes/después). Cruzando contra los picos reales: el pico #2 de todo el
video (659-668s, valor 0.74 — el "más repetido" después del pico #1) tenía
CUATRO candidatos reales cercanos (0.74, 0.71, 0.62, 0.51) y el pipeline
viejo los encontró TODOS pero los rechazó TODOS — sin ninguna marca de
heatmap, porque el archivo es previo a esa función.

**Paso 3 (identificar el error, tras re-correr con el pipeline actual)**:
re-corrí el mismo video con el código de esta noche (heatmap forcing +
ambos arreglos de `_fit_clip`). Mejora parcial: el pico 647-668s ahora SÍ
sale marcado `(más repetido, 71%)` y queda recomendado (score 7) — el
mecanismo de forzado funciona. Pero el pico #1 de TODO el video (202-211s,
valor 1.00) seguía sin ninguna marca de heatmap, cubierto por un candidato
genérico del LLM (`"El DJ más grande y su pasado"`, dato_sorprendente) que
además fue RECHAZADO (score 6). Causa exacta: `_candidates_from_heatmap` SÍ
generó un candidato forzado para ese pico (confirmado: "5 candidatos
forzados por el heatmap" en el log, contando los 5 puntos ≥0.6 reales de
este video), pero `_dedupe_by_score` — que se queda con el de MAYOR score
entre ventanas superpuestas — prefirió un candidato generado por el LLM que
ya traía un score de generación ≥7, empatando o ganando al score fijo (7)
del candidato de heatmap. Al perder el desempate, el candidato de heatmap
desaparece del todo — se pierde tanto la ventana (que estaba mejor anclada
al pico real) como la marca en el título, y con la marca, la indulgencia
especial que le da `RANKING_SYSTEM_PROMPT` a "(más repetido, X%)". La señal
de audiencia real quedaba disponible en el pipeline pero nunca llegaba al
prompt de validación para ESE candidato puntual.

**Paso 4 (ajustar)**: agregado un paso nuevo en `analyze()`
(`clip_engine/analyze.py`), DESPUÉS de `_dedupe_by_score` — para cada
candidato sobreviviente que NO tenga ya la marca de heatmap en el título,
se revisa si su ventana se superpone con algún punto del heatmap real
≥`_HEATMAP_MIN_VALUE`, sin importar de qué mecanismo haya salido el
candidato originalmente. Si hay superposición, se le agrega la marca
`(más repetido, X%)` al título y una línea al `reason` — así la señal de
audiencia llega al ranking pase lo que pase en el desempate del dedupe.

**Paso 5 (re-correr y verificar)**: mismo video, mismo código +
este arreglo. El candidato que cubre el pico #1 (194-206s, ahora
"`(más repetido, 64%) No, pero fue hace mucho.`") pasó de RECHAZADO sin
marca a RECOMENDADO (`keep=true`, score 6) con la marca presente. El pico
647-661s se mantiene igual que antes (regresión limpia — el guard
`if "(más repetido" in title: continue` evita el doble-marcado). Total:
12 candidatos evaluados, 6 recomendados, 3 con marca de heatmap (antes: 2
de 9, y uno de los picos más fuertes del video sin marca ni recomendación).

Regresión chequeada también sobre `Pah7u3Ja-lY` (el caso ya verificado
antes esta noche): sigue generando candidatos marcados
`(más repetido, 72%/78%/68%)` con normalidad, sin duplicados ni títulos
corruptos.

**Paso 6 (decisión)**: SE MANTIENE. Mejora real y verificada sobre un
tercer video (ninguno de los dos videos usados para construir el mecanismo
original), sin regresión en el caso ya verificado.

**Patrón que deja este ciclo, más allá del bug puntual**: cuando una señal
de refuerzo (heatmap, y por extensión el mismo riesgo aplicaría a picos de
audio/visuales) compite por una ventana contra un candidato del LLM y
pierde el desempate, no alcanza con que el mecanismo de forzado exista —
hay que verificar también que la señal SOBREVIVE el desempate, o
reconectarla después si no. Vale la pena tenerlo en cuenta si en el futuro
se agrega alguna otra señal externa forzada de la misma familia.

### Confirmación con volumen (no un fix, un hallazgo que valida algo ya hecho)

`research_db.py` (nuevo esta noche, ver más abajo) ya tenía sincronizados
133 items del corpus con AMBOS pico de audio y heatmap real medidos. Vale
la pena mirar ese número con cuidado en vez de dejarlo suelto: la posición
del pico de audio más fuerte y la posición del tramo más repetido real
difieren, en promedio, 0.56 (mediana 0.61) sobre una escala 0-1 donde 0 es
"coinciden exacto" — PEOR que el ~0.33 que da el azar puro entre dos
posiciones independientes. Solo 12.8% de los casos caen a menos del 10%
de distancia; 60.9% caen a más de la mitad del video de distancia.

**Matiz importante, desglosando por tipo (no mezclar los dos)**: ese 0.56
mezcla clips cortos ya publicados (n=122, media 0.572, mediana 0.654 —
acá el patrón es fuerte) con videos largos (n=11, media 0.387, mediana
0.411 — mucho más cerca del azar, ~0.33). El video largo es el caso
PRINCIPAL de la app (subís tu propio contenido) y con n=11 no alcanza
para afirmar lo mismo ahí — no corresponde generalizar el hallazgo fuerte
de los clips cortos al caso principal sin más datos. Lo que sí se sostiene
con ambos: en ningún caso el pico de audio parece un proxy CONFIABLE del
momento más repetido — en el peor caso (clips) es claramente malo, en el
mejor caso (largos, muestra chica) no es mejor que el azar tampoco.

Esto no generó un cambio de código — al revisar `RANKING_SYSTEM_PROMPT`,
los candidatos "(pico de audio)" YA se juzgan por especificidad de la
descripción, sin el beneficio de la duda que sí se le da a "(más
repetido)" (que tiene respaldo real de audiencia detrás). El diseño ya
estaba calibrado en la dirección correcta antes de tener este número — el
hallazgo confirma con volumen real que esa distinción (dato real de
audiencia vs. heurística nuestra) no era solo cautela de más, hay una
diferencia real y medible entre ambas señales. Documentado como evidencia
de que "no tocar" también puede ser la decisión correcta de un ciclo,
no todo ciclo tiene que terminar en un cambio de código.

## Base de features del corpus, productizada (2026-08-22)

`research_db.py` (ya existía como script suelto, ahora ampliado) es la
base SQLite reproducible que CLAUDE.md pedía como pendiente ("que quede
como script reproducible en vez de comandos sueltos"). Comandos:

- `python research_db.py --sync` — lee manifest + JSON crudos, puebla la
  tabla `items` (no pisa `hook_type` ya clasificado en corridas previas).
- `python research_db.py --classify [--classify-limit N]` — clasifica con
  gpt-4o-mini el `hook_type` (misma taxonomía de 6 categorías que usa el
  pipeline real) de los clips del bucket `clips` que todavía no lo tienen.
  Costo real medido: $0.0006 por 10 clips (~$0.09 para todo el corpus de
  ~1550 clips) — insignificante frente al crédito disponible.
- `python research_db.py --breakdown` — el desglose que pide el mandato:
  duración, silencio antes de arrancar, mediana/p75 de huecos internos y
  % que cierra con "?"/"!", agrupado POR hook_type (no un solo número
  agregado del corpus entero) + ejemplos reales de apertura por categoría.
- `python research_db.py --stats` — panorama general (cobertura de
  heatmap/audio, la comparación de arriba, etc.).

Nuevas columnas en `items` (antes solo existían para picos de audio/visual
agregados, nunca por hook_type): `silence_before_first_word`,
`median_internal_gap`, `p75_internal_gap`, `opening_text`, `closing_text`,
`hook_type`. Clasificación corriendo en background al momento de escribir
esto — pendiente correr `--breakdown` con el corpus completo clasificado
y, si aparece un patrón real por tipo de gancho, ajustar
`SYSTEM_PROMPT`/`RANKING_SYSTEM_PROMPT` en base a ESO (no a la intuición
que ya está escrita ahí sobre pacing por hook_type — esta es la
oportunidad real de confirmarla o corregirla con datos, tal como pide el
mandato).

### Observación abierta, NO convertida en fix — pico de audio en el outro

Revisando picos de audio del bucket `largos` uno por uno (no solo el
agregado), encontré un patrón mecánico claro en 2 de 11 videos:
`Od2j1CgDoFk` (z=6.44, el pico más fuerte de todo el video) y
`wkm0VEQDAuQ` (z=10.18, el más fuerte con margen grande) tienen su pico de
audio más extremo en los últimos 12-20 segundos del video, justo DESPUÉS
de que termina el último diálogo transcripto — probablemente música de
outro/cierre, no contenido real (el patrón se repite: silencio de texto
después de la última línea hablada, coincidiendo exacto con el spike).

Investigué si esto genera daño real antes de tocar código: revisé los 5
videos de test reales de esta sesión (`*.streaming.candidates.json`) — CERO
candidatos "(pico de audio)" con `keep=true` en ninguno. La instrucción ya
existente en `RANKING_SYSTEM_PROMPT` (juzgar estos candidatos por
especificidad del contenido, no por la magnitud del pico) parece estar
filtrando correctamente cualquier cosa floja, música de outro incluida —
no encontré un solo caso real de un clip final malo por esta causa. NO se
tocó código: es una observación mecánicamente explicada pero sin evidencia
de daño real todavía, así que no corresponde "arreglar" algo que no está
demostrado que rompa nada — queda anotado por si en el futuro aparece un
caso real donde sí importe (por ejemplo, si algún día un pico de outro le
gana un desempate de dedupe a un candidato mejor, cosa que no vi pasar
todavía).

### Ciclo 3 — el primer ajuste de CRITERIO real (no un bug), con datos por hook_type

Este es el ciclo que más directamente responde al mandato: no un bug
mecánico, sino corregir el CRITERIO de pacing por tipo de gancho en
`SYSTEM_PROMPT` usando evidencia real en vez de intuición — y encima,
encontrar que la intuición anterior decía tener respaldo de datos
("medido sobre miles de clips reales exitosos") cuando en realidad nunca
se había medido por hook_type, solo agregado.

**Paso 1-2 (generar datos reales + comparar)**: con la clasificación de
`research_db.py --classify` terminada (1557/1577 clips reales, gpt-4o-mini,
costo total real ~$0.10), corrí `--breakdown` para ver duración/silencio
inicial/pausa interna agrupados por hook_type — algo que nunca existía
desglosado, solo como agregado del corpus entero.

**Paso 3 (identificar el error — en DOS capas)**:

1. *Contaminación de datos, encontrada primero*: la primera corrida del
   breakdown daba `emocional` con 91.9s de duración promedio — más de 4
   veces cualquier otra categoría. Investigado antes de confiar en el
   número: el outlier era un item de **19,580 segundos (5.4 horas)**, con
   título genérico `"(nuevo) <id>"` — un stream completo colado en el
   bucket `clips` del manifest, no un clip editado real. Encontrados 11-16
   casos así (de 1577). Corregido en `research_db.py::breakdown_by_hook_type`
   con un techo de duración plausible (`_MAX_PLAUSIBLE_CLIP_DURATION = 180s`)
   — no se tocan ni se borran esos items de la DB, solo se excluyen de este
   desglose estructural específico. Con el filtro, `emocional` bajó a 28.6s,
   un número creíble y en línea con el resto.

2. *El error de criterio real, una vez limpios los datos*: el párrafo de
   pacing por hook_type en `SYSTEM_PROMPT` (agregado esta misma noche, antes
   de este ciclo) decía "`hook_fuerte` tarda más en desarrollarse que el
   resto" y "`dato_sorprendente` necesita más aire... que un chiste" —
   afirmando además que esto estaba "medido sobre miles de clips reales
   exitosos, no a ojo". Eso último era falso: nunca se había medido por
   hook_type hasta este ciclo, solo el agregado general. Y una vez medido de
   verdad (n=74 hook_fuerte, n=48 dato_sorprendente, n=711 gracioso, n=385
   emocional, n=297 debate_polemica, n=30 consejo_practico — final, corpus
   casi completo):
   - **Duración: `hook_fuerte` es de las MÁS CORTAS (20.6s), no la más
     larga** — prácticamente empatado con `dato_sorprendente` (21.3s), muy
     por debajo de `debate_polemica` (29.0s) y `emocional` (28.6s). La
     afirmación original estaba invertida.
   - **Pausa interna: `hook_fuerte` SÍ es un outlier real, pero en pausa, no
     en duración** — mediana 1.46s, p75 1.99s, casi el doble que cualquier
     otra categoría (la siguiente más alta, `gracioso`, tiene mediana
     0.83s). `dato_sorprendente` (0.83s) no se diferencia casi nada de
     `gracioso` (0.83s) ni de `emocional` (0.71s) — la afirmación de que
     "necesita más aire que un chiste" no tiene respaldo real, es
     prácticamente idéntico a un chiste en este eje.
   - `debate_polemica` (mediana 0.40s, arranque con 0.19s de silencio
     inicial — el más inmediato de todos) SÍ se confirmó como el más
     atropellado, y `consejo_practico` (mediana 0.46s) resultó pertenecer al
     mismo grupo de baja pausa, algo que el párrafo original no mencionaba.

**Paso 4 (ajustar)**: reescrito el párrafo de pacing en `SYSTEM_PROMPT`
(`clip_engine/prompts.py`) con la relación correcta: `hook_fuerte` no dura
más, tiene la pausa interna más larga por lejos (el silencio ES la tensión,
no un desarrollo más largo); `debate_polemica` + `consejo_practico` forman
el grupo atropellado (antes solo se mencionaba `debate_polemica` solo);
`gracioso`/`dato_sorprendente`/`emocional` agrupados en pausa media, sin
diferencias grandes entre ellos (antes se le atribuía a `dato_sorprendente`
una necesidad especial de aire que los datos no sostienen). La cita a
"miles de clips... no a ojo" se corrigió a la cifra real (~1557) y a la
fecha/comando exacto de la medición, para que quede trazable.

**Paso 5 (re-correr y verificar) — honesto sobre el límite de este ciclo**:
a diferencia de los ciclos 1 y 2 (bugs mecánicos de `_fit_clip`, donde pude
reproducir el antes/después con una función pura y comparar ventanas
exactas), esto es un cambio de CRITERIO en un prompt de LLM — no hay un
input/output determinístico para verificar "mejoró/empeoró" sin gastar en
una corrida real y sin que a su vez esa corrida puntual pruebe algo
statísticamente (una corrida de un video no es volumen). Lo que SÍ se
verificó: sintaxis del prompt (carga sin error), que los números citados
coinciden exactamente con la salida real de `--breakdown`, y que no quedó
ninguna cita a un respaldo de datos que no exista. La verificación de
comportamiento real (¿el LLM efectivamente deja más aire en momentos de
`hook_fuerte` ahora?) queda pendiente del próximo test en vivo con crédito
disponible — anotado para no perder el hilo, no cerrado como si ya
estuviera probado en producción.

**Paso 6 (decisión)**: SE MANTIENE el cambio de prompt (evidencia sólida,
corrige una afirmación demostrablemente invertida), con la verificación de
comportamiento real marcada como pendiente, no como hecha.

**Por qué este ciclo importa más que los dos anteriores para el mandato**:
los ciclos 1 y 2 arreglaron mecánica del pipeline (bugs que existían
independientemente de qué tan bien calibrado estuviera el criterio). Este
ciclo tocó el CRITERIO en sí — exactamente lo que el usuario pidió con más
énfasis ("Y CONTEXTO PROMPT ALGO MEJORASTE? TE BASAS MUCHO EN LOS BUGS Y NO
EN LO IMPORTANTE") — y de paso encontró que una afirmación de criterio ya
escrita esta noche decía tener respaldo de datos que en realidad no tenía
todavía. Ojo para el futuro: cualquier cita a "medido sobre datos reales"
en un prompt debería ser verificable con un comando real (como
`research_db.py --breakdown` ahora), no una frase que suena a evidencia sin
serlo.

### Resumen INTERMEDIO del loop (2026-08-22, madrugada) — el mandato sigue activo

El usuario pidió explícitamente seguir este loop de forma autónoma "hasta
consumir los créditos" y documentar cada ciclo para un resumen final. Esto
NO es ese resumen final — es un corte intermedio para que quede claro el
estado si se retoma antes de que el loop termine solo.

**Qué se mantuvo y por qué**:
1. Fix de `_fit_clip` (Ciclo 1): tope de estirado por dirección + el
   recorte a `hard_max` nunca puede comerse el segmento ancla original.
   Antes, un segmento roto de Whisper con casi todo hueco interno podía
   producir una ventana final a 100+ segundos de distancia de donde
   correspondía. Verificado con 3 casos reales (roto, colesterol, festejo).
2. Re-etiquetado post-dedupe de candidatos con respaldo de heatmap (Ciclo
   2): si un candidato del LLM le gana el desempate a uno forzado por
   heatmap, igual se le agrega la marca "(más repetido, X%)" si su ventana
   se superpone con un pico real — antes se perdía la señal de audiencia
   por completo. Verificado sobre un tercer video real (`u1O6Av1vO-I`) con
   mejora clara, sin regresión en el caso ya probado (`Pah7u3Ja-lY`).
3. Corrección del párrafo de pacing por hook_type en `SYSTEM_PROMPT`
   (Ciclo 3): la relación estaba invertida para `hook_fuerte` (no es el
   que más dura, es el que más pausa interna necesita) y sobre-atribuida
   para `dato_sorprendente` (no se diferencia de `gracioso` en pausa).
   Corregido con números reales de 1557 clips clasificados. Verificación
   de comportamiento en vivo (no solo de que no rompa nada) sigue
   pendiente — anotado explícitamente como tal.

**Qué se probó y se descartó (documentado, no solo lo que funcionó)**:
- Primer intento de fix del Ciclo 1 (tope de reloj de pared en el estirado
  solo) — insuficiente, no tocaba la causa real (el recorte posterior).
  Reemplazado por el fix que sí funcionó, no se mezclaron ambos.
- Hipótesis "el pico de audio más fuerte no correlaciona con el heatmap
  real" — confirmada con volumen (133 pares, peor que el azar) pero NO
  generó cambio de código: el prompt ya trataba estos candidatos con la
  cautela correcta antes de tener el número. Documentado como validación,
  no como fix.
- Pico de audio en el outro/cierre de 2 videos largos (z=6.44 y z=10.18,
  música sin diálogo) — patrón mecánico real, pero sin evidencia de que
  produzca un clip final malo (0 candidatos "(pico de audio)" con
  `keep=true` en los 5 videos de test reales). No se tocó código por falta
  de daño demostrado, no por falta de causa identificada.
- Patrón de cierre (última línea) por hook_type — inspeccionado
  cualitativamente y con el % que cierra en "?"/"!" (14.6%-28.4% según
  categoría). Señal débil comparada con la de pausa interna (que fue de
  ~2-3x de diferencia) — no se convirtió en regla de prompt para evitar
  sobreajustar a una diferencia que podría ser ruido.

**Patrones nuevos por categoría/hook_type, con evidencia (n real)**:
ver Ciclo 3 arriba — tabla completa reproducible con
`python research_db.py --breakdown`.

**Infraestructura que queda reutilizable** (no solo hallazgos puntuales):
`research_db.py` ahora sincroniza, clasifica (`--classify`) y desglosa
(`--breakdown`) el corpus completo — antes era comandos sueltos sin
persistir. Cualquier ciclo futuro puede arrancar de acá sin repetir
trabajo ya pagado (el `hook_type` clasificado no se vuelve a pisar en cada
`--sync`).

**Costo real acumulado en esta sesión, medido (no estimado)**: clasificación
del corpus completo ~$0.10 (gpt-4o-mini, 1557 clips). Las corridas de
verificación en vivo sobre videos de test ya tenían costo acumulado de
sesiones anteriores de esta misma noche (`qPTNRAo0yRU` $0.62,
`Pah7u3Ja-lY` $1.21, `u1O6Av1vO-I` $0.07 — total acumulado, no solo de
este segmento) — ver `clip_engine/cost_tracker.py::cost_summary` por
video. El loop sigue activo mientras haya crédito disponible; este corte
es solo para dejar registro, no un cierre.

### Ciclo 4 — verificando el Ciclo 3 encontré un bug MÁS grave que lo que buscaba

El usuario pidió explícitamente verificar en vivo si el cambio de pacing del
Ciclo 3 realmente cambiaba el comportamiento del LLM. Diseñé un A/B real:
mismo video (`Pah7u3Ja-lY`), mismo código, la ÚNICA variable es el párrafo
de pacing en `SYSTEM_PROMPT` (viejo vs nuevo), reconstruyendo el texto viejo
en memoria sin tocar el archivo, para aislar de verdad esa única variable.

**Lo que encontré NO fue sobre pacing** — en la corrida con el prompt nuevo
apareció un candidato real de **434.7 segundos** (más de 7 minutos),
`keep=false` pero evaluado como si fuera un clip válido. Rastreado hasta la
causa: el LLM propuso un rango de índices 156 a 399 (244 segmentos) para un
solo "momento" — casi seguro una alucinación del LLM en la generación de
candidatos, no un bug de transcripción esta vez.

**La causa real era MI PROPIO fix de esta madrugada (Ciclo 1)**: la
protección de "nunca recortes más allá del ancla original" que agregué para
el segmento roto de 51.66s usaba como criterio "¿el ancla ya es más grande
que `hard_max`?" — pero ese mismo criterio también es cierto para un rango
de 244 segmentos mal propuesto por el LLM, así que la protección terminaba
aplicándose ahí también, bloqueando por completo al recorte de `hard_max`
de hacer su trabajo. Verificado el mecanismo exacto reproduciendo
`_fit_clip` en aislado con el rango (156, 399): confirmaba los 434.7s.

**El error de diseño**: confundí "duración de reloj de pared ya grande"
con "no hay nada para recortar adentro". Son cosas distintas — un solo
segmento roto (1 segmento, 51.66s) genuinamente no tiene nada más chico
que devolver. Un rango de 244 segmentos SÍ tiene de sobra para elegir una
sub-ventana razonable; protegerlo entero es exactamente lo opuesto de lo
que hace falta.

**Fix**: cambié el criterio de protección de "duración > `hard_max`" a
"cantidad de segmentos del ancla original ≤ 20". Con ese cambio, verifiqué
los CUATRO casos juntos en la misma corrida:
- Rango de 244 segmentos (434.7s) → ahora 39.5s, dentro de `hard_max`.
- Segmento roto original de 51.66s (`qPTNRAo0yRU`, el caso que motivó el
  Ciclo 1) → sigue devolviendo el segmento completo, sin regresión.
- Colesterol (`panchos_test`) → sigue en ~10.8s.
- Festejo (`Pah7u3Ja-lY`, "la concha de la lora") → sigue llegando al
  remate real, ~38.4s.

**Decisión**: SE MANTIENE el fix nuevo (segment-count en vez de duración),
reemplazando el criterio del Ciclo 1 sin perder lo que ese ciclo arregló.

**Sobre la verificación ORIGINAL que pidió el usuario (pacing de
hook_fuerte) — resultado honesto, no concluyente**: una vez arreglado el
bug de los 434s, la corrida con el prompt NUEVO produjo 2 candidatos
`hook_fuerte` (la corrida con el prompt VIEJO no había producido ninguno
en el mismo video). Revisé la estructura de pausa interna de los dos:
- El que quedó `keep=true` ("¡20.000 dólares?") sí tiene una pausa real de
  1.68s antes del remate — pero sus límites salieron del forzado por
  heatmap (`_candidates_from_heatmap`), no del juicio del LLM guiado por
  el párrafo de pacing. O sea, este caso no prueba lo que quería probar.
- El otro ("¡Iniciamos la mejor parte!", rechazado) no tiene ninguna pausa
  real (todos los huecos entre palabras ≤0.2s), y leyendo el texto
  ("Señoras, señores... se da por iniciada la mejor parte del stream")
  no suena a tensión/confrontación en absoluto — parece una clasificación
  de `hook_type` cuestionable del LLM, no evidencia a favor ni en contra
  del pacing.

Conclusión honesta: con UNA sola corrida A/B, y con tan pocos candidatos
`hook_fuerte` genuinos por corrida, no se puede afirmar que el cambio de
prompt del Ciclo 3 esté funcionando como se espera todavía. Sigue
pendiente una verificación real — haría falta correr varios videos con
contenido de tensión/confrontación real y mirar específicamente candidatos
`hook_fuerte` generados por el LLM (no forzados por heatmap/audio) para
aislar el efecto de verdad.

**Lección del ciclo**: la tarea que pidió el usuario (verificar Ciclo 3)
no dio una respuesta limpia, pero el PROCESO de intentar verificarla con
rigor (A/B real, aislando una sola variable) encontró un bug más grave
que el que se estaba buscando. Vale la pena seguir haciendo estas
verificaciones aunque la pregunta original no se conteste del todo —
el valor no estaba solo en la respuesta, estaba en someter el código a
un caso real que el testing anterior no había cubierto.

## Bug real en `research_batch.py::get_heatmap` — heatmaps perdidos sin razón (2026-08-23)

Agregando 11 shorts nuevos de `@clips.virales00` al corpus (bajados en una
sesión anterior pero nunca registrados en el manifest — quedaron en disco
sin transcribir ni analizar, arreglado registrándolos con `status:
"descargado"` para que el daemon retome desde ahí sin re-bajarlos), noté en
el log del daemon: `"No supported JavaScript runtime could be found"`.

Causa: `get_heatmap()` en `research_batch.py` usaba `{"node": {}}` sin el
path explícito al Node 22 portátil — a diferencia de `download_video()`
(que sí pasa `YT_DLP_FIX_ARGS` completo) y de `clip_engine/heatmap.py` (que
ya tenía el fix correcto desde que se construyó esta noche). Sin el path
explícito, yt-dlp no encuentra un runtime JS v22+ (el del sistema es v20) y
el challenge de YouTube falla — el heatmap se guardaba como `None` sin
importar si el video realmente tenía uno.

**Verificado que el bug era real y no solo cosmético**, con un caso
controlado: `IqFk21f7lhk` (4.68 millones de vistas, un video MUY popular
que casi seguro tiene heatmap real) figuraba en el corpus con
`heatmap: None`. Re-consultado con la función ya arreglada:
**recuperó 100 puntos reales**. Confirma que el bug producía falsos
negativos reales, no solo ruido — algunos de los "sin heatmap" que
veníamos asumiendo como "pocas vistas" (interpretación documentada antes
para `KUiIY5I9FyA`) en realidad podían ser este bug mecánico.

Nota de honestidad sobre el proceso de verificación: probé primero con 2
videos de pocas vistas (~4000) que seguían dando `None` incluso con el fix
— por un momento pareció que el arreglo no cambiaba nada. Antes de asumir
eso, probé con un video de vistas altas a propósito (para separar "el
video genuinamente no tiene heatmap" de "el bug seguía rompiendo algo") —
ahí apareció la prueba real. Vale la pena repetir el hábito: un resultado
negativo con pocos casos chicos no alcanza para concluir nada, hace falta
un caso donde el resultado ESPERADO sea claramente distinto de null.

**Arreglado** en `research_batch.py::get_heatmap` (mismo patrón que
`clip_engine/heatmap.py`: `js_runtimes` con el path de `NODE22_PATH` +
`extractor_args` con `player_client=[android, tv]`). El daemon se reinició
para que los items nuevos usen la versión corregida.

**Pendiente, no ejecutado todavía por seguridad**: escribí
`research_heatmap_backfill.py` (script reproducible, re-intenta heatmap
para todo item con `heatmap=None` + URL de YouTube + `status` no
pending/error) para recuperar los falsos negativos ya existentes en el
resto del corpus (~1400 items candidatos). NO lo corrí todavía en paralelo
al daemon a propósito — el manifest ya se comió una pérdida real de ~1500
items una vez esta noche por escrituras concurrentes sin coordinar (ver
nota al principio de `research_batch.py`), y aunque el backfill relee el
archivo fresco antes de cada escritura (mismo patrón que `persist_item`),
dos procesos independientes escribiendo el mismo archivo sin un lock
compartido sigue teniendo una ventana real de choque. Correrlo recién
cuando el daemon esté inactivo (o integrarlo como un paso único al
arranque del daemon, protegido por el mismo lock) es la forma segura de
hacerlo — pendiente para la próxima vez que se retome el corpus.

**Actualización — backfill completo, resultado final (2026-08-23)**: corrió
de punta a punta sobre los 1423 items candidatos (pausado una vez a mitad
de camino para dejarle lugar al daemon de los 11 shorts nuevos, retomado
después sin problema — el script es idempotente, relee el manifest fresco
y saltea lo ya resuelto). **Recuperados en total: solo 3** (`IqFk21f7lhk`,
`LVeDT0N9rSQ`, `KD5JFnFZSPE` — 100 puntos cada uno). Heatmap real en la DB
pasó de 176 a 179 items.

**Honestidad sobre la magnitud real, no solo el caso dramático**: el bug
era 100% real y el fix está bien aplicado (verificado con el caso de
4.68M de vistas) — pero con volumen completo, el IMPACTO fue mucho más
chico de lo que ese único caso hacía pensar: 3 recuperados sobre 1423
re-intentados (~0.2%). La gran mayoría de los "sin heatmap" del corpus
son genuinamente así (pocas vistas u otro motivo real de YouTube), no
víctimas del bug. Vale la pena dejarlo dicho así, sin inflar el hallazgo:
el fix en `research_batch.py::get_heatmap` sigue siendo correcto y se
mantiene (los items nuevos que procese el daemon de acá en más van a
tener heatmap bien pedido), pero el backfill retroactivo no cambió
sustancialmente el tamaño del dataset utilizable — mismo patrón que la
sesión ya viene aplicando: un caso llamativo no es lo mismo que un
problema de escala, y solo el número con volumen real lo distingue.

## Tutorial guiado de onboarding (2026-08-23)

Pedido directo del usuario: un tutorial progresivo que aparezca solo, en
la primera visita de un usuario nuevo (no cada vez), explicando cada
herramienta de la interfaz paso a paso, con opción de saltarlo, usando un
video de ejemplo YA analizado (no que el usuario tenga que subir nada para
probar la app).

**Implementado en `webui/index.html`**: motor de tutorial en JS puro (sin
librería externa) — `TUTORIAL_STEPS` es un array de 13 pasos, cada uno con
un selector CSS real de la interfaz + título + explicación. Por cada paso:
resalta el elemento real (glow con box-shadow, técnica de "spotlight"),
posiciona una tarjeta flotante con la explicación + botones
Anterior/Siguiente/Saltar, hace scroll suave hasta el elemento. Si un
selector no existe en el momento (p. ej. no hay clips cargados todavía),
el paso se salta solo en vez de trabar el tutorial.

- Detección de primera visita: `localStorage.getItem('prende_tutorial_done_v1')`.
  Si no existe, arranca el tutorial solo al cargar la página. Se marca
  como visto al terminarlo O al saltarlo (ambos casos no debería volver a
  aparecer solo).
- Botón "🎓 Tutorial" agregado al header para volver a verlo cuando se
  quiera, sin tener que borrar el localStorage a mano.
- Video de ejemplo: se eligió `u1O6Av1vO-I` (streaming/IRL de Westcol en
  un pool party, ~14.6 min) — YA tiene transcripción + análisis visual
  cacheados de esta misma sesión, así que el análisis fresco para el
  tutorial no repite trabajo desde cero. Se le puso nombre visible
  "Video de ejemplo (tutorial)" vía `/api/set_video_name` para que no
  aparezca como un ID técnico en el selector.
  **Nota de honestidad**: es contenido real de YouTube ya usado como test
  esta sesión, no un video con licencia para demo pública — si esto sale
  a producción con usuarios reales, hay que reemplazarlo por contenido
  propio o con permiso explícito.
- `startTutorial()` fuerza categoría "streaming" y dispara un análisis
  completo (audio+imagen, `target_clips=10`) sobre ese video para tener un
  set de candidatos limpio y variado, en vez de reusar el amontonado de
  corridas de test de toda la noche (que tenía muy pocos `keep=true`).

**Bug real encontrado y arreglado en el propio CSS del tutorial**: la
primera versión tenía DOS capas de oscurecido superpuestas — un
`.tutorial-overlay` de fondo semi-transparente Y el truco de
`box-shadow: 0 0 0 9999px` en `.tutorial-highlight` (que por sí solo ya
oscurece todo menos el elemento resaltado). Con las dos juntas, el
elemento resaltado quedaba per debajo del overlay de fondo también,
así que se veía atenuado en vez de brillante — exactamente al revés de lo
que un spotlight tiene que hacer. Arreglado dejando `.tutorial-overlay`
transparente (solo sirve para bloquear clicks sobre la página de atrás
mientras el tutorial está activo) y el box-shadow del highlight como
única fuente real de oscurecido.

**Limitación de testing, dicha explícita**: este entorno (Windows, sin
`chromium-cli` ni Playwright disponible) no tiene forma de abrir un
navegador real y verificar visualmente que el spotlight/tooltip se ve y
se comporta bien. Se verificó: sintaxis JS válida (`node --check` sobre el
script extraído), que todos los selectores de `TUTORIAL_STEPS` existen de
verdad en el HTML (chequeado uno por uno contra el markup real), y
revisión manual línea por línea de la lógica de posicionamiento/skip. NO
se verificó visualmente en un navegador real — si algo se ve mal
(posición de la tarjeta, timing del scroll), habría que probarlo a mano.

**Estado al momento de escribir esto**: el análisis fresco del video de
ejemplo (`u1O6Av1vO-I`, categoría streaming, completo) está corriendo en
background — el código del tutorial está listo y no depende de que
termine para funcionar (ya carga bien contra CUALQUIER candidatos.json
existente), pero para que un usuario nuevo vea clips reales sin esperar
hace falta que esta corrida puntual termine y quede cacheada. Se dejó un
monitor en background para avisar cuando termine.

## Primer test real del motor sobre podcast — pedido explícito del usuario (2026-08-23)

El usuario pidió expandir el corpus hacia podcast/radio (LuzuTV, OLGA,
Blender, Wild Project, y ~40 canales más de Argentina/LatAm/España/
internacional, con los internacionales en inglés etiquetados aparte para
no contaminar las estadísticas LatAm ya calibradas) y, con justa razón,
señaló que juntar transcripciones/energía/visual del corpus no es lo mismo
que probar si el motor de selección REAL funciona bien en este género —
eso solo se sabe corriendo `analyze()` de verdad sobre un episodio y
mirando los clips uno por uno.

**Primer test real**: episodio de Podium Podcast (entrevista a Sofía
Vergara y Vicky Martín Berrocal, ~31 min), categoría "general" (que ya
asume podcast/entrevista), modo completo (audio+imagen).

**Bug real encontrado — el candidato con MÁS puntaje de toda la corrida
duraba el doble del máximo permitido**: "Griselda Blanco: ¿Más peligrosa
que Escobar?" quedó con score 9 (el más alto) y `keep=true`, pero medía
71.3 segundos — el doble de `max_clip_seconds` (35s). Causa exacta:
el umbral de protección de ancla del Ciclo 4 de esta madrugada
(`anchor_segment_count <= 20`, elegido a ojo esa noche, nunca calibrado
contra habla lenta) resultó demasiado generoso para podcast — un rango de
19 segmentos ahí es diálogo real y continuo, no un caso patológico, pero
en un podcast con habla pausada esos 19 segmentos son 71 segundos reales
(vs. streaming, donde 19 segmentos de cruces rápidos duran mucho menos).
El umbral bloqueaba el recorte a `hard_max` pensando que protegía un caso
atómico, cuando en realidad había de sobra para recortar.

**Arreglado**: el umbral pasó de "≤20 segmentos" a "≤1 segmento" — la
protección ahora aplica SOLO al caso que la motivó de verdad (un único
segmento roto de Whisper, sin nada más chico para devolver). Cualquier
rango de 2+ segmentos con contenido real ya tiene de dónde recortar, así
que dejarlo sin proteger es lo correcto. Verificado con los 5 casos juntos
en la misma corrida:
- Griselda Blanco (podcast, 19 segmentos) → antes 71.3s, ahora 42.6s.
- Rango de 244 segmentos (Ciclo 4, streaming) → sigue en 39.5s, sin regresión.
- Segmento roto de un solo índice (Ciclo 1, streaming) → sigue devolviendo
  el segmento completo (51.66s), sin regresión.
- Colesterol (`panchos_test`) → sigue en ~10.8s.
- Festejo (`Pah7u3Ja-lY`) → ahora 32.5s (antes 38.4s), sigue llegando al
  remate real ("la concha de la lora... ¡sí, señor!").

**Lección del hallazgo**: un umbral numérico elegido "a ojo" para un
síntoma real (el de streaming) puede fallar silenciosamente en otro
género con un ritmo de habla distinto — exactamente el tipo de cosa que
solo aparece corriendo el motor sobre contenido real del género nuevo,
no extrapolando desde estadísticas de corpus agregadas. Confirma que el
paso "correr sobre un video real y mirar los clips uno por uno" no es
opcional ni reemplazable por juntar más datos de corpus.

**Segundo hallazgo, sobre el mismo test — validación pendiente de acción,
no bug de código**: de 13 candidatos forzados por pico de audio, CERO
sobrevivieron el ranking (todos rechazados por "confuso, sin remate
claro"). De 14 candidatos forzados por momento visual, CERO sobrevivieron
tampoco (todos rechazados por "descripción genérica, sin gesto
concreto") — el video es una entrevista de dos personas sentadas
hablando, sin acción visual dinámica que describir con especificidad, muy
distinto del streaming/gaming donde SÍ hay gestos/reacciones puntuales
que nombrar. Los candidatos que SÍ sobrevivieron (10 de 40) fueron todos
generados directamente por el LLM a partir del texto — temas como el dato
sobre Griselda Blanco, autoironía sobre la fama, reflexiones sobre el
envejecimiento y el amor — cualitativamente razonables como clips reales
de este tipo de entrevista.

Interpretación honesta, no una conclusión cerrada todavía (un solo video
no es volumen): para el formato de entrevista/podcast sentado, el
forzado por audio y por imagen podría estar gastando crédito real (625
frames escaneados con gpt-4o-mini en este video solo) sin aportar ningún
candidato final — a diferencia de streaming, donde SÍ se verificaron
casos reales de candidatos forzados que terminaron recomendados. Antes de
tocar código (por ejemplo, desactivar el forzado visual para categoría
podcast/general), hace falta repetir este mismo test en 2-3 episodios más
para confirmar que no es un caso aislado de ESTE video puntual.
