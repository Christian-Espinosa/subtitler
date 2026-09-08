# Subtitler — manual

Herramienta **local** (en tu PC, con GPU) para generar subtítulos **SRT** listos para YouTube / Bilibili.

Sirve sobre todo para vídeos donde se habla **español, inglés y/o chino mezclados**. El resultado por defecto es:

- un maestro en **inglés** (`subs.en.srt`)
- una pista en **castellano** (`subs.es.srt`)

No sube nada a internet para transcribir: los modelos se descargan una vez y luego se usan desde caché.

---

## 1. Glosario (términos que verás en logs y código)

| Término | Qué es, en este proyecto |
|---------|--------------------------|
| **SRT** | Formato de subtítulos: número, reloj de inicio/fin, texto. Es el único formato que genera este repo. |
| **Cue** | Un bloque del SRT: “desde 00:00:03 hasta 00:00:08, este texto”. |
| **ASR** | *Automatic Speech Recognition*: pasar audio a texto. Aquí lo hace **Whisper** vía **WhisperX**. |
| **Whisper / WhisperX** | Modelo de OpenAI (Whisper) + wrapper WhisperX (alineado de tiempos, VAD, CLI). El runtime instalado es `python -m whisperx`, no hace falta usar la carpeta `whisperX/` del repo. |
| **`--task transcribe`** | Escribe en el **idioma del habla** (o el que fuerces con `--language`). |
| **`--task translate`** | Whisper **solo traduce a inglés**. Aunque el audio sea español o chino, el SRT sale en inglés. Es intencional: las 3 pasadas se comparan en el mismo idioma. |
| **`--language es\|en\|zh`** | **Asume** que todo el audio es ese idioma. Whisper no cambia de idioma a mitad de vídeo. Por eso hacemos 3 pasadas. |
| **Pasada / pass** | Una ejecución de WhisperX con un idioma forzado. Archivos intermedios: `[nombre]._pass_en_from_es.srt` (etc.). |
| **Hipótesis** | Cada pasada propone un texto inglés distinto para el mismo tramo. El árbitro elige una. |
| **Árbitro** | Script `arbitrate_en_srt.py`: compara las 3 hipótesis y se queda con la más “inglés natural”. |
| **Perplejidad (PPL)** | Número: “qué tan sorprendido está un modelo de lenguaje inglés” ante un texto. **Más bajo = más fluido**. No mide si el contenido es fiel al audio. |
| **`distilgpt2`** | Mini GPT-2 usado solo para calcular esa perplejidad. No genera subtítulos. |
| **Ventana / probe** | Trozo de tiempo (por defecto 3 s) donde el árbitro mira si hay un cue cuyo **centro (midpoint)** cae ahí. |
| **Midpoint** | Mitad temporal de un cue. Evita re-puntuar el mismo párrafo largo en cada ventana de 3 s. |
| **Resegment** | Partir cues densos (p. ej. 20 s de texto) en cues cortos que se puedan leer en pantalla. |
| **CPS** | *Characters per second*: velocidad de lectura. ~18 es un tope típico de YouTube. |
| **NLLB** | *No Language Left Behind* (Meta): traductor EN → ES o ZH. **No** escucha el audio; traduce el SRT inglés. |
| **Hugging Face (HF)** | Sitio de modelos. Caché típica: `%USERPROFILE%\.cache\huggingface\hub\`. |
| **`HF_ENDPOINT`** | URL alternativa del hub. En China / bloqueos: `https://hf-mirror.com`. |
| **CUDA / `float16`** | Cálculo en GPU NVIDIA. `float16` ahorra VRAM (RTX 3060 Ti). |
| **`batch_size`** | Cuántos trozos de audio procesa Whisper a la vez. Bájalo (p. ej. 4) si te quedas sin VRAM. |
| **VAD** | *Voice Activity Detection*: detectar dónde hay voz. WhisperX usa Pyannote. |
| **FFmpeg / ffprobe** | Extraen audio del vídeo y duraciones. Deben estar en el PATH de Windows. |
| **torchcodec / triton** | Avisos frecuentes en Windows. **No paran** el pipeline si el SRT se genera. |
| **Diarización** | Separar hablantes. **No** está activada (hace falta token HF). |

---

## 2. Cómo ejecutar

### Pipeline alineado (recomendado desde 2026-09)

Doble clic en `subtitular_alineado.bat` (pide la ruta) o:

```bat
subtitular_alineado.bat "D:\Videos\mi_video.mp4"
subtitular_alineado.bat "video.mp4" --language es --langs es,en
```

Transcribe en el idioma hablado, **alinea palabra por palabra** y traduce después. Es el que arregla los subtítulos que se adelantaban y los bloques con demasiado texto: ver [§3-bis](#3-bis-por-qué-el-pipeline-alineado-medido). Tarda **~1 minuto por cada 20 de vídeo**, frente a los ~25 del pipeline de 3 pasadas.

**Salida:** `[nombre] subs.es.srt` (idioma hablado) y `[nombre] subs.en.srt` (traducción).

### Pipeline de 3 pasadas + árbitro (heredado)

Sigue disponible, pero **no se recomienda**: mide peor en todo (§3-bis).

```bat
subtitular_en_master.bat "D:\Videos\mi_video.mp4"
```

Las comillas se quitan solas si pegas la ruta desde el Explorador.

**Salida, misma carpeta que el vídeo:**

| Archivo | Qué es |
|---------|--------|
| `[nombre] subs.en.srt` | Maestro inglés, ya resegmentado para pantalla |
| `[nombre] subs.es.srt` | Castellano (traducción NLLB del inglés) |
| `[nombre]._arbiter.json` | Depuración: qué pasada ganó cada cue |

Los pases `._pass_en_from_*.srt` se **borran** al terminar. Para conservarlos: `--keep-passes`.

### Modo simple (1 sola pasada Whisper)

Si el vídeo es **casi todo un idioma** y no hace falta árbitro:

```bat
subtitular.bat "D:\Videos\mi_video.mp4"
subtitular.bat "video.mp4" --language es
```

Salida: `[nombre] subs.srt`.

### Flags del pipeline (se pasan al `.bat`)

```bat
subtitular_en_master.bat "video.mp4" --langs en,es,zh
subtitular_en_master.bat "video.mp4" --keep-passes
subtitular_en_master.bat "video.mp4" --skip-resegment
subtitular_en_master.bat "video.mp4" --skip-nllb
subtitular_en_master.bat "video.mp4" --window 3
subtitular_en_master.bat "video.mp4" --skip-whisperx
```

`--skip-whisperx` reutiliza los `._pass_*.srt` que ya existan (hace falta `--keep-passes` en una corrida anterior).

---

## 3. Flujo del pipeline (paso a paso)

El orquestador es `pipeline_en_master.py`. Lo lanza `subtitular_en_master.bat`.

```text
vídeo / audio
        │
        ▼
 ┌─────────────────────────────────────────────────────────┐
 │ PASOS 1–3  WhisperX × 3                                 │
 │   cada uno: --task translate --language es | en | zh    │
 │   los 3 SRT están en INGLÉS                             │
 │   archivos: [stem]._pass_en_from_es.srt  (etc.)         │
 └─────────────────────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────────────────────┐
 │ PASO 4  Árbitro (distilgpt2)                            │
 │   elige 1 hipótesis por cue (menor PPL)                 │
 │   escribe [stem] subs.en.srt  (cues aún largos)         │
 │   escribe [stem]._arbiter.json                          │
 │   borra los _pass_*  (salvo --keep-passes)              │
 └─────────────────────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────────────────────┐
 │ PASO 5  Resegment                                       │
 │   parte frases, 2 líneas, ~42 chars, ~18 CPS, 1–6 s     │
 │   reescribe el mismo subs.en.srt                        │
 └─────────────────────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────────────────────┐
 │ PASO 6  NLLB                                            │
 │   traduce cada cue EN → ES (mismos tiempos)             │
 │   escribe [stem] subs.es.srt                            │
 └─────────────────────────────────────────────────────────┘
```

**Por qué 3 pasadas.** Whisper **fija un idioma para todo el archivo**. Si fuerzas `es` y de pronto hablas chino, esa pasada suele salir mal. La pasada cuyo idioma coincide con lo hablado produce inglés coherente; las otras, basura o calcos. El árbitro elige tramo a tramo.

> Esta idea tiene un fallo que solo se ve midiendo: el árbitro premia el inglés más *fluido*, no el más *fiel*, y una alucinación siempre suena fluida. Ver §3-bis.

**Qué no hace el pipeline.** No corrige errores de ASR (“breakfast” vs “ayuno”). El árbitro solo elige entre tres textos. NLLB solo traduce el inglés ya elegido.

En consola verás `PASO x/y`, tiempo transcurrido y un **ETA** aproximado (no es exacto; la primera vez con descarga es más lenta).

---

## 3-bis. Por qué el pipeline alineado (medido)

Diagnóstico hecho el 2026-09-04 sobre `Microfono 1.0.mp4` (23 min, español), comparando la salida real del pipeline de 3 pasadas contra la del alineado.

### La causa de casi todo: `--task translate` desactiva la alineación

En `whisperX/whisperx/transcribe.py`:

```python
if task == "translate":
    # translation cannot be aligned
    no_align = True
```

El pipeline de 3 pasadas usa `--task translate` en las tres. Es decir, **desactiva el alineado forzado**, que es justamente lo que WhisperX añade sobre Whisper. Consecuencias en cadena:

1. Sin alineado, WhisperX emite **un cue por cada bloque VAD de 30 s** (`chunk_size=30` en `asr.py`). Medido: los cues que el árbitro eligió tenían una **mediana de 23,6 s**.
2. El árbitro compara párrafos enteros de 24 s con perplejidad, y al ganar uno salta las ventanas siguientes (`covered_until`). En este vídeo eso **saltó 15,4 minutos** de decisiones.
3. `resegment_srt.py` recibe bloques de 24 s con solo un inicio y un final, así que **estima** dónde va cada frase contando caracteres. De los 309 cues finales, solo unos 50 tenían límites tomados del audio: el resto eran cuentas.

Por eso los subtítulos se adelantaban dentro de un párrafo y volvían a cuadrar en el siguiente.

### El árbitro premia fluidez, no fidelidad

En un vídeo **100% en español**, la pasada que más ganó fue la del **chino: 64%** (32 de 50 decisiones), frente al 18% de la española. Y el caso extremo:

```text
269
00:17:17,303 --> 00:17:21,110
Leading Starring Qin Liying,
```

Créditos de televisión china inventados sobre un tramo **sin voz**, que se quedaron en pantalla mientras nadie hablaba. Whisper alucina cuando se le fuerza un idioma que no oye, y `distilgpt2` puntúa esa alucinación como buen inglés, porque lo es.

### Números, antes y después

Pista inglesa, mismo vídeo:

| Métrica | 3 pasadas + árbitro | Alineado | |
|---|---|---|---|
| Cue más largo | 10,75 s | **5,98 s** | el límite era 6 s |
| Cues > 6 s | 7,1% (15% del tiempo) | **0%** | |
| Líneas > 42 caracteres | 9,3% | **0,6%** | |
| Cues que empiezan **en silencio** | 26,5% | **0%** | |
| Subtítulos en pantalla | 17,6 min | **15,1 min** | hay 14,9 min de voz |
| Tiempo de proceso | ~25 min | **1 min 10 s** | |

Dos avisos honestos:

- Los **0%** del alineado son coherencia interna: los cues se construyen desde la misma alineación con la que se mide. Lo que sí es independiente es el 26,5% del antiguo, y que sacaba 17,6 min de subtítulos para 14,9 min de voz.
- El alineado tiene **más cues densos** (13% por encima de 18 CPS, frente al 2%). No es un empeoramiento real: el antiguo bajaba el CPS dejando el texto en pantalla durante los silencios. Aquí el cue dura lo que dura la frase más 0,6 s de cortesía.

### Flujo nuevo

```text
vídeo → ffmpeg (16 kHz mono)
      → WhisperX  --task transcribe  (alineación ON)  → JSON con tiempos por palabra
      → build_subs.py    corta en fronteras reales de palabra
      → translate_srt.py NLLB, traduce el texto y NO toca los tiempos
```

La traducción va al final a propósito: así el idioma de salida no condiciona la sincronía. Lo que sí cambia es el **estilo**: Whisper traduciendo escribe un inglés más natural que NLLB, que es más literal ("Muy buenas gente" → "Very good people"). A cambio, lo que dice es lo que se dijo: donde el antiguo puso *"Go."*, el audio decía *"¡Fuego!"*.

---

## 4. Qué hace cada archivo

### `subtitular_en_master.bat`

Punto de entrada. Comprueba Python 3.12, pide o recibe la ruta, quita comillas, llama a `pipeline_en_master.py`.

| Variable | Default | Para qué |
|----------|---------|----------|
| `PY312` | ruta a `python.exe` 3.12 | Si cambias de instalación de Python |
| `PIPE` | `pipeline_en_master.py` junto al `.bat` | No hace falta tocarla |
| `EXTRA_ARGS` | vacío | Flags **siempre** añadidos (ej. `--langs en,es,zh`) |
| `HF_ENDPOINT` | `https://hf-mirror.com` | Espejo HF; vacío = hub oficial |

### `subtitular.bat`

Una sola llamada a WhisperX (`transcribe` implícito, no translate). Renombra `[stem].srt` → `[stem] subs.srt`.

| Variable | Default | Para qué |
|----------|---------|----------|
| `PY312` | Python 3.12 | Igual que arriba |
| `WHISPER_MODEL` | `large-v2` | Modelo ASR |
| `BATCH_SIZE` | `8` | VRAM; bajar si OOM |
| `COMPUTE_TYPE` | `float16` | Precisión GPU |
| `DEFAULT_LANGUAGE` | vacío | Auto-detect; o `es` / `en` / `zh` |
| `HF_ENDPOINT` | espejo | Igual que arriba |

### `pipeline_en_master.py`

Orquesta los 4 programas. **Aquí está el CONFIG principal** si quieres cambiar el comportamiento por defecto.

| Variable | Default | Significado |
|----------|---------|-------------|
| `PYTHON` | Python 3.12 | Interprete para WhisperX y los scripts hijos |
| `HF_ENDPOINT` | `https://hf-mirror.com` | Espejo Hugging Face |
| `WHISPER_MODEL` | `large-v2` | Tamaño del ASR (`large-v3` es más pesado) |
| `WHISPER_BATCH_SIZE` | `8` | Lotes en GPU |
| `WHISPER_COMPUTE_TYPE` | `float16` | Tipo de cálculo |
| `WHISPER_PASS_LANGS` | `es, en, zh` | Idiomas de audio forzados (3 hipótesis) |
| `WHISPER_ETA_RTF` | `0.30` | Solo ETA: 0.30 = ~30 % de la duración del vídeo por pasada |
| `ARBITER_WINDOW_SEC` | `3.0` | Tamaño del probe del árbitro (s) |
| `FLUENCY_MODEL` | `distilgpt2` | LM de perplejidad |
| `ARBITER_ETA_SEC_PER_MIN` | `8.0` | Solo ETA |
| `OUTPUT_SUB_LANGS` | `en, es` | Pistas finales. Añade `"zh"` para chino |
| `RESEGMENT` | `True` | Si `False`, no parte cues para pantalla |
| `RESEGMENT_MAX_CHARS` | `42` | Máx. caracteres por línea |
| `RESEGMENT_MAX_LINES` | `2` | Máx. líneas por cue |
| `RESEGMENT_MAX_CPS` | `18.0` | Tope de lectura |
| `RESEGMENT_MIN_SEC` | `1.0` | Duración mínima de un cue |
| `RESEGMENT_MAX_SEC` | `6.0` | Duración máxima objetivo |
| `NLLB_MODEL` | `facebook/nllb-200-distilled-600M` | Traductor |
| `NLLB_ETA_SEC_PER_MIN` | `12.0` | Solo ETA |
| `KEEP_PASSES` | `False` | Conservar `_pass_*.srt` |

Los flags CLI (`--langs`, `--window`, …) **pisan** estos defaults en esa ejecución.

### `pipeline_aligned.py` + `subtitular_alineado.bat`

Orquestador del flujo recomendado. **Aquí está el CONFIG** del pipeline nuevo.

| Variable | Default | Significado |
|----------|---------|-------------|
| `PYTHON` | Python 3.12 | Intérprete |
| `WHISPER_MODEL` | `large-v2` | ASR |
| `WHISPER_BATCH_SIZE` | `8` | Bajar a 4 si falta VRAM |
| `SPOKEN_LANGUAGE` | vacío | Idioma hablado; vacío = detectar |
| `OUTPUT_SUB_LANGS` | `es, en` | Pistas a escribir |
| `MAX_CHARS_PER_LINE` / `MAX_LINES` | `42` / `2` | Límites de pantalla |
| `MAX_CUE_SEC` / `MIN_CUE_SEC` | `6.0` / `1.0` | Duración de cue |
| `KEEP_WORK` | `False` | Conservar `_work/` (audio + JSON alineado) |

Flags útiles: `--language es`, `--langs es,en,zh`, `--skip-translate`, `--keep-work`, y `--skip-whisperx` para rehacer los subtítulos sin volver a transcribir (necesita `--keep-work` de una corrida anterior).

### `build_subs.py`

Convierte la alineación por palabra en cues legibles. Es el sustituto de `resegment_srt.py` en el flujo alineado, y la diferencia es de fondo: aquí cada palabra trae su timestamp, así que los cortes se **miden** en vez de estimarse.

Orden de preferencia para cerrar un cue: final de frase → coma o punto y coma → el silencio más largo dentro del cue → el límite de caracteres.

```bat
python build_subs.py --align _work/video.json --out "video subs.es.srt"
```

No lleva regla de CPS a propósito: partir un cue no cambia la velocidad a la que habla nadie. La densidad se compensa dejando el cue en pantalla 0,6 s tras la última palabra, si el silencio lo permite.

### `analyze_srt.py`

Mide un SRT: duraciones, líneas largas, CPS, solapes y, si le pasas el `_arbiter.json`, cómo repartió las victorias el árbitro.

```bat
python analyze_srt.py --in "video subs.en.srt" --compare "otro.srt"
python analyze_srt.py --in "video subs.en.srt" --arbiter "video._arbiter.json"
```

### `measure_drift.py`

Comprueba si un SRT está sincronizado, usando la alineación por palabra como referencia. Dice cuántos cues aparecen **antes de que alguien hable** y lista los peores.

```bat
python measure_drift.py --srt "video subs.en.srt" --align _work/video.json
```

### `arbitrate_en_srt.py`

Lee 3 SRT en inglés. Recorre el tiempo en ventanas. Si el **midpoint** de un cue cae en la ventana, puntúa las 3 hipótesis con `distilgpt2` y gana la de **menor PPL**. Escribe el SRT maestro con los **tiempos originales de Whisper** (cues largos). No vuelve a decidir hasta que ese cue termina.

Uso suelto:

```bat
python arbitrate_en_srt.py --es from_es.srt --en from_en.srt --zh from_zh.srt --out "video subs.en.srt" --json decisions.json
```

| Variable | Default | Significado |
|----------|---------|-------------|
| `DEFAULT_WINDOW_SEC` | `3.0` | Probe |
| `DEFAULT_FLUENCY_MODEL` | `distilgpt2` | Prefiere snapshot local en caché HF |

El JSON incluye `probe_start` / `probe_end` (ventana), `start` / `end` (cue ganador), `winner`, `scores`, `candidates`.

### `resegment_srt.py`

Parte el maestro inglés para que se pueda leer: frases, wrap a 2 líneas, huecos mínimos entre cues.

```bat
python resegment_srt.py --in "video subs.en.srt"
```

| Variable | Default | Significado |
|----------|---------|-------------|
| `MAX_CHARS_PER_LINE` | `42` | Ancho de línea |
| `MAX_LINES` | `2` | Líneas por cue |
| `MAX_CPS` | `18.0` | Velocidad de lectura |
| `MIN_CUE_SEC` | `1.0` | No más cortos que esto (salvo necesidad) |
| `MAX_CUE_SEC` | `6.0` | Intenta no pasar de esto |
| `MIN_GAP_SEC` | `0.08` | Hueco entre cues adyacentes |

**No** lo uses sobre el ES después de NLLB: partiría tiempos y desincronizaría EN/ES. El ES solo hace *wrap* de líneas (ver abajo).

### `translate_srt.py`

NLLB: cada cue inglés → español (y/o chino). **Mismos timestamps** que el EN. Luego envuelve líneas (`WRAP_*`) sin cambiar tiempos.

```bat
python translate_srt.py --in "video subs.en.srt" --langs es
python translate_srt.py --in "video subs.en.srt" --langs es,zh
```

Si corriges el EN a mano, vuelve a lanzar **solo** este script para regenerar ES.

| Variable | Default | Significado |
|----------|---------|-------------|
| `NLLB_MODEL` | `nllb-200-distilled-600M` | Modelo de traducción |
| `DEFAULT_TARGET_LANGS` | `es` | Si no pasas `--langs` |
| `BATCH_SIZE` | `16` | Cues por lote GPU |
| `MAX_LENGTH` | `256` | Tokens máximos generados |
| `DEVICE` | auto CUDA | `cuda` o `cpu` |
| `WRAP_MAX_CHARS` | `42` | Wrap visual ES/ZH |
| `WRAP_MAX_LINES` | `2` | Líneas |

Códigos NLLB internos: `en` → `eng_Latn`, `es` → `spa_Latn`, `zh` → `zho_Hans` (simplificado).

### `safe_downloader/`

Plantilla para descargar modelos grandes con watchdog (red inestable / China). Ver [`safe_downloader/README.md`](safe_downloader/README.md).

Scripts típicos: `install_distilgpt2.ps1`, `install_nllb.ps1`, `install_faster_whisper_large_v2.ps1`.

### `whisperX/`

Clon de referencia del código de WhisperX. **El que se ejecuta** es el paquete `pip` (`python -m whisperx`). No hace falta tocar esta carpeta para subtitular.

---

## 5. Requisitos

| Pieza | Notas |
|-------|--------|
| Python **3.12** | WhisperX no soporta 3.14 |
| `whisperx` + PyTorch **CUDA** | En ese Python |
| FFmpeg y ffprobe | En el PATH |
| GPU | Probado en RTX 3060 Ti |
| Modelos | ASR `large-v2`, `distilgpt2`, NLLB-600M (caché HF) |

Si `huggingface.co` no conecta, deja `HF_ENDPOINT=https://hf-mirror.com` (ya está en los `.bat` y en el pipeline).

---

## 6. Layout del repo

```text
subtitler/
  subtitular_alineado.bat     entrada recomendada
  pipeline_aligned.py         orquestador del flujo alineado
  build_subs.py               alineación por palabra → cues
  analyze_srt.py              mide legibilidad de un SRT
  measure_drift.py            mide sincronía contra la alineación
  translate_srt.py            NLLB, --src cualquiera → ES/EN/ZH

  subtitular_en_master.bat    entrada del pipeline heredado
  subtitular.bat              1 pasada Whisper
  pipeline_en_master.py       orquestador heredado
  arbitrate_en_srt.py         elige entre 3 SRT EN
  resegment_srt.py            cues legibles (flujo heredado)

  safe_downloader/            descargas de modelos
  whisperX/                   clon de referencia
  README.md                   este manual
```

No dejes `_test_*` ni `.log` en la raíz (están en `.gitignore`).

---

## 7. Problemas conocidos (calidad)

Usable como **borrador**. Revisa antes de publicar.

### Del pipeline alineado (el recomendado)

**Un idioma por vídeo.** WhisperX alinea con un modelo por idioma, así que el flujo alineado asume que el vídeo es mayormente de un idioma. Para vídeos de verdad mezclados haría falta detectar el idioma por segmento y alinear cada grupo con su modelo; **no está hecho**.

**NLLB traduce literal.** “Muy buenas gente” → “Very good people”. El inglés que escribía Whisper traduciendo sonaba mejor, aunque no siempre dijera lo que se dijo. Si quieres las dos cosas, la traducción con un LLM (DeepSeek) es el siguiente paso natural.

**Cues densos.** ~13% pasan de 18 CPS cuando se habla rápido. Es real: el cue dura lo que dura la frase. Se puede aflojar subiendo `LEAD_OUT_SEC` en `build_subs.py`.

**ASR.** Sigue confundiendo nombres propios (“Ches” → “Chess”). Eso se arregla editando el SRT del idioma hablado y relanzando solo `translate_srt.py`.

### Del pipeline heredado (3 pasadas)

**Árbitro.** No mide fidelidad, mide fluidez. Medido en un vídeo en español: eligió la pasada **china el 64%** de las veces, e incluyó créditos de TV china inventados sobre tramos sin voz (§3-bis).

**Resegment.** La 2ª línea a veces supera 42 caracteres. Algunos cues densos superan 6 s. Es inevitable: parte bloques de 24 s estimando por caracteres.

**Sincronía.** El 26,5% de los cues aparecían sin que nadie estuviera hablando.

**Avisos en consola.** `triton not found`, `torchcodec` / FFmpeg DLL, Lightning `upgrade_checkpoint`, TF32: **ruido** si el progreso de Whisper sigue y al final hay SRT.

**Flujo de revisión recomendado**

1. Pipeline completo.  
2. Editar `subs.en.srt` (términos clave).  
3. Regenerar ES: `python translate_srt.py --in "... subs.en.srt" --langs es`.  
4. Subir a YouTube / Bilibili.

---

## 8. Cómo modificarlo más adelante (atajos)

| Quiero… | Dónde |
|---------|--------|
| También chino en salida | `--langs es,en,zh` |
| Solo el idioma hablado | `--skip-translate` |
| Rehacer subtítulos sin transcribir otra vez | `--keep-work` una vez, luego `--skip-whisperx` |
| Cues más largos o más cortos | `--max-sec` / `--max-chars`, o el CONFIG de `build_subs.py` |
| Que el texto aguante más en pantalla | `LEAD_OUT_SEC` en `build_subs.py` |
| Saber si un SRT está bien | `analyze_srt.py` y `measure_drift.py` |
| Subtítulos más cortos/largos (flujo heredado) | `RESEGMENT_*` en el pipeline (o `resegment_srt.py`) |
| Conservar las 3 pasadas Whisper | `KEEP_PASSES = True` o `--keep-passes` |
| OOM / falta de VRAM | `WHISPER_BATCH_SIZE = 4` |
| Python en otra ruta | `PY312` en los `.bat` **y** `PYTHON` en `pipeline_en_master.py` |
| Hub oficial HF | `HF_ENDPOINT=` vacío en `.bat` y pipeline |
| Re-traducir tras editar EN | solo `translate_srt.py` |
| Re-arbitrar sin Whisper | `--keep-passes` una vez, luego `--skip-whisperx` |

Los defaults se editan en el bloque `# CONFIG` al **principio** de cada archivo (no hace falta buscar por el código).
