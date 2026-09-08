# safe_downloader

Mini-guía + plantilla para que una IA (o tú) descargue **paquetes grandes / flaky** en Windows **sin dejar el PC colgado** y **sin duplicar procesos**.

Úsala cuando digas: *“mira `safe_downloader/` y aplica el watchdog”*.

---

## Problema que resuelve

Descargas tipo PyTorch CUDA (~2–3 GB), wheels, modelos HF, etc. desde redes inestables (p. ej. China sin VPN):

- `pip` se corta o se queda a 200 KB/s
- Varias terminales / agentes lanzan **varios `pip` a la vez** y se pisan
- Tras un corte, hay que **reanudar**, no empezar de cero
- Hace falta un proceso que **revise solo** cada N minutos y retome si hace falta

---

## Principios (obligatorios)

1. **Un solo instalador a la vez** — matar duplicados antes de arrancar.
2. **Reanudación** — preferir `curl -C -` para archivos grandes; `pip` con `--retries` y timeout alto.
3. **Watchdog autónomo** — bucle cada **N minutos** (default **5**); no pide confirmación.
4. **Log siempre** — timestamp + qué hizo; así se audita a la mañana siguiente.
5. **Lock file** — evita dos watchdogs en paralelo.
6. **Criterio de éxito explícito** — p. ej. `import torch` + `cuda`, o tamaño de archivo, o `DONE OK` en el log.
7. **PATH / Python correcto** — en este PC: Python **3.12** (no 3.14 si el paquete no lo soporta).
8. **Espejos** — si PyPI va mal (China), probar Tsinghua / Aliyun **después** (o primero) según el caso; **no** sustituir wheels CUDA de `download.pytorch.org` por CPU de un mirror.

---

## Cómo lo hicimos en Subtitler (precedente real)

Contexto: instalar WhisperX + PyTorch CUDA en Windows, red lenta/cortes.

| Paso | Qué se hizo |
|------|-------------|
| 1 | Instalar Python 3.12 y ponerlo **delante** en el User PATH |
| 2 | Clonar / usar el proyecto; FFmpeg ya estaba |
| 3 | Descargar wheels CUDA con **`curl -L -C - --retry …`** a una carpeta `.wheels/` (nombres PEP 427 válidos para pip) |
| 4 | `pip install` de wheels locales, luego `whisperx` (espejos CN si pypi.org cuelga) |
| 5 | **Cuidado:** `pip install whisperx` puede **pisar** torch CUDA con torch CPU → **force-reinstall** de wheels CUDA `--no-deps` al final |
| 6 | Script `watch_install.ps1`: cada 5 min comprueba tamaño/procesos; si stall o muerto, reanuda; escribe `watch_install.log` + `install_whisperx.log`; termina con `=== DONE OK ===` |

Lecciones:

- No lanzar 2–3 `pip install torch` en paralelo (pasó; se mataron y se unificó en un watchdog).
- `Tee-Object` en PowerShell puede liar encoding del log; mejor `Add-Content -Encoding utf8`.
- Nombres tipo `torch_cu128.whl` **fallan** en pip → usar `torch-2.11.0+cu128-cp312-cp312-win_amd64.whl`.
- El watchdog debe sobrevivir aunque Cursor se desconecte (`Start-Process -WindowStyle Hidden` o tarea en background).

---

## Uso rápido (plantilla)

```powershell
# Desde la raíz del proyecto que esté descargando, o pasando -WorkDir
powershell -NoProfile -ExecutionPolicy Bypass -File .\safe_downloader\watch_download.ps1 `
  -IntervalMinutes 5 `
  -WorkDir "C:\ruta\al\proyecto" `
  -SuccessProbe { ... }   # o usar -SuccessCommand
```

Parámetros típicos:

| Parámetro | Default | Significado |
|-----------|---------|-------------|
| `-IntervalMinutes` | **5** | Cada cuántos minutos revisa / reintenta |
| `-WorkDir` | carpeta actual | Dónde escribir logs y lock |
| `-LogName` | `safe_download.log` | Log principal |
| `-LockName` | `.safe_download.lock` | Lock |
| `-InstallScript` | (obligatorio en uso real) | Script o bloque que **hace** la descarga/install (idempotente / resumible) |
| `-SuccessCommand` | — | Comando que exit 0 = terminado (ej. `python -c "import whisperx"`) |

Comportamiento del bucle:

1. Si `-SuccessCommand` OK → escribe `=== DONE OK ===` y **sale**.
2. Si hay lock fresco (< Interval) → no lanza otro install.
3. Si no hay proceso de install / curl / pip relevante, o el progreso (tamaño de archivo) no crece en 2 ticks → mata huérfanos y vuelve a lanzar el install.
4. Duerme `IntervalMinutes` y repite.
5. **No pregunta nada** al usuario.

---

## Qué debe hacer la IA cuando le pidas “usa safe_downloader”

1. Leer este README.
2. **No** abrir 2 terminales de `pip`/`curl` a la vez.
3. Copiar o invocar `watch_download.ps1` con:
   - ventana = 5 min (o la que indiques),
   - un único script de install **reanudable**,
   - probe de éxito claro,
   - log en el repo destino.
4. Arrancar el watchdog en background / Hidden.
5. Dejar dicho al usuario: dónde está el log y qué significa `DONE OK`.
6. Si corta la sesión del chat, el watchdog **sigue** en el PC.

---

## Archivos en esta carpeta

| Archivo | Rol |
|---------|-----|
| `README.md` | Esta documentación (para humanos e IAs) |
| `watch_download.ps1` | Plantilla de watchdog parametrizable |
| `EXAMPLE_install_job.ps1` | Ejemplo mínimo de “job” idempotente (curl resume + pip) |
| `install_distilgpt2.ps1` | Job real: baja `distilgpt2` (HF / hf-mirror) para el árbitro de SRT |
| `install_nllb.ps1` | Job: NLLB-200 distilled 600M (traducción ES/ZH) |
| `install_faster_whisper_large_v2.ps1` | Job: `Systran/faster-whisper-large-v2` (~3 GB) para WhisperX |
| `install_nllb.ps1` | Job real: `sentencepiece` + `facebook/nllb-200-distilled-600M` para ES/ZH |

---

## Anti-patrones (no hacer)

- Varios agentes instalando el mismo wheel a la vez  
- Confiar solo en `pip` sin resume para un .whl de 2+ GB en red mala  
- Olvidar verificar CUDA vs CPU después de instalar deps que arrastran `torch`  
- Watchdog sin log  
- Pedir “¿sigo?” cada 5 minutos — debe ser autónomo  

---

## Señal de terminado

Buscar en el log:

```text
=== DONE OK ===
```

Si tras horas no aparece: mirar las últimas líneas del log (curl exit, pip FAILED, DNS, stallTicks).
