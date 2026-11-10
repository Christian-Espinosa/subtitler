"""
Aligned subtitle pipeline: transcribe -> align -> cut by word -> translate.

Replaces the 3-pass + perplexity arbiter flow of `pipeline_en_master.py`.

Why this one instead of that one:
  * `--task translate` makes WhisperX skip forced alignment (see its
    transcribe.py: "translation cannot be aligned"). Without alignment you get
    one cue per 30 s VAD chunk and every cue boundary inside it has to be
    guessed from character counts. That is what made subtitles drift.
  * Forcing a language Whisper does not hear makes it hallucinate over music
    and silence, and a fluency-based arbiter happily picks those hallucinations
    because they read like good English.

Here Whisper transcribes in the spoken language, WhisperX aligns word by word,
cues are cut on real word boundaries, and translation happens afterwards on the
text only, so timings never move.

Usage:
  python pipeline_aligned.py "D:\\Videos\\video.mp4"
  python pipeline_aligned.py "video.mp4" --language es --langs es,en
  python pipeline_aligned.py "video.mp4" --skip-translate --keep-work
"""
from __future__ import annotations

# =============================================================================
# CONFIG — edit these defaults (CLI flags override)
# =============================================================================
# The Python running this script runs every step: no path to a particular PC.
# Hugging Face mirror: set HF_ENDPOINT in the environment (or in
# subtitler.local.bat on Windows) when huggingface.co is blocked, e.g. in China.

WHISPER_MODEL = "auto"                      # auto = large-v2 on GPU, small on CPU
WHISPER_MODEL_GPU = "large-v2"
WHISPER_MODEL_CPU = "small"
WHISPER_BATCH_SIZE = 8
WHISPER_COMPUTE_TYPE = "auto"               # auto = float16 on GPU, int8 on CPU
DEVICE = "auto"                             # auto | cuda | cpu
SPOKEN_LANGUAGE = ""                        # "" = let Whisper detect it

OUTPUT_SUB_LANGS = ["es", "en"]             # tracks to write
TRANSLATOR = "llm"                          # llm (NVIDIA free -> DeepSeek Flash) | nllb (CC-BY-NC, opt-in)
NLLB_MODEL = "facebook/nllb-200-distilled-600M"

# Screen limits, passed to build_subs.py
MAX_CHARS_PER_LINE = 42
MAX_LINES = 2
MAX_CUE_SEC = 6.0
MIN_CUE_SEC = 1.0

KEEP_WORK = False                           # keep _work/ (audio + alignment JSON)
# =============================================================================

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable
ROOT = Path(__file__).resolve().parent
TRANSCRIBE = ROOT / "transcribe.py"
BUILD_SUBS = ROOT / "build_subs.py"
TRANSLATE = ROOT / "translate_srt.py"
ANALYZE = ROOT / "analyze_srt.py"


def normalize_path(raw: str) -> Path:
    return Path(str(raw).strip().strip('"').strip("'")).expanduser()


def fmt_dur(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def run(cmd: list[str], step: str) -> None:
    print(f"\n>>> {step}", flush=True)
    t0 = time.monotonic()
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(f"[ERROR] fallo en: {step}")
    print(f"    hecho en {fmt_dur(time.monotonic() - t0)}", flush=True)


def media_duration(path: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return float(out)
    except Exception:  # noqa: BLE001
        return 0.0


def hub_reachable(url: str = "https://huggingface.co", timeout: float = 3.0) -> bool:
    import urllib.request

    try:
        urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=timeout)
        return True
    except Exception:  # noqa: BLE001 - any failure means "not reachable from here"
        return False


def ensure_hub(env: dict | None = None, reachable=None) -> str:
    """Where models come from. HF_ENDPOINT if set (a mirror, e.g. in subtitler.local.bat);
    else huggingface.co; if it does not answer (blocked network), offline mode with the
    models already downloaded, so a PC that has them keeps working without a mirror."""
    import os

    env = os.environ if env is None else env
    if env.get("HF_ENDPOINT") or env.get("HF_HUB_OFFLINE"):
        return f"models from {env.get('HF_ENDPOINT') or 'local cache'}"
    if (reachable or hub_reachable)():
        return "models from huggingface.co"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    return ("huggingface.co does not answer: offline mode, using the models already downloaded. "
            "If one is missing, set HF_ENDPOINT to a mirror (subtitler.local.bat on Windows).")


def resolve_model(model: str, device: str) -> str:
    """`auto`: large-v2 where there is a GPU, small on CPU (large-v2 on CPU is ~10x slower)."""
    if model and model != "auto":
        return model
    from transcribe import pick_device

    return WHISPER_MODEL_GPU if pick_device(device) == "cuda" else WHISPER_MODEL_CPU


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Aligned subtitle pipeline")
    p.add_argument("media", help="Video or audio file")
    p.add_argument("--language", default=SPOKEN_LANGUAGE,
                   help="Spoken language (es|en|zh). Empty = auto-detect")
    p.add_argument("--langs", default=",".join(OUTPUT_SUB_LANGS),
                   help="Subtitle tracks to write, e.g. es,en,zh")
    p.add_argument("--model", default=WHISPER_MODEL)
    p.add_argument("--batch-size", type=int, default=WHISPER_BATCH_SIZE)
    p.add_argument("--compute-type", default=WHISPER_COMPUTE_TYPE)
    p.add_argument("--device", default=DEVICE, help="auto|cuda|cpu")
    p.add_argument("--max-chars", type=int, default=MAX_CHARS_PER_LINE)
    p.add_argument("--max-sec", type=float, default=MAX_CUE_SEC)
    p.add_argument("--skip-translate", action="store_true")
    p.add_argument("--translator", default=TRANSLATOR, choices=["llm", "nllb"],
                   help="llm (default; needs NVIDIA_API_KEY or DEEPSEEK_API_KEY) or nllb (local, non-commercial)")
    p.add_argument("--skip-whisperx", action="store_true",
                   help="Reuse the alignment JSON from a previous run")
    p.add_argument("--keep-work", action="store_true", default=KEEP_WORK)
    args = p.parse_args(argv)

    print(ensure_hub(), flush=True)
    media = normalize_path(args.media)
    if not media.is_file():
        print(f"[ERROR] no existe: {media}", file=sys.stderr)
        return 1

    out_dir = media.parent
    work = out_dir / "_work"
    work.mkdir(exist_ok=True)
    audio = work / f"{media.stem}.wav"
    align_json = work / f"{media.stem}.json"
    targets = [x.strip().lower() for x in args.langs.split(",") if x.strip()]

    duration = media_duration(media)
    print(f"Fuente   : {media.name}")
    print(f"Duracion : {fmt_dur(duration)}")
    print(f"Pistas   : {', '.join(targets)}")

    total = time.monotonic()

    if not args.skip_whisperx:
        # 1) Audio a 16 kHz mono: Whisper trabaja asi, y evita releer el video
        # entero en cada intento.
        run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(media),
             "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(audio)],
            "1/4  Extraer audio",
        )

        # 2) Transcribir EN EL IDIOMA HABLADO (no translate) para que WhisperX
        # alinee palabra por palabra.
        # (Python API, not `python -m whisperx`: its CLI writes the wrong language, see transcribe.py)
        model = resolve_model(args.model, args.device)
        cmd = [PYTHON, str(TRANSCRIBE), str(audio), "--out", str(align_json),
               "--model", model, "--device", args.device,
               "--compute-type", args.compute_type, "--batch-size", str(args.batch_size)]
        if args.language:
            cmd += ["--language", args.language]
        run(cmd, f"2/4  WhisperX transcribe + alineacion ({args.language or 'auto'})")

    if not align_json.is_file():
        print(f"[ERROR] no se genero la alineacion: {align_json}", file=sys.stderr)
        return 1

    spoken = json.loads(align_json.read_text(encoding="utf-8")).get("language") or args.language
    if not spoken:
        print("[ERROR] no se pudo determinar el idioma hablado", file=sys.stderr)
        return 1
    print(f"\nIdioma hablado detectado: {spoken}")

    # 3) Cortar en fronteras reales de palabra.
    native_srt = out_dir / f"{media.stem} subs.{spoken}.srt"
    run(
        [PYTHON, str(BUILD_SUBS), "--align", str(align_json), "--out", str(native_srt),
         "--max-chars", str(args.max_chars), "--max-sec", str(args.max_sec),
         "--min-sec", str(MIN_CUE_SEC), "--max-lines", str(MAX_LINES)],
        "3/4  Construir subtitulos desde la alineacion",
    )

    # 4) Traducir el texto, nunca los tiempos.
    others = [t for t in targets if t != spoken]
    if others and not args.skip_translate:
        run(
            [PYTHON, str(TRANSLATE), "--in", str(native_srt), "--src", spoken,
             "--langs", ",".join(others), "--translator", args.translator, "--model", NLLB_MODEL],
            f"4/4  Traducir {spoken} -> {', '.join(others)}",
        )
    else:
        print("\n>>> 4/4  Traduccion omitida")

    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"\nTrabajo conservado en {work}")

    print(f"\nTotal: {fmt_dur(time.monotonic() - total)}")
    print("\nSubtitulos escritos:")
    for lang in targets:
        srt = out_dir / f"{media.stem} subs.{lang}.srt"
        if srt.is_file():
            print(f"  {srt}")
    print(f"\nPara medir la calidad:\n"
          f'  python "{ANALYZE}" --in "{native_srt}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
