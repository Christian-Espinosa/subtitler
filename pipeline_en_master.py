"""
Full pipeline: 3x WhisperX translate -> perplexity arbiter -> EN master -> NLLB (ES/ZH).

Usage:
  python pipeline_en_master.py "D:\\video.mp4"
  python pipeline_en_master.py "D:\\video.mp4" --keep-passes --langs en,es,zh
"""
from __future__ import annotations

# =============================================================================
# CONFIG - edit these (CLI flags override when passed)
# =============================================================================
PYTHON = r"C:\Users\chres\AppData\Local\Programs\Python\Python312\python.exe"

# Hugging Face mirror (china / blocked huggingface.co). Empty = official hub.
HF_ENDPOINT = "https://hf-mirror.com"

# WhisperX passes (audio language forced + --task translate -> English SRTs)
WHISPER_MODEL = "large-v2"
WHISPER_BATCH_SIZE = 8
WHISPER_COMPUTE_TYPE = "float16"
WHISPER_PASS_LANGS = ["es", "en", "zh"]
# Rough realtime factor for ETA (1.0 = as long as the video). RTX 3060 Ti ~0.2-0.4
WHISPER_ETA_RTF = 0.30

# Perplexity arbiter
ARBITER_WINDOW_SEC = 3.0
FLUENCY_MODEL = "distilgpt2"
ARBITER_ETA_SEC_PER_MIN = 8.0  # ~seconds of wall time per minute of media

# Final subtitle languages. Default: Spanish + English. Add "zh" for Chinese.
OUTPUT_SUB_LANGS = ["en", "es"]

# On-screen resegmentation (after arbiter, before NLLB)
RESEGMENT = True
RESEGMENT_MAX_CHARS = 42
RESEGMENT_MAX_LINES = 2
RESEGMENT_MAX_CPS = 18.0
RESEGMENT_MIN_SEC = 1.0
RESEGMENT_MAX_SEC = 6.0

# NLLB (only used for langs other than "en")
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
NLLB_ETA_SEC_PER_MIN = 12.0

KEEP_PASSES = False
# =============================================================================

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARBITER = ROOT / "arbitrate_en_srt.py"
TRANSLATE = ROOT / "translate_srt.py"
RESEGMENT_SCRIPT = ROOT / "resegment_srt.py"


def normalize_path(raw: str | Path) -> Path:
    """Accept paths with or without surrounding quotes (e.g. pasted from Explorer)."""
    s = str(raw).strip().strip('"').strip("'").strip()
    return Path(s).expanduser()


def fmt_dur(sec: float | None) -> str:
    if sec is None or sec < 0 or sec != sec:
        return "?"
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def media_duration_sec(path: Path) -> float | None:
    try:
        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0 and r.stdout.strip():
            return float(r.stdout.strip())
    except (OSError, ValueError):
        pass
    return None


def ensure_hf_env(env: dict[str, str] | None = None) -> dict[str, str]:
    e = dict(os.environ if env is None else env)
    if HF_ENDPOINT:
        e["HF_ENDPOINT"] = HF_ENDPOINT
        # Some stacks still hardcode hub URL; endpoint is the main switch.
    e.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    e.setdefault("PYTHONIOENCODING", "utf-8")
    return e


def banner(title: str) -> None:
    line = "=" * 60
    print(f"\n{line}\n  {title}\n{line}", flush=True)


class Progress:
    def __init__(self, media_sec: float | None, total_steps: int):
        self.media_sec = media_sec
        self.total_steps = total_steps
        self.step = 0
        self.t0 = time.monotonic()
        self.whisper_times: list[float] = []

    def start_step(self, name: str, eta_hint: float | None = None) -> float:
        self.step += 1
        elapsed = time.monotonic() - self.t0
        remaining_steps = self.total_steps - self.step + 1
        # Prefer measured whisper average for remaining whisper-like work
        eta = eta_hint
        if eta is None and self.whisper_times:
            avg = sum(self.whisper_times) / len(self.whisper_times)
            # rough: remaining steps similar cost if still in whisper phase
            eta = avg * remaining_steps
        banner(
            f"PASO {self.step}/{self.total_steps}: {name}\n"
            f"  Tiempo transcurrido : {fmt_dur(elapsed)}\n"
            f"  ETA resto (aprox.)  : {fmt_dur(eta) if eta is not None else 'calculando...'}"
        )
        return time.monotonic()

    def end_step(self, t_start: float, *, is_whisper: bool = False) -> None:
        dt = time.monotonic() - t_start
        if is_whisper:
            self.whisper_times.append(dt)
        print(f"  [OK] Paso {self.step}/{self.total_steps} en {fmt_dur(dt)}", flush=True)

    def whisper_eta(self, remaining_passes: int) -> float:
        if self.whisper_times:
            avg = sum(self.whisper_times) / len(self.whisper_times)
            return avg * remaining_passes
        if self.media_sec:
            return self.media_sec * WHISPER_ETA_RTF * remaining_passes
        return 600.0 * remaining_passes  # fallback 10 min/pass


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=ensure_hf_env(env))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="3x WhisperX translate + arbiter + NLLB to subtitle SRTs"
    )
    p.add_argument("input", type=str, help="Video or audio file (quotes OK)")
    p.add_argument("--model", default=WHISPER_MODEL, help="Whisper model")
    p.add_argument("--batch_size", type=int, default=WHISPER_BATCH_SIZE)
    p.add_argument("--compute_type", default=WHISPER_COMPUTE_TYPE)
    p.add_argument("--window", type=float, default=ARBITER_WINDOW_SEC)
    p.add_argument("--fluency-model", default=FLUENCY_MODEL)
    p.add_argument(
        "--langs",
        default=",".join(OUTPUT_SUB_LANGS),
        help="Output subtitle langs, e.g. en,es or en,es,zh (default from CONFIG)",
    )
    p.add_argument("--nllb-model", default=NLLB_MODEL)
    p.add_argument(
        "--keep-passes",
        action="store_true",
        default=KEEP_PASSES,
        help="Keep intermediate WhisperX pass SRTs",
    )
    p.add_argument(
        "--no-keep-passes",
        action="store_true",
        help="Force delete intermediate passes (overrides CONFIG KEEP_PASSES)",
    )
    p.add_argument("--skip-whisperx", action="store_true", help="Reuse existing _pass_*.srt")
    p.add_argument("--skip-nllb", action="store_true", help="Only produce English master")
    p.add_argument("--skip-resegment", action="store_true", help="Skip on-screen resegmentation")
    p.add_argument("--no-resegment", action="store_true", help="Alias of --skip-resegment")
    args = p.parse_args(argv)

    keep_passes = KEEP_PASSES or args.keep_passes
    if args.no_keep_passes:
        keep_passes = False

    media = normalize_path(args.input)
    try:
        media = media.resolve(strict=False)
    except OSError:
        pass
    if not media.is_file():
        print(f"[ERROR] file not found: {media}", file=sys.stderr)
        print("  Tip: path can include quotes; they are stripped automatically.", file=sys.stderr)
        return 1

    required = [ARBITER, RESEGMENT_SCRIPT, TRANSLATE]
    missing = [p for p in required if not p.is_file()]
    if missing:
        print("[ERROR] missing scripts in repo root:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        return 1

    if HF_ENDPOINT:
        os.environ["HF_ENDPOINT"] = HF_ENDPOINT

    out_langs = [x.strip().lower() for x in args.langs.split(",") if x.strip()]
    if "en" not in out_langs:
        out_langs = ["en"] + out_langs

    need_mt = [L for L in out_langs if L != "en"]
    do_whisper = not args.skip_whisperx
    do_resegment = RESEGMENT and not (args.skip_resegment or args.no_resegment)
    do_nllb = bool(need_mt) and not args.skip_nllb

    n_whisper = len(WHISPER_PASS_LANGS) if do_whisper else 0
    # whisper passes + arbiter + optional resegment + optional NLLB
    total_steps = n_whisper + 1 + (1 if do_resegment else 0) + (1 if do_nllb else 0)

    out_dir = media.parent
    stem = media.stem
    passes = {
        lang: out_dir / f"{stem}._pass_en_from_{lang}.srt" for lang in WHISPER_PASS_LANGS
    }
    wx_tmp = out_dir / f"{stem}.srt"
    master = out_dir / f"{stem} subs.en.srt"
    decisions = out_dir / f"{stem}._arbiter.json"

    media_sec = media_duration_sec(media)
    media_min = (media_sec / 60.0) if media_sec else None

    banner("PIPELINE SUBTITLER")
    print(f"  Entrada     : {media}", flush=True)
    print(f"  Duracion    : {fmt_dur(media_sec)}", flush=True)
    print(f"  Modelo ASR  : {args.model}", flush=True)
    print(f"  Pasadas     : {', '.join(WHISPER_PASS_LANGS)} (translate -> EN)", flush=True)
    print(f"  Salida      : {', '.join(out_langs)}", flush=True)
    print(f"  HF endpoint : {HF_ENDPOINT or '(oficial huggingface.co)'}", flush=True)
    if media_sec and do_whisper:
        rough = media_sec * WHISPER_ETA_RTF * n_whisper
        rough += (media_min or 0) * ARBITER_ETA_SEC_PER_MIN
        if do_nllb:
            rough += (media_min or 0) * NLLB_ETA_SEC_PER_MIN * len(need_mt)
        print(f"  ETA total aprox. (primera vez puede ser mas): {fmt_dur(rough)}", flush=True)
    print(f"  Pasos totales: {total_steps}", flush=True)

    prog = Progress(media_sec, total_steps)

    if do_whisper:
        for i, lang in enumerate(WHISPER_PASS_LANGS):
            dest = passes[lang]
            remaining = n_whisper - i
            t0 = prog.start_step(
                f"WhisperX translate (idioma audio forzado: {lang}) -> ingles",
                eta_hint=prog.whisper_eta(remaining)
                + ((media_min or 0) * ARBITER_ETA_SEC_PER_MIN)
                + (
                    (media_min or 0) * NLLB_ETA_SEC_PER_MIN * len(need_mt)
                    if do_nllb
                    else 0
                ),
            )
            if wx_tmp.exists():
                wx_tmp.unlink()
            cmd = [
                PYTHON,
                "-m",
                "whisperx",
                str(media),
                "--model",
                args.model,
                "--task",
                "translate",
                "--language",
                lang,
                "--output_dir",
                str(out_dir),
                "--output_format",
                "srt",
                "--compute_type",
                args.compute_type,
                "--batch_size",
                str(args.batch_size),
                "--print_progress",
                "True",
            ]
            run(cmd)
            if not wx_tmp.is_file():
                print(f"[ERROR] WhisperX did not create {wx_tmp}", file=sys.stderr)
                return 1
            if dest.exists():
                dest.unlink()
            wx_tmp.rename(dest)
            print(f"  Guardado: {dest.name}", flush=True)
            prog.end_step(t0, is_whisper=True)
    else:
        for lang, dest in passes.items():
            if not dest.is_file():
                print(f"[ERROR] missing pass SRT: {dest}", file=sys.stderr)
                return 1

    t0 = prog.start_step(
        "Arbitro de perplejidad (distilgpt2) -> subs.en.srt",
        eta_hint=(media_min or 5) * ARBITER_ETA_SEC_PER_MIN
        + ((media_min or 0) * NLLB_ETA_SEC_PER_MIN * len(need_mt) if do_nllb else 0),
    )
    arb_cmd = [
        PYTHON,
        str(ARBITER),
        "--out",
        str(master),
        "--window",
        str(args.window),
        "--model",
        args.fluency_model,
        "--json",
        str(decisions),
    ]
    for lang in WHISPER_PASS_LANGS:
        arb_cmd.extend([f"--{lang}", str(passes[lang])])
    run(arb_cmd)
    prog.end_step(t0)

    if not keep_passes:
        for dest in passes.values():
            try:
                dest.unlink(missing_ok=True)
            except OSError:
                pass
        print("  (passes intermedios borrados; usa --keep-passes para conservarlos)", flush=True)

    outputs = [master, decisions]
    if do_resegment:
        t0 = prog.start_step(
            "Resegmentar SRT (lineas / CPS / duracion en pantalla)",
            eta_hint=15.0
            + ((media_min or 0) * NLLB_ETA_SEC_PER_MIN * len(need_mt) if do_nllb else 0),
        )
        run(
            [
                PYTHON,
                str(RESEGMENT_SCRIPT),
                "--in",
                str(master),
                "--out",
                str(master),
                "--max-chars",
                str(RESEGMENT_MAX_CHARS),
                "--max-lines",
                str(RESEGMENT_MAX_LINES),
                "--max-cps",
                str(RESEGMENT_MAX_CPS),
                "--min-sec",
                str(RESEGMENT_MIN_SEC),
                "--max-sec",
                str(RESEGMENT_MAX_SEC),
            ]
        )
        prog.end_step(t0)

    if do_nllb:
        t0 = prog.start_step(
            f"NLLB traduccion EN -> {', '.join(need_mt)}",
            eta_hint=(media_min or 5) * NLLB_ETA_SEC_PER_MIN * len(need_mt),
        )
        run(
            [
                PYTHON,
                str(TRANSLATE),
                "--in",
                str(master),
                "--langs",
                ",".join(need_mt),
                "--model",
                args.nllb_model,
            ]
        )
        for L in need_mt:
            outputs.append(out_dir / f"{stem} subs.{L}.srt")
        prog.end_step(t0)

    banner(f"COMPLETADO en {fmt_dur(time.monotonic() - prog.t0)}")
    print("Archivos:", flush=True)
    for o in outputs:
        print(f"  {o}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] command failed with {e.returncode}", file=sys.stderr)
        raise SystemExit(e.returncode)
