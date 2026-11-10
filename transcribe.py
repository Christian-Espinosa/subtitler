"""
Transcribe + align with the WhisperX Python API and write the alignment JSON.

Why not `python -m whisperx`: its CLI (3.8.x) writes the JSON with the language
the *alignment* started with ("en" by default), not the one it detected. A
Spanish video without `--language` came out as `"language": "en"`, so the
Spanish track was named `.en.srt` and the "translations" were made from the
wrong source language. Measured 2026-11-09 on a 9 s clip. Here the language
written is the one Whisper detected (or the one you forced).

Usage:
  python transcribe.py audio.wav --out audio.json
  python transcribe.py audio.wav --out audio.json --language es --model small --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def pick_device(device: str = "auto") -> str:
    if device and device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def pick_compute_type(device: str, compute_type: str = "auto") -> str:
    """float16 needs a GPU; on CPU int8 is the fast one that works everywhere."""
    if compute_type and compute_type != "auto":
        return compute_type
    return "float16" if device == "cuda" else "int8"


def result_json(aligned: dict, language: str) -> dict:
    """What `build_subs.py` reads: segments, word_segments and the real language."""
    return {
        "segments": aligned.get("segments") or [],
        "word_segments": aligned.get("word_segments") or [],
        "language": language,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="WhisperX transcribe + word alignment -> JSON")
    p.add_argument("audio", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--model", default="large-v2")
    p.add_argument("--language", default="", help="es|en|zh... Empty = detect")
    p.add_argument("--device", default="auto", help="auto|cuda|cpu")
    p.add_argument("--compute-type", default="auto", help="auto|float16|int8|float32")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=15,
                   help="VAD chunk seconds. WhisperX decodes one text per chunk; with 30 s chunks whole "
                        "sentences were dropped (measured 2026-11-10, FLEURS en: 3 of 8 lost; 15 s: none)")
    args = p.parse_args(argv)

    import time

    import whisperx  # heavy: imported only when actually transcribing

    t0 = time.monotonic()

    device = pick_device(args.device)
    compute_type = pick_compute_type(device, args.compute_type)
    print(f"device={device} compute_type={compute_type} model={args.model}", flush=True)

    audio = whisperx.load_audio(str(args.audio))
    model = whisperx.load_model(args.model, device, compute_type=compute_type,
                                language=args.language or None,
                                vad_options={"chunk_size": args.chunk_size})
    t1 = time.monotonic()
    result = model.transcribe(audio, batch_size=args.batch_size, chunk_size=args.chunk_size,
                              language=args.language or None, print_progress=True)
    t2 = time.monotonic()
    language = args.language or result.get("language") or ""
    if not language:
        print("[ERROR] Whisper did not report a language; pass --language", file=sys.stderr)
        return 1
    print(f"spoken language: {language}", flush=True)
    del model

    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    aligned = whisperx.align(result["segments"], align_model, metadata, audio, device,
                             return_char_alignments=False)
    print(f"[time] load {t1 - t0:.1f}s, transcribe {t2 - t1:.1f}s, align (incl. load) {time.monotonic() - t2:.1f}s",
          flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result_json(aligned, language), ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
