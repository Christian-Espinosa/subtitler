"""
Transcribe with faster-whisper alone (CTranslate2 + its Silero VAD in ONNX): no torch,
no WhisperX, no wav2vec2 aligner. Word times come from Whisper's own cross-attention
(`word_timestamps=True`) instead of a separate alignment model.

Why: on CPU the WhisperX path peaks at ~2.1 GB (measured 2026-11-10 with large-v3-turbo
int8), most of it torch and the libraries around the model. Subtitles may not need the
wav2vec2 alignment; this writes the SAME JSON as `transcribe.py` so `build_subs.py`
can cut cues from either, and the two can be compared.

Usage:
  python transcribe_fw.py audio.wav --out audio.json --model large-v3-turbo
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def peak_mb() -> int | None:
    try:
        import resource

        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(kb / 1024) if sys.platform != "darwin" else int(kb / 1024 / 1024)
    except ImportError:
        return None


def to_json(segments, language: str) -> dict:
    """faster-whisper segments -> the alignment JSON `build_subs.py` reads. Pure."""
    segs, words = [], []
    for s in segments:
        ws = [{"word": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3),
               "score": round(w.probability, 3)} for w in (s.words or []) if w.word.strip()]
        segs.append({"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip(), "words": ws})
        words.extend(ws)
    return {"segments": segs, "word_segments": words, "language": language}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="faster-whisper (no torch) -> alignment-style JSON")
    p.add_argument("audio", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--model", default="large-v3-turbo")
    p.add_argument("--language", default="")
    p.add_argument("--device", default="cpu")
    p.add_argument("--compute-type", default="int8")
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--threads", type=int, default=0, help="CPU threads (0 = CTranslate2 default)")
    args = p.parse_args(argv)

    from faster_whisper import WhisperModel

    t0 = time.monotonic()
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type, cpu_threads=args.threads)
    t1 = time.monotonic()
    print(f"[mem] peak after load {peak_mb()} MB", flush=True)
    segments, info = model.transcribe(str(args.audio), language=args.language or None, beam_size=args.beam_size,
                                      word_timestamps=True, vad_filter=True,
                                      vad_parameters={"min_silence_duration_ms": 500})
    segments = list(segments)   # the generator does the work
    t2 = time.monotonic()
    language = args.language or info.language
    print(f"spoken language: {language} (p={info.language_probability:.2f})", flush=True)
    print(f"[mem] peak after transcribe {peak_mb()} MB", flush=True)
    print(f"[time] load {t1 - t0:.1f}s, transcribe+words {t2 - t1:.1f}s", flush=True)
    print(f"torch loaded: {'torch' in sys.modules}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(to_json(segments, language), ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
