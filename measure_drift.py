"""
Measure whether an SRT is in sync with the audio, using WhisperX word-level
alignment as ground truth.

The alignment JSON tells us exactly when every word is spoken. A well timed cue
starts when speech starts. So we can measure, without comparing text (the SRT
may be a translation):

  - how far each cue start is from the nearest real word onset
  - how many cues start while nobody is speaking
  - how much speech time the subtitle track covers at all

Usage:
  python measure_drift.py --srt "video subs.en.srt" --align _work/_micro.json
  python measure_drift.py --srt new.srt --align a.json --compare old.srt
"""
from __future__ import annotations

# =============================================================================
# CONFIG
# =============================================================================
GOOD_SYNC_SEC = 0.30     # a cue start this close to a word onset is "on time"
PAUSE_SEC = 0.40         # silence longer than this starts a new utterance
# =============================================================================

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


SRT_BLOCK = re.compile(
    r"(\d+)\s*\n"
    r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*\n"
    r"(.*?)(?=\n\n|\Z)",
    re.DOTALL,
)


@dataclass
class Word:
    start: float
    end: float
    text: str


def ts_to_sec(ts: str) -> float:
    ts = ts.replace(",", ".")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_srt(path: Path) -> list[tuple[float, float, str]]:
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    out: list[tuple[float, float, str]] = []
    for m in SRT_BLOCK.finditer(raw.strip() + "\n\n"):
        text = m.group(4).strip()
        if text:
            out.append((ts_to_sec(m.group(2)), ts_to_sec(m.group(3)), text))
    return out


def load_words(path: Path) -> list[Word]:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    words: list[Word] = []
    for seg in data.get("segments", []):
        for w in seg.get("words", []):
            # Alignment can leave a word without timing (e.g. pure punctuation
            # or a token the aligner could not place). Those are skipped.
            if "start" in w and "end" in w:
                words.append(Word(float(w["start"]), float(w["end"]), str(w.get("word", ""))))
    words.sort(key=lambda x: x.start)
    return words


def utterance_onsets(words: list[Word], pause: float) -> list[float]:
    """Start of each speech burst: a word preceded by silence."""
    onsets: list[float] = []
    prev_end = -99.0
    for w in words:
        if w.start - prev_end > pause:
            onsets.append(w.start)
        prev_end = max(prev_end, w.end)
    return onsets


def speech_intervals(words: list[Word], pause: float) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for w in words:
        if merged and w.start - merged[-1][1] <= pause:
            merged[-1] = (merged[-1][0], max(merged[-1][1], w.end))
        else:
            merged.append((w.start, w.end))
    return merged


def nearest(sorted_times: list[float], t: float) -> float:
    """Signed distance from t to the closest time (negative = t is earlier)."""
    if not sorted_times:
        return 0.0
    lo, hi = 0, len(sorted_times) - 1
    if t <= sorted_times[0]:
        return t - sorted_times[0]
    if t >= sorted_times[-1]:
        return t - sorted_times[-1]
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if sorted_times[mid] <= t:
            lo = mid
        else:
            hi = mid
    before, after = sorted_times[lo], sorted_times[hi]
    return t - before if (t - before) <= (after - t) else t - after


def in_speech(intervals: list[tuple[float, float]], t: float) -> bool:
    for s, e in intervals:
        if s - 1e-6 <= t <= e + 1e-6:
            return True
        if s > t:
            break
    return False


def quantiles(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    v = sorted(values)
    return v[len(v) // 4], v[len(v) // 2], v[int(len(v) * 0.9)]


def report(srt: Path, words: list[Word], label: str) -> dict[str, float]:
    cues = parse_srt(srt)
    if not cues or not words:
        print(f"[{label}] faltan datos")
        return {}

    onsets = utterance_onsets(words, PAUSE_SEC)
    word_starts = [w.start for w in words]
    speech = speech_intervals(words, PAUSE_SEC)
    speech_time = sum(e - s for s, e in speech)

    d_word = [nearest(word_starts, c[0]) for c in cues]
    on_time = sum(1 for d in d_word if abs(d) <= GOOD_SYNC_SEC)
    early = sum(1 for d in d_word if d < -GOOD_SYNC_SEC)
    late = sum(1 for d in d_word if d > GOOD_SYNC_SEC)
    abs_d = [abs(d) for d in d_word]
    q1, med, p90 = quantiles(abs_d)

    silent_start = sum(1 for c in cues if not in_speech(speech, c[0]))
    covered = sum(c[1] - c[0] for c in cues)

    n = len(cues)
    print(f"\n===== {label} =====")
    print(f"cues                          : {n}")
    print(f"desfase |cue start - palabra|  : p25 {q1:.2f}s | mediana {med:.2f}s | p90 {p90:.2f}s")
    print(f"cues en hora (<={GOOD_SYNC_SEC:.2f}s)        : {on_time:4d}  {100.0*on_time/n:5.1f}%")
    print(f"cues ADELANTADOS               : {early:4d}  {100.0*early/n:5.1f}%")
    print(f"cues retrasados                : {late:4d}  {100.0*late/n:5.1f}%")
    print(f"cues que empiezan en silencio  : {silent_start:4d}  {100.0*silent_start/n:5.1f}%")
    print(f"habla detectada                : {speech_time/60:.1f} min en {len(onsets)} intervenciones")
    print(f"tiempo cubierto por subtitulos : {covered/60:.1f} min")

    # Los peores adelantos: cues que aparecen en silencio, ordenados por cuanto
    # tiempo pasa hasta que alguien habla.
    offenders: list[tuple[float, float, str]] = []
    for start, _end, text in cues:
        if in_speech(speech, start):
            continue
        nxt = next((s for s, _ in speech if s > start), None)
        if nxt is not None:
            offenders.append((nxt - start, start, text.replace("\n", " ")))
    if offenders:
        offenders.sort(reverse=True)
        print(f"\n  Peores adelantos (aparecen antes de que se hable):")
        for adelanto, start, text in offenders[:5]:
            print(f"   {start:7.1f}s  {adelanto:5.1f}s antes  {text[:58]}")
    return {
        "cues": n,
        "median_off": med,
        "p90_off": p90,
        "on_time_pct": 100.0 * on_time / n,
        "early_pct": 100.0 * early / n,
        "silent_pct": 100.0 * silent_start / n,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Measure SRT sync against word alignment")
    p.add_argument("--srt", required=True, type=Path)
    p.add_argument("--align", required=True, type=Path, help="WhisperX JSON with word timings")
    p.add_argument("--compare", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.align.is_file():
        print(f"[ERROR] no existe: {args.align}", file=sys.stderr)
        return 1
    words = load_words(args.align)
    if not words:
        print("[ERROR] el JSON no trae tiempos por palabra: "
              "se genero con --no_align o con --task translate", file=sys.stderr)
        return 1

    a = report(args.srt, words, args.srt.name)
    if args.compare and args.compare.is_file():
        b = report(args.compare, words, args.compare.name)
        print("\n===== COMPARATIVA =====")
        print(f"{'metrica':<32}{args.srt.name[:20]:>20}{args.compare.name[:20]:>20}")
        for key, fmt in (
            ("cues", "{:.0f}"),
            ("median_off", "{:.2f}s"),
            ("p90_off", "{:.2f}s"),
            ("on_time_pct", "{:.1f}%"),
            ("early_pct", "{:.1f}%"),
            ("silent_pct", "{:.1f}%"),
        ):
            print(f"{key:<32}{fmt.format(a.get(key, 0)):>20}{fmt.format(b.get(key, 0)):>20}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
